# OpenPI π₀.₅-LIBERO TensorRT FP8 使用说明

本文说明如何在 **`ai/openpi/` 目录内** 把 π₀.₅-LIBERO 拆成三个子图、做 FP8 量化、再用 LIBERO 评测。  
**不要用 `openpi_lz/`**，那套是另一条 TensorRT 部署（融合 LLM engine），不是本仓库这条。

相关文件：

| 文件 | 作用 |
|------|------|
| `deploy/tensorrt/pipeline/fp8/experiments/run_step3_mtq_export.sh` | **当前方案**：MTQ 后再导出 `*_mtqfp8.plan` |
| `deploy/tensorrt/pipeline/fp8/experiments/RESULTS.md` | 精度恢复实验记录（S1–S3 + LIBERO 四套） |
| `deploy/tensorrt/runtime/fp8_policy.py` | 评测时的混合推理（3 个 engine + Python 10 步 Euler） |
| `scripts/run_libero_fp8_smoke.sh` | 1 个任务 × 2 条 episode 冒烟（默认 mtqfp8 三段全开） |
| `bf6_run_eval.sh` | 8 卡评测；`BACKEND=fp8` |
| `deploy/tensorrt/pipeline/fp8/run_all.sh` | **旧** ONNX PTQ 流水线 01→05，Prefix/Denoise 不准 |
| `deploy/tensorrt/pipeline/fp8/03_quantize_fp8.py` | **旧** 图上 PTQ（猜 Linear `MatMul`），不要当部署路径 |
| `deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh` | 给 HoloQuant NPU 仿真导出 config / amax / 参考动作（只写 json/npz） |

权重用已经转好的 PyTorch 文件：

```text
/share_data/public/models/openpi/pi05_libero_pytorch/model.safetensors
```

---

## 1. 这套 FP8 在做什么（先建立直觉）

π₀.₅ 每做一次 `infer()`，可以想成：

1. **Vision（SigLIP）**：三张 224×224 图 → 每张 256 个 2048 维 token。
2. **Prefix Gemma-2B**：把图像 token 和语言 token 拼成 968 个 token，过 18 层 Transformer，得到 **KV cache**。
3. **Denoise expert（Gemma-300M）**：在这份 KV 上循环 10 次（flow matching 的 Euler 步），每次吐出动作速度，再用 Python 做 `x ← x + dt * v`。

我们把 1/2/3 分别导出成 ONNX，再 FP8 量化、编成 TensorRT engine。  
**故意留在 Python 里的**（算量很小，形状还随时间变）：

- 语言 embedding（词表查找）
- 时间 MLP（给 adaRMS 用的条件）
- `action_in_proj` / `action_out_proj`
- Euler 积分、动作反归一化

**当前可用方案（2026-10-13）是 MTQ 后再导出**，不是 `03_quantize_fp8.py` 那条 ONNX PTQ。

- Prefix / Denoise：`modelopt.torch.quantization` 给 `nn.Linear` 挂 FP8，再导出 `{prefix,denoise}_mtqfp8.onnx` → `*_mtqfp8.plan`。
- Vision：仍用已经过关的 Linear FP8 engine（`vision_mtqfp8.plan` 指向 `vision_fp8.plan`）。
- 注意力 Q@K / attn@V、Softmax、RoPE、Conv `patch_embedding`、AdaRMS dense、time MLP、动作投影：**不量化**。
- 旧 ONNX PTQ（在图上猜 Linear `MatMul`）Prefix RMSE ~0.26，LIBERO spatial **0%**。不要再用 `OPENPI_TRT_PRECISION=fp8` 跑三段全开。

实验记录：`deploy/tensorrt/pipeline/fp8/experiments/RESULTS.md`。

TensorRT 11 是 **strongly-typed**：ONNX 里长什么样，engine 就按什么精度跑。  
所以量化脚本要先插入 FP8 的 Q/DQ 节点；**不要**给 `trtexec` 加 `--fp8`（11.x 已经没有这个开关）。

8 卡评测仍然是 **把 episode 均分**，不是把一个模型切到 8 张卡上。

### 模型拆分：哪一块在 FP8，哪一块不是

一次 `infer()` 可以拆成下面这些零件。**FP8 只作用在 Linear 权重矩阵乘法上**，不会把 10 步 Euler 循环本身量化掉。

| 零件 | 在哪跑 | 当前 mtqfp8 是否 FP8 | 说明 |
|------|--------|----------------------|------|
| SigLIP `patch_embedding`（Conv） | Vision engine | 否 | 保持 fp16 |
| SigLIP 其余 Linear | Vision engine | **是** | `vision_fp8.plan`（`vision_mtqfp8.plan` 指向它） |
| 语言 embedding（词表查找） | Python | 否 | 算量小 |
| Prefix Gemma-2B：每层 `q/k/v/o_proj` + MLP `gate/up/down` | Prefix engine | **是** | 18 层 × 7 个 Linear |
| Prefix 注意力 Q@K / attn@V、Softmax、RoPE | Prefix engine | 否 | 保持 fp16 |
| Prefix AdaRMS dense | Prefix engine | 否 | 导出成 `Gemm`，未量化 |
| Denoise expert Gemma-300M：同样 18×7 Linear | Denoise engine | **是** | 每 Euler 步调用一次同一份 engine |
| Denoise 注意力 / Softmax / RoPE / AdaRMS | Denoise engine | 否 | 与 Prefix 相同策略 |
| `embed_suffix`：`action_in_proj` + time MLP | Python | 否 | 每步都跑，形状随时间变 |
| `action_out_proj` | Python | 否 | hidden → 动作速度 |
| 10 步 `while` + `x_t = x_t + dt * v_t` | Python | 否 | 控制流，不是算子，无法 FP8 |
| 动作反归一化 | Python | 否 | |

### Denoise 要不要融进 LLM？还是 10 次循环吗？

**精度上不需要融合。** Prefix 和 Denoise 是两套权重（2B vs 300M expert），现在拆成两个 engine 已经过关：全开 RMSE 0.011，LIBERO 平均 96.95%。`openpi_lz` 把 Prefix+Denoise 融成一个 LLM engine，主要是为了少一次跨引擎调度、压延迟，不是为了把成功率拉回来。

**现在仍然是 10 次循环。** `Fp8ActionSampler.__call__` 里 `num_steps=10`，`while time >= -dt/2` 每步：Python `embed_suffix` → Denoise TRT（expert Linear 是 FP8）→ Python `action_out_proj` → Euler。FP8 量化的是循环**里面**那一层层 Linear，不是把循环展开成一张图。

---

## 2. 环境（一般已就绪）

```bash
cd /share_data/bruce/workspace/ai/openpi
```

需要：

- 项目虚拟环境 `.venv/`（`uv`，**不要**用 conda 的 python 做导出/评测）
- PyTorch **2.7.x + cu128**（RTX 5090 的 sm_120）
- `transformers==4.53.2` 且已拷入 `transformers_replace`
- TensorRT：`/share_data/bruce/software/TensorRT-11.2.1.2`
- ModelOpt：MTQ 导出（当前方案）和旧的第 03 步都需要；conda 环境 `torch270_128` 里通常有 `nvidia-modelopt`

确认 GPU 空闲后再跑（不要去杀别人的进程）：

```bash
nvidia-smi
```

---

## 3. 具体执行过程

**当前部署不要跑 `run_all.sh`。** 那是旧 ONNX PTQ。engine 已经编好：

```text
deploy/tensorrt/artifacts/engines/vision_mtqfp8.plan   # 指向 vision_fp8.plan
deploy/tensorrt/artifacts/engines/prefix_mtqfp8.plan
deploy/tensorrt/artifacts/engines/denoise_mtqfp8.plan
```

若要从头再编 mtqfp8：

```bash
bash deploy/tensorrt/pipeline/fp8/experiments/run_step3_mtq_export.sh
```

下面 01–04 是旧路径，只留给对照实验。

### 1）导出三个 FP16 ONNX

```bash
.venv/bin/python deploy/tensorrt/pipeline/fp8/01_export_onnx.py \
  --checkpoint-dir /share_data/public/models/openpi/pi05_libero_pytorch \
  --device cuda:0
```

成功后会看到：

```text
deploy/tensorrt/artifacts/onnx/vision_fp16.onnx
deploy/tensorrt/artifacts/onnx/prefix_fp16.onnx      # 外加 .data（权重大于 2GB）
deploy/tensorrt/artifacts/onnx/denoise_fp16.onnx
deploy/tensorrt/artifacts/onnx/export_status.json
```

脚本会先用随机输入对比「官方 DynamicCache」和「我们的纯 Tensor 实现」，`max|Δ|` 应在 1e-2 量级或更小（bf16 噪声）。

只导出某一段（例如 prefix 失败时先保住 vision）：

```bash
.venv/bin/python deploy/tensorrt/pipeline/fp8/01_export_onnx.py --segments vision
```

### 2）抓 FP8 校准数据

```bash
.venv/bin/python deploy/tensorrt/pipeline/fp8/02_dump_calib.py \
  --num-calib 32 --device cuda:0
```

它会走和评测相同的 tokenizer / 归一化。如果 `data/libero/videos_libero_spatial_pytorch/` 里有视频，会用第一帧当图像，分布更接近 LIBERO。  
默认把 denoise 的 **10 个 Euler 步**都留下做校准（`--no-denoise-every-step` 可退回只留 3 步）。

产出在 `deploy/tensorrt/artifacts/calib/*.npy`。

### 3）ModelOpt FP8 PTQ

openpi 的 `.venv` 里未必有 `nvidia-modelopt`。`run_all.sh` 会自动找 conda。手动跑：

```bash
# 只看将量化/排除哪些节点（不写 FP8 ONNX）
conda run -n torch270_128 python deploy/tensorrt/pipeline/fp8/03_quantize_fp8.py --dry-run

# 哪一个 python 能 import modelopt 就用哪一个
conda run -n torch270_128 python deploy/tensorrt/pipeline/fp8/03_quantize_fp8.py
```

这一步是 **旧的 ONNX 图上 PTQ**（只量化被认成 Linear 的 `MatMul`）。实测 Prefix/Denoise 动作 RMSE ~0.26，spatial 0/500。  
**不要把它当成当前部署路径。** 当前 engine 来自 `experiments/run_step3_mtq_export.sh`。

产出：

```text
deploy/tensorrt/artifacts/onnx/vision_fp8.onnx
deploy/tensorrt/artifacts/onnx/prefix_fp8.onnx
deploy/tensorrt/artifacts/onnx/denoise_fp8.onnx
```

### 4）编 TensorRT engine

```bash
bash deploy/tensorrt/pipeline/fp8/04_build_engines.sh
```

产出：

```text
deploy/tensorrt/artifacts/engines/vision_fp8.plan
deploy/tensorrt/artifacts/engines/prefix_fp8.plan
deploy/tensorrt/artifacts/engines/denoise_fp8.plan
```

Prefix（Gemma-2B）编译可能要十几分钟，属正常。

### 5）对照 PyTorch 动作 RMSE

同一份高斯噪声下比 `sample_actions` 的输出（归一化后的 32 维动作，尚未反归一化）。脚本默认已是 `mtqfp8`：

```bash
.venv/bin/python deploy/tensorrt/pipeline/fp8/05_compare_actions.py \
  --device cuda:0 --segments vision,prefix,denoise --precision mtqfp8
```

过关线：RMSE < 0.02。mtqfp8 三段全开实测 **0.011**。旧 ONNX PTQ（`--precision fp8`）是 ~0.26，不要当新结果。

### 6）LIBERO 冒烟（1 任务 × 2 episode）

冒烟脚本默认已是 **mtqfp8 三段全开**：

```bash
bash scripts/run_libero_fp8_smoke.sh
# 日志里应看到 TRT engines (mtqfp8): vision=True prefix=True denoise=True
```

8 卡、每任务 50 次（2026-10-13 已跑完四套）：

```bash
BACKEND=fp8 \
OPENPI_TRT_PRECISION=mtqfp8 \
OPENPI_FP8_SEGMENTS=vision,prefix,denoise \
FP8_ARTIFACT_DIR=/share_data/bruce/workspace/ai/openpi/deploy/tensorrt/artifacts \
CHECKPOINT_DIR=/share_data/public/models/openpi/pi05_libero_pytorch \
SUITE=libero_spatial \
TRIALS=50 \
RUN_TAG=mtqfp8 \
bash bf6_run_eval.sh
```

| Suite | 官方 π₀.₅ | mtqfp8 |
|-------|-----------|--------|
| spatial | 98.8% | **98.8%**（494/500） |
| object | 98.2% | **98.6%**（493/500） |
| goal | 98.0% | **96.8%**（484/500） |
| libero_10 | 92.4% | **93.6%**（468/500） |
| 平均 | 96.85% | **96.95%** |

object / goal / libero_10 把 `SUITE=` 换掉即可，仍用同一套 `*_mtqfp8.plan`。

---

## 4. 某一步失败时怎么办

| 现象 | 含义 | 下一步 |
|------|------|--------|
| `01` 的 prefix/denoise ONNX 导出报错 | AdaRMS / RoPE / 注意力里有 ONNX 不认的算子 | 先 `--segments vision`，评测会自动 **Vision=TRT，其余=PyTorch** |
| `03` `No module named modelopt` | 当前解释器没装 ModelOpt | 换 `QUANT_PY=` 指向 conda `torch270_128` |
| `04` `trtexec` 找不到 | TensorRT 没进 PATH | `TRT_ROOT=/share_data/bruce/software/TensorRT-11.2.1.2` |
| `05` `No module named tensorrt` | `.venv` 没有 TensorRT Python 绑定 | 把 TensorRT 自带的 wheel 装进 `.venv`，或设 `PYTHONPATH` |
| 冒烟成功率 0（三段全开） | 多半加载了旧 `*_fp8.plan`（ONNX PTQ，RMSE~0.26） | 设 `OPENPI_TRT_PRECISION=mtqfp8`，日志须出现 `engines (mtqfp8): vision=True prefix=True denoise=True` |
| `05` Prefix RMSE 是 NaN | fp16 attention mask 溢出成 `-inf` | runtime 已用 `_engine_attention_mask` 收到 `-1e4`；确认跑的是当前 `fp8_policy.py` |
| 冒烟成功率 0 且精度已是 mtqfp8 | 形状/dtype 接错，或 KV cache 对不齐 | 先看 `05 --precision mtqfp8` 的 RMSE |

Hybrid 回退是写进 `Fp8Policy` 的：缺哪个 `.plan` 就用 PyTorch 算哪一段，启动日志会打印 `vision=True/False`。

---

## 5. 和 bf16 评测的关系

- JAX bf16、PyTorch bf16 已经在 `LIBERO_BF16_EVAL.md` 里跑完，作为精度上限。
- 本流程 **不再经过 policy server**，仍然是 `examples/libero/main_local.py` 进程内 `policy.infer()`。
- 加 `--args.backend fp8` 后，同一套 LIBERO 循环改走 TensorRT。
