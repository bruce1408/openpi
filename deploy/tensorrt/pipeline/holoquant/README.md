# 给 HoloQuant NPU 仿真的参考导出

本目录属于 **OpenPI**。只做一件事：用本仓库的真实 π₀.₅、ModelOpt MTQ、（可选）TensorRT，
写出 json / npz，给旁边的 HoloQuant 当**精度尺子**。

```text
本目录（OpenPI .venv）  --写出文件-->  Outputs/HoloQuant/pi05_libero/openpi_export/
HoloQuant 的 02 / 03 / 04            --只读文件-->  建静态图、遍历图、调度 NPU 算子
```

两边**不要互相 import**。本仓库是 `.venv`（Python 3.11 / PyTorch 2.7）；
HoloQuant 是 conda `torch212`（PyTorch 2.12）。混装会把环境装坏。

权重不在这里导出。HoloQuant 的 02 自己读：

```text
/share_data/public/models/openpi/pi05_libero_pytorch/model.safetensors
```

旁边的 `../fp8/` 是 TensorRT 自己的 ONNX / engine 流水线，和这里职责不同。
本目录不编 engine，只**消费**已经编好的 engine（加 `--trt` 时）。

---

## 1. 这份导出在量什么（先分清 bf16 和 FP8）

π₀.₅ 官方 PyTorch 权重本来就是 **bf16**。脚本先原样跑一遍，留下参考动作（L0）。
**量化目标是 FP8 E4M3，不是 bf16，也不是 FP16。**

| 台阶 | 是什么 | 谁写 |
|---|---|---|
| L0 | 官方 bf16 `sample_actions` 的动作和中间张量 | 本脚本必做 |
| L1 | ModelOpt 把 Prefix / Denoise 的 Linear 按 **FP8 E4M3** 假量化 | 本脚本必做（MTQ） |
| L1-TRT | 同一批样本再跑 TensorRT `mtqfp8` engine | 加 `--trt` 才有 |
| L2 | HoloQuant `03` 遍历静态图、调度自研 NPU 算子 | 不在本目录 |

可以想成考试：bf16 是标准答案，FP8 是你要练的答题方式。没有 L0，后面没法判断 NPU 偏了多少。

HoloQuant 整网仿真要的就是这套 **FP8 尺子**。不要把「bf16 + MTQ」读成「整网用 bf16」。

---

## 2. 三段各自量化没有

一次 `infer` 可以想成三段。量化范围和 TensorRT 当前方案对齐，但 HoloQuant 建图时对 Vision 更激进一点：

| 段 | 官方 MTQ / `--trt` 对照 | HoloQuant 02 静态图 |
|---|---|---|
| **Vision**（SigLIP + 投影） | **不量化**，仍按 bf16 算。ModelOpt 关掉了 `vision_tower`、`multi_modal_projector` | Linear（QKV/O、MLP、投影）打成 **FP8 E4M3**；patch Conv、LayerNorm 仍是 FP32 |
| **Prefix**（Gemma-2B） | Linear **FP8 E4M3**（`FP8_DEFAULT_CFG`，和 `mtqfp8` 同一配方） | 同样用 `fp8_amax.json` 的 `mtq` 段 |
| **Denoise**（Gemma-300M expert） | Linear **FP8 E4M3**，同上 | 同上 |

Vision **不是 FP16**。官方不量化它，是因为 MTQ 配方故意关掉了；脚本仍用 forward hook 统计每个 Vision Linear 的 `max|x|`，写进 `fp8_amax.json` 的 `vision` 段，给 HoloQuant 的 02 打包用。

注意力的 Q@K / attn@V、Softmax、RoPE、time MLP、`action_in_proj` / `action_out_proj`：**官方也不量化**。HoloQuant 里这些走 FP32 算子。

看产物认字段：`fp8_amax.json` 里 `mtq` 是 Prefix/Denoise（真 MTQ），`vision` 是 hook 统计（不是 ModelOpt）。

---

## 3. 怎么跑

在 **openpi 仓库根**跑下面的 `.sh`。不要用 Cursor 的「Run」：它会选错解释器（系统 python 或 conda `torch212`），`import openpi` 会失败。

```bash
cd /share_data/bruce/workspace/ai/openpi

# 推荐：L0 bf16 参考 + Prefix/Denoise 的 FP8 MTQ + TensorRT FP8 对照
./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh --trt

# 只要 L0 + MTQ，不要 TensorRT
./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh
```

`--trt` 会设 `OPENPI_TRT_PRECISION=mtqfp8`，跑的是已经编好的 **FP8** engine，不是再跑一遍 bf16。
本机还没有 `deploy/tensorrt/artifacts/` 下的 plan 时，不要加 `--trt`，或先走 `../fp8/` 把 engine 编出来。

人在 HoloQuant 目录时，可以转发（它只 `exec` 到本脚本）：

```bash
cd /share_data/bruce/workspace/Quant_Repo/HoloQuant
./holoq/samples/pi05_npu_sim/pi05_full/run_01.sh --trt
```

换机器：`OPENPI_ROOT=/path/to/openpi ./holoq/samples/.../run_01.sh --trt`。

常用参数（加在 `.sh` 后面即可）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--trt` | 关 | 同一批样本再跑 TensorRT `mtqfp8` |
| `--out-dir` | 见下一节 | 产物目录；也可用环境变量 `HOLOQUANT_EXPORT_DIR` |
| `--num-samples` | 4 | 评测条数（和 RESULTS.md 同一把尺子） |
| `--num-calib` | 16 | MTQ 校准条数（和 run_step2/3 相同） |
| `--device` | `cuda:0` | 需要 GPU |
| `--obs` | `random` | `random`：官方 LIBERO 随机观测；`video`：视频第一帧 |

需要加载完整模型和校准，会比本地冒烟久很多，请等它跑完。

---

## 4. 产物写到哪里、里面是什么

默认：

```text
/share_data/bruce/workspace/Outputs/HoloQuant/pi05_libero/openpi_export/
  config.json
  constants.npz
  fp8_amax.json
  samples/sample_000.npz … sample_003.npz
  summary.json
```

| 文件 | 内容 |
|---|---|
| `config.json` | 层数、头数、eps、RoPE、mask 值等，02 建图用 |
| `constants.npz` | RoPE `inv_freq`、时间编码频率表 |
| `fp8_amax.json` | `mtq`：Prefix/Denoise 每个 Linear 的 input/weight amax；`vision`：hook 统计 |
| `samples/sample_XXX.npz` | 输入 + `ref_*`（L0 bf16）+ `fq_actions`（L1 MTQ）+ 可选 `trt_actions` |
| `summary.json` | L1 / TRT 相对 bf16 的动作 RMSE |

成功标志：上面四个文件都在，且 `samples/` 里有 `sample_000.npz`。
`summary.json` 应是**官方**导出，不要是本地假参考。认这几项：

- `obs` 一般是 `random` 或 `video`，**不是** `synthetic_tensors`
- 有 `fq_rmse_mean`（MTQ 相对 bf16）
- 加了 `--trt` 时 `trt_rmse_mean` 有数，不是 `null`

目录里如果已经是别的导出，先把旧目录改名挪走再跑，避免和官方尺子混在一起。
本脚本默认会写入上述路径；HoloQuant 的 02/03 默认也读这里。

---

## 5. 导出之后怎么接到 NPU 仿真

本目录只负责 01。后面在 **HoloQuant / `torch212`**：

```bash
conda activate torch212
cd /share_data/bruce/workspace/Quant_Repo/HoloQuant

# 整网建图（不要加 --smoke，否则只有 1 层）
python holoq/samples/pi05_npu_sim/pi05_full/02_build_static_graphs.py

# 遍历图 + 调度自研算子（mixed：量化/SFU 走芯片，乘加走 torch）
python holoq/samples/pi05_npu_sim/pi05_full/03_run_npu_sim.py --backend mixed

python holoq/samples/pi05_npu_sim/pi05_full/04_compare.py --backend mixed
```

完整命令和成功标志见 HoloQuant 仓库 `Docs/npu仿真与π05整网.md` 第 11 节。

`03 --backend ba2701` 是逐位 MMA，整网要几十小时，不要用来扫精度。

---

## 6. 本目录有什么

| 文件 | 作用 |
|---|---|
| `export_npu_reference.py` | 真正干活：加载模型、录 L0、MTQ 校准、可选 TRT |
| `export_npu_reference.sh` | 选对本仓库 `.venv`，设好 `PYTHONPATH` 和 TensorRT 动态库 |

不要直接 `python export_npu_reference.py`（除非已经在 `.venv` 且设好 `PYTHONPATH`）。日常用 `.sh`。

---

## 7. 不要这样做

| 做法 | 结果 |
|---|---|
| 在 `torch212` 或 Cursor「Run」里跑本脚本 | import OpenPI / MTQ 失败，或选错 torch |
| 把「bf16 + MTQ」理解成整网用 bf16 | 量化目标其实是 FP8 E4M3 |
| 以为 Vision 是 FP16 | 官方不量化（bf16 算）；HoloQuant 图里 Linear 是 FP8 |
| 用合成输入 / 本地假量化当精度尺子 | 那不是本脚本；NPU 精度分析必须用这里的官方导出 |
| `02`/`03` 加 `--smoke` 当整网精度 | 图只有 1 层 / 只跑 1 步，`03` 不对比整网动作 |
