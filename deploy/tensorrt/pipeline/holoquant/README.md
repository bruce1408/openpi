# 给 HoloQuant NPU 仿真的参考导出

本目录属于 **OpenPI**。只做一件事：用本仓库的真实 π₀.₅、ModelOpt MTQ、（可选）TensorRT，
写出 json / npz，给旁边的 HoloQuant 当**精度尺子**。

```text
本目录（OpenPI .venv）  --写出文件-->  Outputs/HoloQuant/pi05_libero/openpi_export/
HoloQuant 的 01 / 02 / 03            --只读文件-->  建静态图、遍历图、调度 NPU 算子
```

两边**不要互相 import**。本仓库是 `.venv`（Python 3.11 / PyTorch 2.7）；
HoloQuant 是 conda `torch212`（PyTorch 2.12）。混装会把环境装坏。

权重不在这里导出。HoloQuant 的 `01_build_static_graphs.py` 自己读 checkpoint 里的 `model.safetensors`。
本脚本找权重的顺序：`--checkpoint-dir` → `OPENPI_CHECKPOINT_DIR` → 仓库内 `checkpoints/pi05_libero_pytorch` → 若本机已有 `deploy.tensorrt.common.DEFAULT_CHECKPOINT` 也可用。

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
| L2 | HoloQuant `02_run_npu_sim.py` 遍历静态图、调度自研 NPU 算子 | 不在本目录 |

可以想成考试：bf16 是标准答案，FP8 是你要练的答题方式。没有 L0，后面没法判断 NPU 偏了多少。

HoloQuant 整网仿真要的就是这套 **FP8 尺子**。不要把「bf16 + MTQ」读成「整网用 bf16」。

---

## 2. 三段各自量化没有

一次 `infer` 可以想成三段。量化范围和 TensorRT 当前方案对齐，但 HoloQuant 建图时对 Vision 更激进一点：

| 段 | 官方 MTQ / `--trt` 对照 | HoloQuant 01 静态图 |
|---|---|---|
| **Vision**（SigLIP + 投影） | **不量化**，仍按 bf16 算。ModelOpt 关掉了 `vision_tower`、`multi_modal_projector` | Linear（QKV/O、MLP、投影）打成 **FP8 E4M3**；patch Conv、LayerNorm 仍是 FP32 |
| **Prefix**（Gemma-2B） | Linear **FP8 E4M3**（`FP8_DEFAULT_CFG`，和 `mtqfp8` 同一配方） | 同样用 `fp8_amax.json` 的 `mtq` 段 |
| **Denoise**（Gemma-300M expert） | Linear **FP8 E4M3**，同上 | 同上 |

Vision **不是 FP16**。官方不量化它，是因为 MTQ 配方故意关掉了；脚本仍用 forward hook 统计每个 Vision Linear 的 `max|x|`，写进 `fp8_amax.json` 的 `vision` 段，给 HoloQuant 的 01 打包用。

注意力的 Q@K / attn@V、Softmax、RoPE、time MLP、`action_in_proj` / `action_out_proj`：**官方也不量化**。HoloQuant 里这些走 FP32 算子。

看产物认字段：`fp8_amax.json` 里 `mtq` 是 Prefix/Denoise（真 MTQ），`vision` 是 hook 统计（不是 ModelOpt）。

---

## 3. 怎么跑

日常在 **openpi 仓库根**跑 `.sh`。Cursor「Run」点 `.py` 也可以：脚本发现不是 `.venv` 时会自己切过去。

权重是官方 **π₀.₅-LIBERO**（`pi05_libero`），四个套件共用这一份，不是 Spatial 专用模型。

```bash
cd /share_data/bruce/workspace/ai/openpi

# 默认：四个评测套件各随机抽 5 条真实初始观测（共 20）+ L0 bf16 + MTQ
./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh

# 同上，再加 TensorRT mtqfp8 对照
./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh --trt

# 以前的假数据尺子（随机噪声图 + 假 prompt）
./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh --obs random --num-samples 10

# 只用 Spatial 视频第一帧
./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh --obs video --num-samples 10

# 只要一个套件、各抽 1 条（冒烟）
./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh --suites libero_spatial --per-suite 1
```

`--obs libero`（默认）会进 LIBERO 仿真器：按套件在「任务 × 初始状态」里不放回抽样，空转 10 步让物体落下，再拍照。  
这是开场那一帧的**数值尺子**，不是在仿真器里把任务跑完算成功率。成功率仍走 `bf6_run_eval.sh`。

若终端刷 `[Warning]: datasets path ... does not exist`：那是 LIBERO 在找演示 hdf5 目录，评测开场用不到。脚本会建一个空目录消掉警告；也可以自己执行：

```bash
mkdir -p third_party/libero/libero/datasets
```

不要为了消警告去下载整套 LIBERO 演示数据。

`--trt` 会设 `OPENPI_TRT_PRECISION=mtqfp8`，跑已经编好的 **FP8** engine。本机还没有 `deploy/tensorrt/artifacts/` 下的 plan 时，不要加 `--trt`，或先走 `../fp8/` 把 engine 编出来。

会覆盖默认产物目录。若要留旧的 4/10 条假数据，先备份或加 `--out-dir`。起 20 次仿真器会比假数据慢。

常用参数（加在 `.sh` 后面即可）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--obs` | `libero` | `libero`：四个评测套件真实初始观测；`random`：假数据；`video`：Spatial 视频帧 |
| `--per-suite` | 5 | `--obs libero` 时每个套件抽几条（默认 4×5=20） |
| `--suites` | `libero_spatial,libero_object,libero_goal,libero_10` | 逗号分隔 |
| `--num-samples` | 10 | 只在 `--obs random` / `video` 时用 |
| `--num-calib` | 16 | MTQ 校准条数（仍用假数据，避免量化格子被评测抽样带偏） |
| `--seed` | 0 | 评测抽样可复现 |
| `--trt` | 关 | 同一批样本再跑 TensorRT `mtqfp8` |
| `--out-dir` | 见下一节 | 产物目录；也可设 `HOLOQUANT_EXPORT_DIR` |
| `--checkpoint-dir` | 自动查找 | 含 `model.safetensors` 的目录；也可设 `OPENPI_CHECKPOINT_DIR` |
| `--device` | `cuda:0` | 需要 GPU |

加 `--trt` 时需要本机能加载 TensorRT：`export TRT_ROOT=/path/to/TensorRT`（目录下有 `lib/`），或把 `trtexec` 放进 `PATH`。不要把别人机器上的绝对路径写进脚本。

---

## 4. 产物写到哪里、里面是什么

默认顺序：`HOLOQUANT_EXPORT_DIR` → 从仓库往上找已存在的 `Outputs/HoloQuant/pi05_libero/openpi_export/` → 否则写到仓库内 `outputs/holoquant_export/`。

本机若已有共享 `Outputs/HoloQuant`，通常是：

```text
<workspace>/Outputs/HoloQuant/pi05_libero/openpi_export/
  config.json
  constants.npz
  fp8_amax.json
  samples/sample_000.npz …（`--obs libero` 默认 20 条）
  summary.json
```

| 文件 | 内容 |
|---|---|
| `config.json` | 层数、头数、eps、RoPE、mask 值等，01 建图用 |
| `constants.npz` | RoPE `inv_freq`、时间编码频率表 |
| `fp8_amax.json` | `mtq`：Prefix/Denoise 每个 Linear 的 input/weight amax；`vision`：hook 统计 |
| `samples/sample_XXX.npz` | 输入 + `ref_*`（L0 bf16）+ `fq_actions`（L1 MTQ）+ 可选 `trt_actions`；`libero` 时还有 `suite` / `task_id` / `episode` / `prompt` |
| `summary.json` | L1 / TRT 相对 bf16 的动作 RMSE，以及 `sample_meta` |

成功标志：上面四个文件都在，且 `samples/` 里有 20 个 `sample_*.npz`（`--obs libero` 默认）。
`summary.json` 认这几项：

- `obs` 为 `libero`（或你显式选的 `random` / `video`），**不是** `synthetic_tensors`
- `libero` 时 `sample_meta` 四个套件各 5 条，带真实任务语言
- 有 `fq_rmse_mean`（MTQ 相对 bf16）
- 加了 `--trt` 时 `trt_rmse_mean` 有数，不是 `null`

目录里如果已经是别的导出，先把旧目录改名挪走再跑，避免和官方尺子混在一起。
本脚本默认会写入上述路径；HoloQuant 的 01/02/03 默认也读这里。

---

## 5. 导出之后怎么接到 NPU 仿真

本目录只负责导出尺子。后面在 **HoloQuant / `torch212`**：

```bash
conda activate torch212
cd /share_data/bruce/workspace/Quant_Repo/HoloQuant

# 整网建图（不要加 --smoke，否则只有 1 层）
python holoq/samples/pi05_npu_sim/pi05_full/01_build_static_graphs.py

# 遍历图 + 调度自研算子（mixed：量化/SFU 走芯片，乘加走 torch）
python holoq/samples/pi05_npu_sim/pi05_full/02_run_npu_sim.py --backend mixed

python holoq/samples/pi05_npu_sim/pi05_full/03_compare.py --backend mixed
```

`02` 会按目录里有几个 `sample_*.npz` 跑几条（默认 20）。完整命令见 HoloQuant `Docs/npu仿真与π05整网.md` 第 11 节。

`02 --backend ba2701` 是逐位 MMA，整网很慢，不要用来扫精度。

---

## 6. 本目录有什么

| 文件 | 作用 |
|---|---|
| `export_npu_reference.py` | 真正干活：加载模型、录 L0、MTQ 校准、可选 TRT |
| `export_npu_reference.sh` | 选对本仓库 `.venv`，设好 `PYTHONPATH` 和 TensorRT 动态库 |

直接点 `.py` 或 `python3 export_npu_reference.py` 也可以：不是 `.venv` 时脚本会自己切过去。日常仍推荐 `.sh`。

---

## 7. 不要这样做

| 做法 | 结果 |
|---|---|
| 仓库里没有 `.venv`，却用系统 / conda python 跑 `.py` | 找不到 `openpi`；先建好 OpenPI 环境 |
| 把「bf16 + MTQ」理解成整网用 bf16 | 量化目标其实是 FP8 E4M3 |
| 以为 Vision 是 FP16 | 官方不量化（bf16 算）；HoloQuant 图里 Linear 是 FP8 |
| 用合成输入 / 本地假量化当精度尺子 | 那不是默认的 `--obs libero`；NPU 精度分析用本脚本导出 |
| 把 20 条开场观测当成四个任务的成功率 | 成功率请跑 `bf6_run_eval.sh` |
| HoloQuant `01`/`02` 加 `--smoke` 当整网精度 | 图只有 1 层 / 只跑 1 步，不对比整网动作 |
