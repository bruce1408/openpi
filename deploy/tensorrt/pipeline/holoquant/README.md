# 给 HoloQuant NPU 仿真的参考导出

本目录属于 **OpenPI**。只做一件事：用本仓库的真实 π₀.₅、ModelOpt MTQ、（可选）TensorRT，
写出 json / npz，给旁边的 HoloQuant 当**精度尺子**。

```text
本目录（OpenPI .venv）  --写出文件-->  Outputs/HoloQuant/pi05_libero/openpi_export_40/
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
| L1 | ModelOpt 把 Prefix / Denoise 的 Linear 按 **FP8 E4M3** 假量化；Vision 的 Linear 按 `vision` amax 假量化（`--l1-vision fp8`，默认） | 本脚本必做（MTQ + Vision hook） |
| L1-TRT | 同一批样本再跑 TensorRT `mtqfp8` engine | 加 `--trt` 才有 |
| L2 | HoloQuant `02_run_npu_sim.py` 遍历静态图、调度自研 NPU 算子 | 不在本目录 |

可以想成考试：bf16 是标准答案，FP8 是你要练的答题方式。没有 L0，后面没法判断 NPU 偏了多少。

HoloQuant 整网仿真要的就是这套 **FP8 尺子**。不要把「bf16 + MTQ」读成「整网用 bf16」。

---

## 2. 三段各自量化没有

一次 `infer` 可以想成三段。Prefix / Denoise 和 TensorRT 当前方案对齐；Vision 官方不量化，但 HoloQuant 建图时量化了，所以 L1 也跟着量化，L1 / L2 才可比：

| 段 | 官方 MTQ / `--trt` 对照 | 本脚本 L1（`fq_actions`） | HoloQuant 01 静态图 |
|---|---|---|---|
| **Vision**（SigLIP + 投影） | **不量化**，仍按 bf16 算。ModelOpt 关掉了 `vision_tower`、`multi_modal_projector` | Linear 按 `vision` amax 假量化成 **FP8 E4M3**（`--l1-vision off` 则不量化） | Linear（QKV/O、MLP、投影）打成 **FP8 E4M3**；patch Conv、LayerNorm 仍是 FP32 |
| **Prefix**（Gemma-2B） | Linear **FP8 E4M3**（`FP8_DEFAULT_CFG`，和 `mtqfp8` 同一配方） | MTQ 假量化 | 同样用 `fp8_amax.json` 的 `mtq` 段 |
| **Denoise**（Gemma-300M expert） | Linear **FP8 E4M3**，同上 | 同上 | 同上 |

Vision **不是 FP16**。官方不量化它，是因为 MTQ 配方故意关掉了；脚本用 forward hook 统计每个 Vision Linear 的 `max|x|`，写进 `fp8_amax.json` 的 `vision` 段。HoloQuant 01 用这份 amax 打包；L1 也用同一份 amax、同一套格子（`S = amax/448`，`x·(1/S)` 截断到 ±448 后转 E4M3）假量化 Vision 的输入和权重。

所以 TRT 对照仍是「Vision bf16」，L1 默认是「Vision FP8」，两者 Vision 不同；比 L1 和 L2 时看 L1，比 MTQ 和 TRT 时加 `--l1-vision off`。

注意力的 Q@K / attn@V、Softmax、RoPE、time MLP、`action_in_proj` / `action_out_proj`：**官方也不量化**。HoloQuant 里这些走 FP32 算子。

看产物认字段：`fp8_amax.json` 里 `mtq` 是 Prefix/Denoise（真 MTQ），`vision` 是 hook 统计（不是 ModelOpt）。

---

## 3. 怎么跑

日常在 **openpi 仓库根**跑 `.sh`。Cursor「Run」点 `.py` 也可以：脚本发现不是 `.venv` 时会自己切过去。

权重是官方 **π₀.₅-LIBERO**（`pi05_libero`），四个套件共用这一份，不是 Spatial 专用模型。

### 3.1 LIBERO 数据集与本次评测子集

LIBERO 用 **suite（评测套件）** 组织机器人操作任务；一个 suite 包含多个 **task（具体任务）**，一个 task 又配有多个可用于重置仿真器的初始状态。它们的层级关系是：

```text
LIBERO benchmark
└── suite：针对某一类泛化能力设计的任务集合
    └── task：一个具体的「场景 + 语言指令 + 成功条件」
        ├── initial state：该 task 的一种可复现起始摆放和机器人状态
        ├── episode / rollout：从某个初始状态开始的一次完整执行
        └── export sample：本脚本截取的开场观测和数值参考结果
```

#### Suite 是什么

Suite 是为了检验某一类泛化能力而组织的**任务集合**，不是机器人直接执行的一条指令。例如 `libero_spatial` 专门考察空间关系变化：操作对象和最终目标可以相似，但黑色碗的起始位置不同。把多个同类 task 放进同一 suite，可以统计模型对这一类变化的整体适应能力。

#### Task 是什么

Task 是机器人真正要完成的**一个具体问题**。在 LIBERO 代码中，一个 task 至少关联：

- 一条自然语言指令，例如「拿起炉子上的黑色碗，把它放到盘子上」；
- 一份 BDDL 场景文件，定义场景、物体、初始逻辑关系和目标谓词；
- 一套成功条件，用来判断一次 rollout 是否完成任务；
- 一个 `.pruned_init` 初始状态文件，本地数据中每个 task 包含 50 个可复现初始状态。

例如下面两条属于同一个 `libero_spatial` suite，但是两个不同 task：

```text
task A: pick up the black bowl on the stove and place it on the plate
task B: pick up the black bowl next to the ramekin and place it on the plate
```

两者都要把黑色碗放到盘子上，但碗的起始空间关系不同，所以是两个 task；它们都在测试空间位置理解，所以归入同一 suite。

#### 初始状态、Episode 和导出样本不是 Task

同一 task 的 50 个初始状态不是 50 个新 task；它们只是在任务语义和成功条件不变的情况下，改变物体的精确位姿、机器人关节状态等起始数值。从某个初始状态开始、让机器人一直运行到成功或超时，才构成一次 episode/rollout。

本导出脚本为了评估 NPU 数值误差，只在每个被选中的初始状态上截取开场观测，然后用相同的图像、语言、机器人状态和 Flow Matching 初始噪声比较 bf16、FP8 假量化和 NPU 仿真结果。因此一条 export sample 是**数值对齐样本**，不是一次完整 rollout，也不能直接用来计算任务成功率。

本脚本使用的四个 suite 如下：

| Suite | 主要关注的任务变化 | Task 数量 | 每个 Task 初始状态数 | 候选状态总数 |
|---|---|---:|---:|---:|
| `libero_spatial` | 空间位置和物体相对关系 | 10 | 50 | 500 |
| `libero_object` | 操作对象的类别和外观 | 10 | 50 | 500 |
| `libero_goal` | 语言目标和操作意图 | 10 | 50 | 500 |
| `libero_10` | 从 LIBERO-100 划出的 10 个组合操作任务 | 10 | 50 | 500 |
| **合计** | — | **40 个 task** | — | **2000 个初始状态** |

这里的「2000 个初始状态」是 40 个 task 的仿真器重置状态候选池，**不等于 2000 条演示轨迹，也不等于本脚本最终导出 2000 条样本**。完整的本地 LIBERO 任务定义还包含 `libero_90` 的 90 个 task；`libero_90` 与 `libero_10` 合称 LIBERO-100。加上 Spatial/Object/Goal 后，完整 LIBERO 共有 130 个不同 task 和 6500 个初始状态。本次 π₀.₅ NPU 数值对齐只使用上表的 40 个 task，不使用 `libero_90`。

默认 `--per-task 1` 采用按 task 分层抽样：遍历全部 40 个 task，在每个 task 的 50 个初始状态中使用固定 `--seed` 不放回抽 1 个，因此最终精确导出：

```text
4 suite × 10 task/suite × 1 sample/task = 40 samples
```

这种抽法保证 40 个实际 task 各出现一次，比「从每个 suite 的 500 个 `(task_id, 初始状态)` 组合中混合抽 10 条」更适合做 NPU 数值精度覆盖，因为后者可能重复抽到同一 task 并遗漏其他 task。

### 3.2 执行命令

```bash
cd /share_data/bruce/workspace/ai/openpi

# 默认：4 个套件 × 10 task/套件 × 每 task 随机抽 1 条初始观测 = 40 条
./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh

# 同上，再加 TensorRT mtqfp8 对照
./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh --trt

# 以前的假数据尺子（随机噪声图 + 假 prompt）
./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh --obs random --num-samples 10

# 只用 Spatial 视频第一帧
./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh --obs video --num-samples 10

# 只跑 Spatial 套件：10 个 task 各 1 条，共 10 条
./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh --suites libero_spatial --per-task 1
```

`--obs libero`（默认）会进 LIBERO 仿真器：恢复每条分层抽样选中的初始状态，空转 10 步让物体落下，再拍照。
这是开场那一帧的**数值尺子**，不是在仿真器里把任务跑完算成功率。成功率仍走 `bf6_run_eval.sh`。

若终端刷 `[Warning]: datasets path ... does not exist`：那是 LIBERO 在找演示 hdf5 目录，评测开场用不到。脚本会建一个空目录消掉警告；也可以自己执行：

```bash
mkdir -p third_party/libero/libero/datasets
```

不要为了消警告去下载整套 LIBERO 演示数据。

`--trt` 会设 `OPENPI_TRT_PRECISION=mtqfp8`，跑已经编好的 **FP8** engine。本机还没有 `deploy/tensorrt/artifacts/` 下的 plan 时，不要加 `--trt`，或先走 `../fp8/` 把 engine 编出来。

默认 LIBERO 产物现在写到 `openpi_export_40/`，不会覆盖旧的 `openpi_export/samples/`。起 40 次仿真器会比假数据慢。

常用参数（加在 `.sh` 后面即可）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--obs` | `libero` | `libero`：四个评测套件真实初始观测；`random`：假数据；`video`：Spatial 视频帧 |
| `--per-task` | 1 | `--obs libero` 时每个实际 task 抽几个初始状态（默认 4×10×1=40） |
| `--suites` | `libero_spatial,libero_object,libero_goal,libero_10` | 逗号分隔 |
| `--num-samples` | 10 | 只在 `--obs random` / `video` 时用 |
| `--num-calib` | 16 | MTQ 校准条数（仍用假数据，避免量化格子被评测抽样带偏） |
| `--seed` | 0 | 评测抽样可复现 |
| `--l1-vision` | `fp8` | L1 是否假量化 Vision。`fp8`：和 HoloQuant 静态图一致；`off`：只有 MTQ（旧行为，和 TRT 一致） |
| `--trt` | 关 | 同一批样本再跑 TensorRT `mtqfp8` |
| `--out-dir` | 见下一节 | 产物目录；也可设 `HOLOQUANT_EXPORT_DIR` |
| `--checkpoint-dir` | 自动查找 | 含 `model.safetensors` 的目录；也可设 `OPENPI_CHECKPOINT_DIR` |
| `--device` | `cuda:0` | 需要 GPU |

加 `--trt` 时需要本机能加载 TensorRT：`export TRT_ROOT=/path/to/TensorRT`（目录下有 `lib/`），或把 `trtexec` 放进 `PATH`。不要把别人机器上的绝对路径写进脚本。

---

## 4. 产物写到哪里、里面是什么

默认顺序：显式的 `HOLOQUANT_EXPORT_DIR` → 从仓库往上找已存在的 `Outputs/HoloQuant/pi05_libero/openpi_export_40/` → 否则写到仓库内 `outputs/holoquant_export_40/`。
后缀数字由本次 LIBERO 计划样本数自动生成；例如只选 1 个 suite、`--per-task 1` 时是 `_10`。显式传入 `--out-dir` 或 `HOLOQUANT_EXPORT_DIR` 时，你给的路径会原样使用，不自动追加后缀。

本机若已有共享 `Outputs/HoloQuant`，通常是：

```text
<workspace>/Outputs/HoloQuant/pi05_libero/openpi_export_40/
  config.json
  constants.npz
  fp8_amax.json
  samples/sample_000.npz … sample_039.npz（`--obs libero` 默认 40 条）
  summary.json
```

| 文件 | 内容 |
|---|---|
| `config.json` | 层数、头数、eps、RoPE、mask 值等，01 建图用 |
| `constants.npz` | RoPE `inv_freq`、时间编码频率表 |
| `fp8_amax.json` | `mtq`：Prefix/Denoise 每个 Linear 的 input/weight amax；`vision`：hook 统计 |
| `samples/sample_XXX.npz` | 输入 + `ref_*`（L0 bf16）+ `fq_actions`（L1：MTQ + 按 `l1_vision` 处理 Vision）+ 可选 `trt_actions`；`libero` 时还有 `suite` / `task_id` / `episode` / `prompt` |
| `summary.json` | L1 / TRT 相对 bf16 的动作 RMSE、`l1_vision`、`checkpoint`（路径 + 大小 + 完整 SHA256），以及 `sample_meta` |

成功标志：上面四个文件都在，且 `samples/` 里有 40 个 `sample_*.npz`（`--obs libero` 默认）。
`summary.json` 认这几项：

- `obs` 为 `libero`（或你显式选的 `random` / `video`），**不是** `synthetic_tensors`
- `libero` 时 `sample_meta` 覆盖 40 个 task，每个 task 恰好 1 条，带真实任务语言
- 有 `fq_rmse_mean`（L1 相对 bf16），`l1_vision` 为 `fp8`（要和 HoloQuant L2 比时）
- 加了 `--trt` 时 `trt_rmse_mean` 有数，不是 `null`

目录里如果已经是别的导出，先把旧目录改名挪走再跑，避免和官方尺子混在一起。
本脚本默认会写入上述带样本数后缀的路径。HoloQuant 端要通过 `--openpi-out` 指向这个新根目录。

---

## 5. 导出之后怎么接到 NPU 仿真

本目录只负责导出尺子。后面在 **HoloQuant / `torch212`**：

```bash
conda activate torch212
cd /share_data/bruce/workspace/Quant_Repo/HoloQuant

# 整网建图（不要加 --smoke，否则只有 1 层）
python holoq/samples/pi05_npu_sim/pi05_full/01_build_static_graphs.py \
  --openpi-out /share_data/bruce/workspace/Outputs/HoloQuant/pi05_libero/openpi_export_40

# 遍历图 + 调度自研算子（mixed：量化/SFU 走芯片，乘加走 torch）
python holoq/samples/pi05_npu_sim/pi05_full/02_run_npu_sim.py \
  --openpi-out /share_data/bruce/workspace/Outputs/HoloQuant/pi05_libero/openpi_export_40 \
  --backend mixed

python holoq/samples/pi05_npu_sim/pi05_full/03_compare.py \
  --openpi-out /share_data/bruce/workspace/Outputs/HoloQuant/pi05_libero/openpi_export_40 \
  --backend mixed
```

`02` 会按目录里有几个 `sample_*.npz` 跑几条（默认 40）。完整命令见 HoloQuant `Docs/npu仿真与π05整网.md` 第 11 节。

`02 --backend ba2701` 乘加也走芯片 MMA 模型，一条样本约 50 分钟（40 条约 33 小时），不要用来扫精度。

**重新导出后**，`fp8_amax.json` / `config.json` 可能变了：01、02 要重跑。03 会比对 01 记下的指纹，对不上就停下。

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
| 以为 Vision 是 FP16 | 官方不量化（bf16 算）；HoloQuant 图里和 L1（默认）的 Linear 是 FP8 |
| 用 `--l1-vision off` 的导出去比 HoloQuant L2 | L1 少量化了 Vision，L2 偏大一截是正常的，不是算子问题 |
| 用合成输入 / 本地假量化当精度尺子 | 那不是默认的 `--obs libero`；NPU 精度分析用本脚本导出 |
| 把 40 条开场观测当成四个任务的成功率 | 这4个是 suite，不是单独 task；成功率请跑 `bf6_run_eval.sh` |
| HoloQuant `01`/`02` 加 `--smoke` 当整网精度 | 图只有 1 层 / 只跑 1 步，不对比整网动作 |
