# OpenPI π₀.₅-LIBERO BF16 评测使用说明

本文说明如何在本机用 **bf16** 复现 LIBERO 评测。  
**不需要 policy server**：仿真和模型推理在同一个 Python 进程里完成。

相关文件：

| 文件 | 作用 |
|------|------|
| `bf6_run_eval.sh` | 推荐入口：默认 **8 卡** 并行评测并汇总成功率 |
| `examples/libero/main_local.py` | 本地评测脚本（进程内 `policy.infer()`） |
| `/share_data/public/models/openpi/pi05_libero` | 公开的 π₀.₅-LIBERO 权重 |

---

## 1. 评测在做什么

1. 加载本地 JAX checkpoint，权重按 **bfloat16** 推理。  
2. 在 LIBERO 仿真里跑任务（spatial / object / goal / libero_10）。  
3. 统计成功率：成功 episode 数 / 总 episode 数。

官方 `examples/libero/main.py` 走 websocket + `serve_policy.py`。  
本仓库的 `main_local.py` 把「问 server 要动作」改成了「本进程直接 infer」，精度仍是 bf16。

8 卡用法：**不是**把一个模型切到 8 张卡上（单卡 5090 足够），而是 8 个进程各载一份模型，把 episode 均分。

---

## 2. 环境准备（一般已就绪）

在 openpi 仓库根目录：

```bash
cd /share_data/bruce/workspace/ai/openpi
```

需要：

- 项目虚拟环境：`.venv/`（用 `uv`，不要用 conda 的 `python`）
- LIBERO 子模块：`third_party/libero`
- 权重：`/share_data/public/models/openpi/pi05_libero`

若环境被删过，在仓库根目录执行：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

LIBERO 仿真依赖（本机已装过则可跳过）：

```bash
uv pip install -e third_party/libero 'bddl==1.0.1' easydict future
uv pip install 'robosuite==1.4.1' 'gym==0.25.2'
```

---

## 3. 推荐：8 卡全量评测

```bash
cd /share_data/bruce/workspace/ai/openpi
bash bf6_run_eval.sh
```

默认：

- GPU：`0`～`7` 共 8 张
- Suite：`libero_spatial`
- 每任务次数：`50`（官方设置；spatial 共 10 任务 × 50 = 500 条）
- 权重：`/share_data/public/models/openpi/pi05_libero`

可改参数：

```bash
# 换任务集
SUITE=libero_object bash bf6_run_eval.sh
SUITE=libero_goal   bash bf6_run_eval.sh
SUITE=libero_10     bash bf6_run_eval.sh

# 换卡数 / 每任务次数
NUM_GPUS=4 TRIALS=2 bash bf6_run_eval.sh

# 换权重路径
CHECKPOINT_DIR=/share_data/public/models/openpi/pi05_libero bash bf6_run_eval.sh

# EGL 渲染报错时改用 glx
MUJOCO_GL=glx bash bf6_run_eval.sh
```

脚本会：

1. 启动 8 个 worker，每卡一个  
2. 日志写到 `data/libero/logs_<suite>_<时间戳>/rank*.log`  
3. 全部结束后汇总 `summary.json`

---

## 4. 如何查看成功率

### 跑完以后（最重要）

```bash
cat data/libero/logs_libero_spatial_<时间戳>/summary.json
```

关注：

```json
"episodes": 500,
"successes": 494,
"success_rate": 0.988
```

`success_rate` 就是这次评测的总成功率。也可以看脚本结束时打印的：

```text
Total episodes: ...
Total successes: ...
Total success rate: 0.xxxx (xx.x%)
```

### 跑的过程中

```bash
# 看某一张卡
tail -f data/libero/logs_libero_spatial_<时间戳>/rank0.log

# 搜成功/失败
grep -E "Success:|success rate|Total success" data/libero/logs_libero_spatial_<时间戳>/rank*.log
```

日志含义：

| 日志 | 含义 |
|------|------|
| `Success: True/False` | 这一条 episode 是否完成任务 |
| `# successes: N (xx.x%)` | 当前进程累计成功数 |
| `Current task success rate` | 当前任务成功率 |
| `Total success rate` | 该 shard 的总成功率（汇总前只代表这一张卡） |

**完整数字以 `summary.json` 为准**（8 卡加总）。

### 回放视频

```text
data/libero/videos_<suite>/rank<卡号>/rollout_tXX_eXXX_<任务名>_success.mp4
data/libero/videos_<suite>/rank<卡号>/rollout_tXX_eXXX_<任务名>_failure.mp4
```

文件名里的 `success` / `failure` 就是该条成没成功。

---

## 5. 单卡 / 冒烟（可选）

不跑 8 卡、只验证通路：

```bash
cd /share_data/bruce/workspace/ai/openpi
export PYTHONPATH=/share_data/bruce/workspace/ai/openpi/third_party/libero
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl

.venv/bin/python examples/libero/main_local.py \
  --args.config-name pi05_libero \
  --args.checkpoint-dir /share_data/public/models/openpi/pi05_libero \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 2 \
  --args.max-tasks 1
```

必须用 `.venv/bin/python` 或 `uv run python`，不要用已经 `conda activate` 后的 `python`。

只测模型能不能出动作（不算 LIBERO 成功率）：

```bash
uv run python scripts/smoke_infer_bf16.py \
  --checkpoint_dir /share_data/public/models/openpi/pi05_droid
```

---

## 6. 和官方数字对比

π₀.₅-LIBERO（`pi05_libero`，每任务 50 次）：

| Suite | 官方成功率 |
|-------|------------|
| libero_spatial | 98.8% |
| libero_object | 98.2% |
| libero_goal | 98.0% |
| libero_10 | 92.4% |
| 平均 | 96.85% |

只有 **全 suite、每任务 50 次** 才适合和上表对比。  
1 个任务 × 2 条的冒烟即使 100%，也不能当成复现结果。

---

## 7. 常见问题

### `No module named 'libero'`

没找到 LIBERO 包。用 `bash bf6_run_eval.sh`（脚本会设 `PYTHONPATH`），或：

```bash
export PYTHONPATH=/share_data/bruce/workspace/ai/openpi/third_party/libero
```

不要只复制 `python examples/libero/main_local.py ...` 而漏掉 `export`。

### 末尾 `libGLU.so.0` / `EGLError` / `Exception ignored`

评测**已经结束**，Python 退出时 EGL 渲染器清理失败。  
只要前面已经打印 `Total success rate` 或生成了 `summary.json`，就可以忽略。  
可选：`sudo apt-get install -y libglu1-mesa`

### `Ctrl+C` 后出现 orbax / TensorStore 长 traceback

那是中断加载 checkpoint，不是权重损坏。重新跑即可。

### 用错了 Python

`which python` 若指向 conda（例如 `torch270_128`），会缺包。  
本评测始终用：

```text
/share_data/bruce/workspace/ai/openpi/.venv/bin/python
```

### 为什么不用 `serve_policy.py`？

那是给真机 / 远程客户端用的。纯仿真评测用 `main_local.py` 更简单，精度同样是 bf16。

---

## 8. 其它权重位置

| 模型 | 路径 | 用途 |
|------|------|------|
| π₀.₅-LIBERO | `/share_data/public/models/openpi/pi05_libero` | 本评测 |
| π₀.₅-DROID | `/share_data/public/models/openpi/pi05_droid` | DROID 推理 |
| π₀.₅-base | `/share_data/public/models/openpi/pi05_base` | 自己数据微调 |

任何人可读。推理精度均为 **bf16 为主 + 少量 fp32**。
