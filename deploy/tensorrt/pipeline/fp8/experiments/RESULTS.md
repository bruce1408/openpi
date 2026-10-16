# π₀.₅ FP8 精度恢复实验记录

目标：找出哪一刀能把 Prefix / Denoise 的动作 RMSE 从 ~0.26 拉回 <0.02。  
尺子：同一噪声、4 条样本、相对 PyTorch `sample_actions` 的 RMSE。

| RMSE | 判定 |
|------|------|
| < 0.02 | 过关，可冒烟 LIBERO |
| 0.02–0.08 | 可冒烟，再决定 500 条 |
| > 0.15 | 不过关，换下一刀，先不跑 LIBERO |

分段含义：只把列出的段换成 TensorRT，其余仍是 PyTorch。

---

## 基线（已完成）

| ID | 操作 | Prefix RMSE | Denoise RMSE | Vision RMSE | 全开 RMSE | LIBERO spatial |
|----|------|-------------|--------------|-------------|-----------|----------------|
| B0 | PyTorch bf16 | — | — | — | 0 | **98.8%** |
| B1 | 只有 Vision FP8 | — | — | **0.007** | — | **98.0%** |
| B2 | ONNX PTQ，注意力也量化 | — | ~0.34 | ~0.01 | ~0.25 | 冒烟 0/2 |
| B3 | ONNX PTQ，只量化 Linear | **0.256** | **0.363** | **0.007** | **0.263** | **0.0%**（0/500） |

B3 结论：排除注意力 **无效**。Prefix 单独 FP8 已 0.26；Denoise 单独 0.36。

---

## 第 1 步：FP16 TensorRT（不量化）

**改什么：** 现有 `*_fp16.onnx` 直接编 engine，不插 QDQ。  
**目的：** 分清是量化问题，还是 TRT 图 / KV 接口问题。

| ID | 操作 | RMSE mean | RMSE max | 判定 | JSON |
|----|------|-----------|----------|------|------|
| S1-V | 只有 Vision FP16 | **0.0016** | 0.0023 | 过关 | `s1_vision_fp16.json` |
| S1-P | 只有 Prefix FP16（原始 mask） | **NaN** | NaN | 接口坏了 | `s1_prefix_fp16.json` |
| S1-D | 只有 Denoise FP16 | **0.0012** | 0.0013 | 过关 | `s1_denoise_fp16.json` |

### S1 诊断（Prefix NaN）

`s1_prefix_nan_diag.log`：

- Prefix 长度 968，其中 **452 行整行被 mask**（padding）。
- PyTorch additive mask 是 **-2.38e38**（bf16/fp32 能表示）。
- 交给 FP16 engine 后变成 **-inf**（670768 个 `-inf`）。
- TRT `past_kv`：**100352 个 NaN**；PyTorch KV：0 个 NaN。
- 把屏蔽值改成 **-1e4** 之后：TRT KV **0 NaN**。

原因（给初学者）：softmax 遇到「一整行都是 -inf」会变成 0/0 → NaN。padding 的 KV 变成 NaN 后，后面 denoise 就算不想看这些位置，`0 × NaN` 仍是 NaN，动作就全坏了。

**修复：** `deploy/tensorrt/runtime/fp8_policy.py` 里 `_engine_attention_mask`，进 TensorRT 前把 mask 收成 fp16 能表示的有限值。

### S1b：修好 mask 再测 FP16

| ID | 操作 | RMSE mean | RMSE max | 判定 | JSON |
|----|------|-----------|----------|------|------|
| S1b-P | Prefix FP16 + 安全 mask | **0.0021** | 0.0041 | 过关 | `s1b_prefix_fp16.json` |
| S1b-D | Denoise FP16 + 安全 mask | **0.0012** | 0.0013 | 过关 | `s1b_denoise_fp16.json` |
| S1b-all | 三段都 FP16 | **0.0035** | 0.0048 | 过关 | `s1b_all_fp16.json` |

### S1c：同一套运行时，再测旧的 Linear FP8 engine

| ID | 操作 | RMSE mean | RMSE max | 判定 | JSON |
|----|------|-----------|----------|------|------|
| S1c-P | Prefix FP8 + 安全 mask | **0.256** | 0.281 | 不过关 | `s1c_prefix_fp8.json` |
| S1c-D | Denoise FP8 + 安全 mask | **0.367** | 0.396 | 不过关 | `s1c_denoise_fp8.json` |

### 第 1 步结论

1. **三段 FP16 TensorRT 图和 KV 接口是对的**（全开 RMSE 0.0035）。原先 Prefix NaN 是 mask 溢出，不是切图切错。
2. **Linear-only ONNX PTQ 的 0.26 不是这个 mask 造成的**：修好 mask 之后 FP8 Prefix 仍是 0.256，和 B3 几乎一样。
3. 因此下一刀针对 **FP8 量化本身**，而不是再改 10 步融合或 KV 布局。

不跑 LIBERO（FP8 RMSE 仍 > 0.15）。

---

## 第 2 步：在 PyTorch 里模拟 FP8（还没编 TensorRT engine）

这里的「假量化」（fake-quant）是量化文献里的术语，**不是说实验作假**。

含义：模型还在 **PyTorch + GPU 的 fp16/fp32** 里算。`mtq.quantize` 只在每个 Linear 前后插入 `TensorQuantizer`：先把数收成 FP8 能表示的格子，再立刻升回 fp16 继续乘。用来回答「如果真的按 FP8 的格子存这些权重，误差大不大」。

它 **还不是** TensorRT 的 FP8 Tensor Core，也 **还没有** `.plan` 文件。真 engine 是第 3 步才编出来的。

**改什么：** 不经过 ONNX / TensorRT。用 `modelopt.torch.quantization` 给 Linear 挂上模拟 FP8，在 PyTorch 里跑 `sample_actions`。  
**目的：** 分清是「FP8 Linear 配方就不准」，还是「ONNX PTQ / TRT 把配方走歪了」。

不量化：注意力 MatMul、Conv、time MLP、AdaRMS dense。校准 16 条 dummy LIBERO 样本。

| ID | 操作 | RMSE mean | RMSE max | 启用量化器 | 判定 | JSON |
|----|------|-----------|----------|------------|------|------|
| S2-P | 只给 Prefix Linear 挂模拟 FP8 | **0.0072** | 0.0134 | 252（18 层 × 7 个 Linear × 2） | 过关 | `s2_prefix_mtq.json` |
| S2-D | 只给 Denoise Linear 挂模拟 FP8 | **0.0027** | 0.0036 | 252（expert 同样 18×7×2） | 过关 | `s2_denoise_mtq.json` |

### 第 2 步结论

同一套「只量化 Linear」在 **PyTorch 模拟 FP8** 里过关（0.007 / 0.003），在 **ONNX PTQ → TensorRT** 里失败（0.256 / 0.367）。

所以问题不在「E4M3 per-tensor 能不能表示这些 Linear」，而在 **ONNX 图上插 QDQ / trtexec 怎么编译** 这一段。

对照：

| 路径 | Prefix | Denoise |
|------|--------|---------|
| FP16 TensorRT（S1b） | 0.002 | 0.001 |
| PyTorch 模拟 FP8（S2） | 0.007 | 0.003 |
| Linear-only ONNX PTQ（S1c） | 0.256 | 0.367 |

下一刀是第 3 步：把已经挂上模拟 FP8 的模块导出成带 Q/DQ 的 ONNX，再编真正的 TensorRT engine。

---

## 第 3 步：先 MTQ 再导出（已完成）

**改什么：** 把 S2 里已经校准好的量化模块，用 `PrefixWrapper` / `DenoiseWrapper` 导出成 `{seg}_mtqfp8.onnx`，直接 `trtexec`（不再跑 `03_quantize_fp8.py`）。  
**目的：** 看 TRT 能不能复现假量化的 0.007，而不是 ONNX PTQ 的 0.26。

| ID | 操作 | RMSE mean | RMSE max | 判定 | JSON |
|----|------|-----------|----------|------|------|
| S3-P | Prefix：MTQ → ONNX → TRT | **0.0082** | 0.0097 | 过关 | `s3_prefix_mtqfp8.json` |
| S3-D | Denoise：MTQ → ONNX → TRT | **0.0026** | 0.0035 | 过关 | `s3_denoise_mtqfp8.json` |
| S3-PD | Prefix + Denoise | **0.0080** | 0.0112 | 过关 | `s3_prefix_denoise_mtqfp8.json` |
| S3-all | 三段全开（Vision 仍用已过关的 Linear FP8） | **0.0113** | 0.0135 | 过关 | `s3_all_mtqfp8.json` |

S3 和 S2 假量化几乎一样：QDQ 跟着 Linear 模块走就对了。原来 ONNX PTQ 在图上猜哪些 MatMul 是 Linear，把 0.007 走成了 0.26。

全开 RMSE 0.011 < 0.02。LIBERO 四套、每任务 50 次、8 卡分片（各 500 条）已跑完。

| Suite | 官方 π₀.₅ | 本仓库 MTQ FP8 | 条数 |
|-------|-----------|----------------|------|
| spatial | 98.8% | **98.8%**（494/500） | `logs_libero_spatial_mtqfp8_20261013_140330` |
| object | 98.2% | **98.6%**（493/500） | `logs_libero_object_mtqfp8_20261013_141641` |
| goal | 98.0% | **96.8%**（484/500） | `logs_libero_goal_mtqfp8_20261013_142537` |
| libero_10 | 92.4% | **93.6%**（468/500） | `logs_libero_10_mtqfp8_20261013_143358` |
| **平均** | **96.85%** | **96.95%** | |

对照：旧 ONNX PTQ 三段全开 spatial 是 **0.0%**。rank 日志确认加载了 `vision/prefix/denoise_mtqfp8.plan`。

产物：`artifacts/onnx/{prefix,denoise}_mtqfp8.onnx`、`artifacts/engines/{prefix,denoise}_mtqfp8.plan`。`vision_mtqfp8.plan` 指向已过关的 `vision_fp8.plan`。

---

## 环境备忘

- 评测权重：`/share_data/public/models/openpi/pi05_libero_pytorch`
- 对照脚本：`05_compare_actions.py`（4 条样本、同一噪声）
- 某次 venv 被叠上 torch 2.14.1+cu130，和 torchvision 0.22.1+cu128 不匹配，Gemma 导入失败。第 1 步 RMSE 用缓存的 **torch 2.7.1+cu128**。
