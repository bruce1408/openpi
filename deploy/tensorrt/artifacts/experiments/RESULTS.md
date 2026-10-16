# π₀.₅ FP8 精度恢复实验记录（本目录是副本）

**请看正式记录：** [`../pipeline/fp8/experiments/RESULTS.md`](../../pipeline/fp8/experiments/RESULTS.md)

这份副本曾经停在 B3「评测中」。结论已经写完：

- 旧 ONNX PTQ（Linear MatMul）Prefix/Denoise RMSE ~0.26，LIBERO spatial **0%**。
- 当前方案：先 MTQ 再导出 `*_mtqfp8.plan`。三段全开 RMSE **0.011**，LIBERO 四套平均 **96.95%**。
