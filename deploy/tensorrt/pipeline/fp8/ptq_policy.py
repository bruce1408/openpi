"""ONNX FP8 PTQ 白名单：只量化 Linear 投影，不改注意力算法。

**旧路径。** 在 ONNX 图上靠名字猜 Linear ``MatMul`` 会把 Prefix/Denoise
走成 RMSE ~0.26。当前方案是 PyTorch MTQ 后再导出 ``*_mtqfp8.onnx``。

本仓库导出的 π₀.₅ ONNX 里，``nn.Linear`` 几乎全部是 ``MatMul``（不是 ``Gemm``）。
注意力的 Q@K / attn@V 也是 ``MatMul``。如果按算子类型一刀切，两者会一起进 FP8。

对齐 openpi_lz 的精度策略、但不改 ``GemmaAttention``：

* **量化**：q/k/v/o_proj、MLP、SigLIP fc、multimodal projector（Linear 投影）
* **不量化**：注意力 score 的 MatMul、Softmax、RoPE、Conv patch_embedding、
  AdaRMS dense（它们导出成 ``Gemm``）、残差 Add

注意力结构仍是原始的 ``cat(prefix, suffix)`` 一次 MHA。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import onnx

# ModelOpt 的 nodes_to_exclude 走 re.match(pattern, node.name)。
EXCLUDE_NAME_PATTERNS: tuple[str, ...] = (
    r".*/self_attn/MatMul(_\d+)?$",  # Vision：Q@K / attn@V
    r"^/MatMul(_\d+)?$",  # Prefix / Denoise：eager attention 的匿名 MatMul
    r".*/rotary_emb/MatMul$",  # RoPE，不是 Linear
)

# 只把 MatMul 交给 FP8。Conv / Gemm / Add 不进这个名单：
#   Conv  = SigLIP patch_embedding
#   Gemm  = AdaRMS dense（只出现在 denoise）
#   Add   = 残差，ModelOpt FP8 默认也会扫，这里关掉
OP_TYPES_TO_QUANTIZE: tuple[str, ...] = ("MatMul",)
OP_TYPES_TO_EXCLUDE: tuple[str, ...] = ("Conv", "Gemm", "Add")


def node_is_excluded(name: str, patterns: Iterable[str] = EXCLUDE_NAME_PATTERNS) -> bool:
    """名字是否匹配 ModelOpt 的排除正则（与 ``re.match`` 规则一致）。"""
    return any(re.match(p, name or "") for p in patterns)


def classify_node(name: str, op_type: str) -> str:
    """把一个 ONNX 节点归到 PTQ 角色，方便打印对照表。"""
    if op_type == "Conv":
        return "skip_conv_patch"
    if op_type == "Gemm":
        return "skip_adarms_dense"
    if op_type != "MatMul":
        return "ignore"
    if node_is_excluded(name):
        if "rotary_emb" in (name or ""):
            return "skip_rope"
        return "skip_attn_score"
    return "quantize_linear"


def plan_for_onnx(onnx_path: str | Path) -> dict:
    """扫描 FP16 ONNX，列出将量化 / 排除的节点（不跑 ModelOpt）。"""
    model = onnx.load(str(onnx_path), load_external_data=False)
    roles: dict[str, list[str]] = {}
    op_counts: Counter[str] = Counter()
    for node in model.graph.node:
        op_counts[node.op_type] += 1
        role = classify_node(node.name, node.op_type)
        if role == "ignore":
            continue
        roles.setdefault(role, []).append(node.name)
    return {
        "onnx": str(onnx_path),
        "num_nodes": len(model.graph.node),
        "op_counts": dict(op_counts),
        "roles": {k: v for k, v in roles.items()},
        "role_counts": {k: len(v) for k, v in roles.items()},
        "exclude_patterns": list(EXCLUDE_NAME_PATTERNS),
        "op_types_to_quantize": list(OP_TYPES_TO_QUANTIZE),
        "op_types_to_exclude": list(OP_TYPES_TO_EXCLUDE),
    }


def print_plan(plan: dict) -> None:
    counts = plan["role_counts"]
    print(f"     PTQ plan  nodes={plan['num_nodes']}")
    print(
        "     quantize Linear MatMul={q} | skip attn={a} rope={r} "
        "adarms_Gemm={g} conv={c}".format(
            q=counts.get("quantize_linear", 0),
            a=counts.get("skip_attn_score", 0),
            r=counts.get("skip_rope", 0),
            g=counts.get("skip_adarms_dense", 0),
            c=counts.get("skip_conv_patch", 0),
        )
    )


def count_qdq(onnx_path: str | Path) -> dict[str, int]:
    """量化后的粗统计：QDQ 数量、以及还裸着的 MatMul/Gemm/Conv。"""
    model = onnx.load(str(onnx_path), load_external_data=False)
    counts: Counter[str] = Counter(n.op_type for n in model.graph.node)
    qdq = counts.get("QuantizeLinear", 0) + counts.get("TRT_FP8QuantizeLinear", 0)
    dq = counts.get("DequantizeLinear", 0) + counts.get("TRT_FP8DequantizeLinear", 0)
    return {
        "QuantizeLinear": int(qdq),
        "DequantizeLinear": int(dq),
        "MatMul": int(counts.get("MatMul", 0)),
        "Gemm": int(counts.get("Gemm", 0)),
        "Conv": int(counts.get("Conv", 0)),
        "Softmax": int(counts.get("Softmax", 0)),
    }
