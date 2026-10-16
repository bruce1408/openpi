#!/usr/bin/env python3
"""ModelOpt ONNX PTQ：只给 Linear 投影插 FP8 Q/DQ。

**旧路径。** Prefix/Denoise 这条实测 RMSE ~0.26，LIBERO 0%。
当前部署请用 ``experiments/run_step3_mtq_export.sh`` 的 ``*_mtqfp8.plan``。

需要 ``nvidia-modelopt``（conda 环境 ``torch270_128`` 里通常已经有）。

默认策略（对齐 openpi_lz 的「敏感算子不量化」，但不改注意力结构）::

    量化     MatMul 且名字是 q/k/v/o_proj、MLP、projector
    不量化   注意力 Q@K / attn@V、RoPE、Softmax、Conv patch_embedding、AdaRMS Gemm

用法::

    python deploy/tensorrt/pipeline/fp8/03_quantize_fp8.py --segments vision,prefix,denoise
    python deploy/tensorrt/pipeline/fp8/03_quantize_fp8.py --dry-run   # 只打印将量化哪些节点
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deploy.tensorrt.common import CALIB_DIR, ONNX_DIR, write_json  # noqa: E402
from ptq_policy import (  # noqa: E402
    EXCLUDE_NAME_PATTERNS,
    OP_TYPES_TO_EXCLUDE,
    OP_TYPES_TO_QUANTIZE,
    count_qdq,
    plan_for_onnx,
    print_plan,
)


def _load(name: str) -> np.ndarray:
    path = CALIB_DIR / name
    if not path.is_file():
        raise SystemExit(f"缺少校准文件 {path}，请先跑 02_dump_calib.py")
    return np.load(path)


def quantize_one(
    *,
    segment: str,
    input_onnx: Path,
    output_onnx: Path,
    calibration_data: dict[str, np.ndarray],
    calibration_method: str,
    calibration_eps: list[str],
    disable_mha_qdq: bool,
    dry_run: bool,
) -> None:
    if not input_onnx.is_file():
        raise SystemExit(f"找不到 {input_onnx}，请先跑 01_export_onnx.py")
    output_onnx.parent.mkdir(parents=True, exist_ok=True)

    plan = plan_for_onnx(input_onnx)
    print(f"[03] {segment}: {input_onnx.name} -> {output_onnx.name}")
    print(f"     method={calibration_method}  samples={ {k: v.shape[0] for k, v in calibration_data.items()} }")
    print_plan(plan)

    if dry_run:
        write_json(output_onnx.with_suffix(".plan.json"), plan)
        print(f"[03] dry-run，已写 {output_onnx.with_suffix('.plan.json')}")
        return

    import modelopt.onnx.quantization as moq

    extra: dict = {
        "op_types_to_quantize": list(OP_TYPES_TO_QUANTIZE),
        "op_types_to_exclude": list(OP_TYPES_TO_EXCLUDE),
        # 必须是 list：ModelOpt FP8 路径会对它 .extend()
        "nodes_to_exclude": list(EXCLUDE_NAME_PATTERNS),
        "disable_mha_qdq": disable_mha_qdq,
    }
    if disable_mha_qdq:
        extra["opset"] = 19

    moq.quantize(
        onnx_path=str(input_onnx),
        quantize_mode="fp8",
        calibration_data=calibration_data,
        calibration_method=calibration_method,
        calibration_eps=calibration_eps,
        output_path=str(output_onnx),
        use_external_data_format=True,
        **extra,
    )
    qdq = count_qdq(output_onnx) if output_onnx.is_file() else {}
    write_json(
        output_onnx.with_suffix(".manifest.json"),
        {
            "segment": segment,
            "source_onnx": str(input_onnx),
            "quantize_mode": "fp8",
            "policy": "linear_matmul_only",
            "calibration_method": calibration_method,
            "calib_shapes": {k: list(v.shape) for k, v in calibration_data.items()},
            "op_types_to_quantize": list(OP_TYPES_TO_QUANTIZE),
            "op_types_to_exclude": list(OP_TYPES_TO_EXCLUDE),
            "nodes_to_exclude": list(EXCLUDE_NAME_PATTERNS),
            "disable_mha_qdq": disable_mha_qdq,
            "role_counts": plan["role_counts"],
            "qdq_counts": qdq,
        },
    )
    print(f"[03] wrote {output_onnx}  qdq={qdq}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FP8 PTQ with ModelOpt (Linear MatMul only)")
    parser.add_argument("--segments", default="vision,prefix,denoise")
    parser.add_argument("--calibration-method", choices=("max", "entropy"), default="max")
    parser.add_argument("--calibration-eps", nargs="+", default=["cpu"])
    parser.add_argument(
        "--disable-mha-qdq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="跳过 ModelOpt 的 MHA QDQ 模板（默认开：注意力 MatMul 由名字排除，不再走 MHA 模式）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只扫描 FP16 ONNX 并打印将量化/排除的节点，不调用 ModelOpt",
    )
    args = parser.parse_args()

    if not args.dry_run:
        try:
            import modelopt.onnx.quantization as _moq  # noqa: F401
        except ImportError:
            raise SystemExit(
                "当前 Python 没有 nvidia-modelopt。\n"
                "请用已安装 modelopt 的解释器跑本脚本，例如:\n"
                "  conda run -n torch270_128 python deploy/tensorrt/pipeline/fp8/03_quantize_fp8.py"
            ) from None

    segments = [s.strip() for s in args.segments.split(",") if s.strip()]
    specs = {
        "vision": {
            "input": ONNX_DIR / "vision_fp16.onnx",
            "output": ONNX_DIR / "vision_fp8.onnx",
            "calib_keys": ["pixel_values"],
        },
        "prefix": {
            "input": ONNX_DIR / "prefix_fp16.onnx",
            "output": ONNX_DIR / "prefix_fp8.onnx",
            "calib_keys": ["inputs_embeds", "prefix_attention_mask", "prefix_position_ids"],
            "rename": {
                "prefix_attention_mask": "attention_mask",
                "prefix_position_ids": "position_ids",
            },
        },
        "denoise": {
            "input": ONNX_DIR / "denoise_fp16.onnx",
            "output": ONNX_DIR / "denoise_fp8.onnx",
            "calib_keys": [
                "suffix_embs",
                "denoise_attention_mask",
                "denoise_position_ids",
                "adarms_cond",
                "past_kv",
            ],
            "rename": {
                "denoise_attention_mask": "attention_mask",
                "denoise_position_ids": "position_ids",
            },
        },
    }
    for name in segments:
        if name not in specs:
            raise SystemExit(f"未知 segment={name}，可选: {', '.join(specs)}")
        spec = specs[name]
        rename = spec.get("rename", {})
        if args.dry_run:
            calib = {}
        else:
            calib = {}
            for key in spec["calib_keys"]:
                calib[rename.get(key, key)] = _load(f"{key}.npy")
        quantize_one(
            segment=name,
            input_onnx=spec["input"],
            output_onnx=spec["output"],
            calibration_data=calib,
            calibration_method=args.calibration_method,
            calibration_eps=args.calibration_eps,
            disable_mha_qdq=args.disable_mha_qdq,
            dry_run=args.dry_run,
        )
    print("[03] done")


if __name__ == "__main__":
    main()
