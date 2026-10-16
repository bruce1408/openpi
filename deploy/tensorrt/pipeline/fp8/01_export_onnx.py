#!/usr/bin/env python3
"""Export Vision / Prefix / Denoise subgraphs to static-shape FP16 ONNX.

用法（在 openpi 仓库根目录）::

    .venv/bin/python deploy/tensorrt/pipeline/fp8/01_export_onnx.py --device cuda:0

默认三段都导出。某一段失败不会挡住已经成功的段（会写 status.json）。
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from deploy.tensorrt.common import (  # noqa: E402
    ACTION_HORIZON,
    DEFAULT_CHECKPOINT,
    DENOISE_KV_LEN,
    EXPERT_WIDTH,
    HEAD_DIM,
    IMAGE_SIZE,
    NUM_KV_HEADS,
    NUM_LAYERS,
    ONNX_DIR,
    PREFIX_LEN,
    SUFFIX_LEN,
    VLM_WIDTH,
    cache_to_kv_tensor,
    export_onnx,
    load_pi05_pytorch,
    write_json,
)
from deploy.tensorrt.pipeline.fp8.wrappers import (  # noqa: E402
    DenoiseWrapper,
    PrefixWrapper,
    VisionWrapper,
)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _set_eager(model) -> None:
    lm = model.paligemma_with_expert.paligemma.language_model
    expert = model.paligemma_with_expert.gemma_expert.model
    lm.config._attn_implementation = "eager"  # noqa: SLF001
    expert.config._attn_implementation = "eager"  # noqa: SLF001


@torch.no_grad()
def validate_wrappers(model, device: str) -> dict:
    """Compare custom tensor KV path against HuggingFace DynamicCache path."""
    _set_eager(model)
    bsz = 1
    prefix_embs = torch.randn(bsz, PREFIX_LEN, VLM_WIDTH, device=device, dtype=torch.bfloat16)
    prefix_mask = torch.zeros(bsz, 1, PREFIX_LEN, PREFIX_LEN, device=device, dtype=torch.float32)
    prefix_pos = torch.arange(PREFIX_LEN, device=device).unsqueeze(0)

    paligemma = model.paligemma_with_expert
    _, past_hf = paligemma.forward(
        attention_mask=prefix_mask,
        position_ids=prefix_pos,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    kv_hf = cache_to_kv_tensor(past_hf)

    prefix_wrap = PrefixWrapper(paligemma.paligemma.language_model)
    kv_wrap = prefix_wrap(prefix_embs, prefix_mask, prefix_pos)
    kv_hf_engine = kv_hf.permute(2, 0, 1, 3, 4, 5).contiguous()
    prefix_err = _max_abs(kv_hf_engine, kv_wrap)

    suffix_embs = torch.randn(bsz, SUFFIX_LEN, EXPERT_WIDTH, device=device, dtype=torch.bfloat16)
    denoise_mask = torch.zeros(bsz, 1, SUFFIX_LEN, DENOISE_KV_LEN, device=device, dtype=torch.float32)
    denoise_pos = (PREFIX_LEN + torch.arange(SUFFIX_LEN, device=device)).unsqueeze(0)
    # time MLP 在 Python 里是 float32 输出；adaRMS dense 也被留在 float32。
    adarms = torch.randn(bsz, EXPERT_WIDTH, device=device, dtype=torch.float32)

    ( _, suffix_hf), _ = paligemma.forward(
        attention_mask=denoise_mask,
        position_ids=denoise_pos,
        past_key_values=past_hf,
        inputs_embeds=[None, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms],
    )
    denoise_wrap = DenoiseWrapper(paligemma.gemma_expert.model)
    suffix_wrap = denoise_wrap(suffix_embs, denoise_mask, denoise_pos, adarms, kv_wrap)
    denoise_err = _max_abs(suffix_hf, suffix_wrap)

    report = {
        "prefix_kv_max_abs": prefix_err,
        "denoise_hidden_max_abs": denoise_err,
        "ok": prefix_err < 2e-2 and denoise_err < 2e-2,
    }
    print(f"[validate] prefix KV max|Δ|={prefix_err:.4e}  denoise hidden max|Δ|={denoise_err:.4e}")
    if not report["ok"]:
        print("[validate] WARNING: wrapper drift is larger than expected; exporting anyway.")
    return report


def export_vision(model, device: str, out_dir: Path) -> dict:
    wrapper = VisionWrapper(model.paligemma_with_expert).to(device)
    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device, dtype=torch.float16)
    wrapper = wrapper.to(dtype=torch.float16)
    with torch.inference_mode():
        out = wrapper(dummy)
    path = out_dir / "vision_fp16.onnx"
    export_onnx(wrapper, (dummy,), path, ["pixel_values"], ["image_tokens"])
    manifest = {
        "segment": "vision",
        "input_name": "pixel_values",
        "input_shape": list(dummy.shape),
        "output_name": "image_tokens",
        "output_shape": list(out.shape),
        "onnx": str(path),
    }
    write_json(path.with_suffix(".manifest.json"), manifest)
    return manifest


def export_prefix(model, device: str, out_dir: Path) -> dict:
    wrapper = PrefixWrapper(model.paligemma_with_expert.paligemma.language_model).to(device)
    wrapper = wrapper.to(dtype=torch.float16)
    dummy_emb = torch.zeros(1, PREFIX_LEN, VLM_WIDTH, device=device, dtype=torch.float16)
    dummy_mask = torch.zeros(1, 1, PREFIX_LEN, PREFIX_LEN, device=device, dtype=torch.float16)
    dummy_pos = torch.arange(PREFIX_LEN, device=device, dtype=torch.int32).unsqueeze(0)
    with torch.inference_mode():
        out = wrapper(dummy_emb, dummy_mask, dummy_pos)
    path = out_dir / "prefix_fp16.onnx"
    export_onnx(
        wrapper,
        (dummy_emb, dummy_mask, dummy_pos),
        path,
        ["inputs_embeds", "attention_mask", "position_ids"],
        ["past_kv"],
    )
    manifest = {
        "segment": "prefix",
        "inputs": {
            "inputs_embeds": list(dummy_emb.shape),
            "attention_mask": list(dummy_mask.shape),
            "position_ids": list(dummy_pos.shape),
        },
        "output_shape": list(out.shape),
        "onnx": str(path),
    }
    write_json(path.with_suffix(".manifest.json"), manifest)
    return manifest


def export_denoise(model, device: str, out_dir: Path) -> dict:
    wrapper = DenoiseWrapper(model.paligemma_with_expert.gemma_expert.model).to(device)
    wrapper = wrapper.to(dtype=torch.float16)
    dummy_suffix = torch.zeros(1, SUFFIX_LEN, EXPERT_WIDTH, device=device, dtype=torch.float16)
    dummy_mask = torch.zeros(1, 1, SUFFIX_LEN, DENOISE_KV_LEN, device=device, dtype=torch.float16)
    dummy_pos = (PREFIX_LEN + torch.arange(SUFFIX_LEN, device=device, dtype=torch.int32)).unsqueeze(0)
    dummy_adarms = torch.zeros(1, EXPERT_WIDTH, device=device, dtype=torch.float16)
    dummy_kv = torch.zeros(
        1, NUM_LAYERS, 2, NUM_KV_HEADS, PREFIX_LEN, HEAD_DIM, device=device, dtype=torch.float16
    )
    with torch.inference_mode():
        out = wrapper(dummy_suffix, dummy_mask, dummy_pos, dummy_adarms, dummy_kv)
    path = out_dir / "denoise_fp16.onnx"
    export_onnx(
        wrapper,
        (dummy_suffix, dummy_mask, dummy_pos, dummy_adarms, dummy_kv),
        path,
        ["suffix_embs", "attention_mask", "position_ids", "adarms_cond", "past_kv"],
        ["suffix_hidden"],
    )
    manifest = {
        "segment": "denoise",
        "inputs": {
            "suffix_embs": list(dummy_suffix.shape),
            "attention_mask": list(dummy_mask.shape),
            "position_ids": list(dummy_pos.shape),
            "adarms_cond": list(dummy_adarms.shape),
            "past_kv": list(dummy_kv.shape),
        },
        "output_shape": list(out.shape),
        "note": "action_out_proj stays in Python; this output is the 10 suffix tokens.",
        "onnx": str(path),
    }
    write_json(path.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export π₀.₅ subgraphs to FP16 ONNX")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--segments",
        default="vision,prefix,denoise",
        help="Comma-separated: vision,prefix,denoise",
    )
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ONNX_DIR)
    args = parser.parse_args()

    segments = [s.strip() for s in args.segments.split(",") if s.strip()]
    print(f"[01] checkpoint={args.checkpoint_dir}")
    print(f"[01] device={args.device} segments={segments}")

    model, _cfg = load_pi05_pytorch(args.checkpoint_dir, args.device)
    _set_eager(model)

    status: dict = {"segments": {}}
    if not args.skip_validate:
        try:
            status["validate"] = validate_wrappers(model, args.device)
        except Exception:
            traceback.print_exc()
            status["validate"] = {"ok": False, "error": "see traceback"}

    # ONNX / TensorRT 更吃 fp16；bf16 权重在导出前统一转一次。
    print("[01] converting module weights to float16 for ONNX export")
    model.to(dtype=torch.float16)

    exporters = {
        "vision": export_vision,
        "prefix": export_prefix,
        "denoise": export_denoise,
    }
    for name in segments:
        if name not in exporters:
            raise SystemExit(f"unknown segment {name}")
        print(f"\n==== export {name} ====")
        try:
            status["segments"][name] = {"ok": True, **exporters[name](model, args.device, args.out_dir)}
        except Exception as exc:
            traceback.print_exc()
            status["segments"][name] = {"ok": False, "error": str(exc)}

    write_json(args.out_dir / "export_status.json", status)
    failed = [k for k, v in status["segments"].items() if not v.get("ok")]
    if failed:
        raise SystemExit(f"export failed for: {failed}. See {args.out_dir / 'export_status.json'}")
    print("\n[01] all requested segments exported.")


if __name__ == "__main__":
    main()
