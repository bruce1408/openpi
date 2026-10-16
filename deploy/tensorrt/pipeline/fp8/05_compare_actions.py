#!/usr/bin/env python3
"""Compare PyTorch bf16 actions vs FP8 TensorRT actions on the same noise.

用法::

    .venv/bin/python deploy/tensorrt/pipeline/fp8/05_compare_actions.py --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import jax
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from deploy.tensorrt.common import (  # noqa: E402
    ACTION_HORIZON,
    ARTIFACTS,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    inductor_cache_env,
    write_json,
)
from deploy.tensorrt.runtime.fp8_policy import Fp8ActionSampler  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--artifact-dir", default=str(ARTIFACTS))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--segments",
        default=os.environ.get("OPENPI_FP8_SEGMENTS", "vision,prefix,denoise"),
        help="对照时启用的 engine，逗号分隔。",
    )
    parser.add_argument(
        "--precision",
        default=os.environ.get("OPENPI_TRT_PRECISION", "mtqfp8"),
        help="加载 {seg}_{precision}.plan。当前方案用 mtqfp8；旧 ONNX PTQ 才是 fp8。",
    )
    parser.add_argument("--out", default="", help="结果 JSON 路径；默认 artifacts/action_compare.json")
    args = parser.parse_args()
    os.environ["OPENPI_FP8_SEGMENTS"] = args.segments
    os.environ["OPENPI_TRT_PRECISION"] = args.precision

    inductor_cache_env()
    from openpi.models import model as openpi_model
    from openpi.policies import policy_config as _policy_config
    from openpi.policies.libero_policy import make_libero_example
    from openpi.training import config as train_config

    cfg = train_config.get_config(args.config_name)
    object.__setattr__(cfg.model, "pytorch_compile_mode", None)
    policy = _policy_config.create_trained_policy(
        cfg, args.checkpoint_dir, pytorch_device=args.device
    )
    model = policy._model
    fp8 = Fp8ActionSampler(model, artifact_dir=args.artifact_dir, device=args.device)

    rmses = []
    rng = np.random.default_rng(0)
    for i in range(args.num_samples):
        example = make_libero_example()
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(args.device)[None, ...],
            policy._input_transform(example),
        )
        observation = openpi_model.Observation.from_dict(inputs)
        noise = torch.from_numpy(rng.standard_normal((1, ACTION_HORIZON, 32)).astype(np.float32)).to(
            args.device
        )
        with torch.no_grad():
            ref = model.sample_actions(args.device, observation, noise=noise.clone())
            hyp = fp8(args.device, observation, noise=noise.clone())
        delta = (ref.float() - hyp.float()).cpu().numpy()
        rmse = float(np.sqrt(np.mean(delta**2)))
        rmses.append(rmse)
        print(f"[{i}] action RMSE={rmse:.6f}  max|Δ|={np.abs(delta).max():.6f}")

    summary = {
        "rmse_mean": float(np.mean(rmses)),
        "rmse_max": float(np.max(rmses)),
        "num_samples": args.num_samples,
        "segments": args.segments,
        "precision": args.precision,
        "engines": {
            "vision": bool(fp8.vision),
            "prefix": bool(fp8.prefix),
            "denoise": bool(fp8.denoise),
        },
    }
    out = Path(args.out) if args.out else Path(args.artifact_dir) / "action_compare.json"
    write_json(out, summary)
    print(summary)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
