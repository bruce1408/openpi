#!/usr/bin/env python3
"""第 3 步：MTQ 假量化之后，把 Prefix / Denoise wrapper 导出成带 QDQ 的 ONNX。

不跑 03_quantize_fp8.py。产物名：{seg}_mtqfp8.onnx，避免覆盖原来的 ONNX PTQ 文件。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from deploy.tensorrt.common import (  # noqa: E402
    DENOISE_KV_LEN,
    EXPERT_WIDTH,
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_LAYERS,
    ONNX_DIR,
    PREFIX_LEN,
    SUFFIX_LEN,
    VLM_WIDTH,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    export_onnx,
    inductor_cache_env,
)
from deploy.tensorrt.pipeline.fp8.wrappers import DenoiseWrapper, PrefixWrapper  # noqa: E402


def _load_s2():
    path = Path(__file__).with_name("run_step2_mtq_fakequant.py")
    spec = importlib.util.spec_from_file_location("s2_mtq", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _quantize(model, policy, device: str, scope: str, num_calib: int):
    import copy

    import modelopt.torch.quantization as mtq

    s2 = _load_s2()
    calib_rng = np.random.default_rng(123)
    calib = []
    for _ in range(num_calib):
        observation = s2._make_obs(policy, device)
        noise = torch.from_numpy(
            calib_rng.standard_normal((1, 10, 32)).astype(np.float32)
        ).to(device)
        calib.append((observation, noise))

    quant_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    for pattern in s2._scope_disable_patterns(scope):
        s2._append_disable(quant_cfg, pattern)

    def forward_loop(mdl) -> None:
        mdl.eval()
        with torch.no_grad():
            for i, (observation, noise) in enumerate(calib):
                mdl.sample_actions(device, observation, noise=noise.clone())
                if (i + 1) % 4 == 0 or i + 1 == len(calib):
                    print(f"  calib {i + 1}/{len(calib)}")

    print(f"[s3] mtq.quantize scope={scope}")
    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    enabled = s2._enabled_quantizers(model)
    print(f"[s3] enabled TensorQuantizer: {len(enabled)}")
    return enabled


def export_prefix(model, device: str, out_dir: Path) -> Path:
    wrapper = PrefixWrapper(model.paligemma_with_expert.paligemma.language_model).to(device)
    dummy_emb = torch.zeros(1, PREFIX_LEN, VLM_WIDTH, device=device, dtype=torch.float16)
    dummy_mask = torch.zeros(1, 1, PREFIX_LEN, PREFIX_LEN, device=device, dtype=torch.float16)
    dummy_pos = torch.arange(PREFIX_LEN, device=device, dtype=torch.int32).unsqueeze(0)
    path = out_dir / "prefix_mtqfp8.onnx"
    export_onnx(
        wrapper,
        (dummy_emb, dummy_mask, dummy_pos),
        path,
        ["inputs_embeds", "attention_mask", "position_ids"],
        ["past_kv"],
    )
    return path


def export_denoise(model, device: str, out_dir: Path) -> Path:
    wrapper = DenoiseWrapper(model.paligemma_with_expert.gemma_expert.model).to(device)
    dummy_suffix = torch.zeros(1, SUFFIX_LEN, EXPERT_WIDTH, device=device, dtype=torch.float16)
    dummy_mask = torch.zeros(1, 1, SUFFIX_LEN, DENOISE_KV_LEN, device=device, dtype=torch.float16)
    dummy_pos = (PREFIX_LEN + torch.arange(SUFFIX_LEN, device=device, dtype=torch.int32)).unsqueeze(0)
    dummy_adarms = torch.zeros(1, EXPERT_WIDTH, device=device, dtype=torch.float16)
    dummy_kv = torch.zeros(
        1, NUM_LAYERS, 2, NUM_KV_HEADS, PREFIX_LEN, HEAD_DIM, device=device, dtype=torch.float16
    )
    path = out_dir / "denoise_mtqfp8.onnx"
    export_onnx(
        wrapper,
        (dummy_suffix, dummy_mask, dummy_pos, dummy_adarms, dummy_kv),
        path,
        ["suffix_embs", "attention_mask", "position_ids", "adarms_cond", "past_kv"],
        ["suffix_hidden"],
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scope", choices=("prefix", "denoise"), default="prefix")
    parser.add_argument("--num-calib", type=int, default=16)
    parser.add_argument("--out-dir", default=str(ONNX_DIR))
    args = parser.parse_args()

    inductor_cache_env()
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as train_config

    cfg = train_config.get_config(args.config_name)
    object.__setattr__(cfg.model, "pytorch_compile_mode", None)
    policy = _policy_config.create_trained_policy(
        cfg, args.checkpoint_dir, pytorch_device=args.device
    )
    model = policy._model
    model.eval()
    _quantize(model, policy, args.device, args.scope, args.num_calib)

    # 和 01_export_onnx 一样：ONNX / TRT 更吃 fp16。
    print("[s3] converting to float16 for ONNX export")
    model.to(dtype=torch.float16)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.scope == "prefix":
        path = export_prefix(model, args.device, out_dir)
    else:
        path = export_denoise(model, args.device, out_dir)
    print(f"[s3] wrote {path}")


if __name__ == "__main__":
    main()
