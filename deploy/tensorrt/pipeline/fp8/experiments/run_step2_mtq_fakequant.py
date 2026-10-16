#!/usr/bin/env python3
"""第 2 步：只在 PyTorch 里对 Linear 做 FP8 fake-quant，对照 bf16 动作 RMSE。

不经过 ONNX / TensorRT。注意力 MatMul 不量化（MTQ 默认只挂在 nn.Linear 上）。
"""

from __future__ import annotations

import argparse
import copy
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
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    inductor_cache_env,
    write_json,
)

OUT_DIR = Path(__file__).resolve().parent

# 这些模块即使在「只量化 Denoise Linear」时也要关掉（和 openpi_lz 配方对齐）。
_ALWAYS_DISABLE = (
    "*vision_tower*",
    "*multi_modal_projector*",
    "*patch_embedding*",
    "*time_mlp*",
    "*action_in_proj*",
    "*action_out_proj*",
    "*state_proj*",
    "*norm.dense*",
)


def _append_disable(quant_cfg: dict, pattern: str) -> None:
    qc = quant_cfg["quant_cfg"]
    entry = {"quantizer_name": pattern, "enable": False}
    if isinstance(qc, list):
        qc.append(entry)
    elif isinstance(qc, dict):
        qc[pattern] = {"enable": False}
    else:
        raise TypeError(f"不认识的 quant_cfg 类型: {type(qc)}")


def _scope_disable_patterns(scope: str) -> list[str]:
    """额外关掉「不是当前实验对象」的量化器。"""
    if scope == "prefix":
        # 只留 PaliGemma language_model 的 Linear。
        return ["*gemma_expert*", *_ALWAYS_DISABLE]
    if scope == "denoise":
        # 只留 action expert 的 Linear；AdaRMS dense / time MLP 仍关。
        return ["*language_model*", *_ALWAYS_DISABLE]
    raise ValueError(f"未知 scope: {scope}")


def _enabled_quantizers(model: torch.nn.Module) -> list[str]:
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

    names = []
    for name, mod in model.named_modules():
        if isinstance(mod, TensorQuantizer) and getattr(mod, "is_enabled", False):
            names.append(name)
    return names


def _make_obs(policy, device: str):
    from openpi.models import model as openpi_model
    from openpi.policies.libero_policy import make_libero_example

    example = make_libero_example()
    inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...],
        policy._input_transform(example),
    )
    return openpi_model.Observation.from_dict(inputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scope", choices=("prefix", "denoise"), default="prefix")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-calib", type=int, default=16)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    inductor_cache_env()
    import modelopt.torch.quantization as mtq
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as train_config

    cfg = train_config.get_config(args.config_name)
    object.__setattr__(cfg.model, "pytorch_compile_mode", None)
    policy = _policy_config.create_trained_policy(
        cfg, args.checkpoint_dir, pytorch_device=args.device
    )
    model = policy._model
    model.eval()

    rng = np.random.default_rng(0)
    eval_pack = []
    print(f"[s2] 先跑 {args.num_samples} 条 bf16 参考动作")
    for i in range(args.num_samples):
        observation = _make_obs(policy, args.device)
        noise = torch.from_numpy(rng.standard_normal((1, ACTION_HORIZON, 32)).astype(np.float32)).to(
            args.device
        )
        with torch.no_grad():
            ref = model.sample_actions(args.device, observation, noise=noise.clone())
        eval_pack.append((observation, noise, ref.detach().float().cpu()))
        print(f"  ref[{i}] finite={bool(torch.isfinite(ref).all())} absmax={float(ref.float().abs().max()):.4f}")

    calib_rng = np.random.default_rng(123)
    calib = []
    for _ in range(args.num_calib):
        observation = _make_obs(policy, args.device)
        noise = torch.from_numpy(
            calib_rng.standard_normal((1, ACTION_HORIZON, 32)).astype(np.float32)
        ).to(args.device)
        calib.append((observation, noise))

    quant_cfg = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    for pattern in _scope_disable_patterns(args.scope):
        _append_disable(quant_cfg, pattern)
    print(f"[s2] scope={args.scope}  disable={_scope_disable_patterns(args.scope)}")

    def forward_loop(mdl) -> None:
        mdl.eval()
        with torch.no_grad():
            for i, (observation, noise) in enumerate(calib):
                mdl.sample_actions(args.device, observation, noise=noise.clone())
                if (i + 1) % 4 == 0 or i + 1 == len(calib):
                    print(f"  calib {i + 1}/{len(calib)}")

    print("[s2] mtq.quantize（Linear FP8 fake-quant + 校准）")
    mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    enabled = _enabled_quantizers(model)
    print(f"[s2] 启用的 TensorQuantizer 数量: {len(enabled)}")
    for name in enabled[:30]:
        print(f"  {name}")
    if len(enabled) > 30:
        print(f"  ... 还有 {len(enabled) - 30} 个")

    rmses = []
    maxes = []
    print("[s2] 对照 fake-quant vs 刚才的 bf16 参考")
    for i, (observation, noise, ref_cpu) in enumerate(eval_pack):
        with torch.no_grad():
            hyp = model.sample_actions(args.device, observation, noise=noise.clone())
        delta = (ref_cpu - hyp.float().cpu()).numpy()
        rmse = float(np.sqrt(np.mean(delta**2)))
        mx = float(np.abs(delta).max())
        rmses.append(rmse)
        maxes.append(mx)
        print(f"[{i}] action RMSE={rmse:.6f}  max|Δ|={mx:.6f}  hyp_nan={int(hyp.isnan().sum())}")

    summary = {
        "rmse_mean": float(np.mean(rmses)),
        "rmse_max": float(np.max(rmses)),
        "num_samples": args.num_samples,
        "num_calib": args.num_calib,
        "scope": args.scope,
        "precision": "mtq_fp8_fakequant",
        "num_enabled_quantizers": len(enabled),
        "enabled_quantizers_head": enabled[:80],
    }
    out = Path(args.out) if args.out else OUT_DIR / f"s2_{args.scope}_mtq.json"
    write_json(out, summary)
    print(summary)
    print(f"wrote {out}")

    mean = summary["rmse_mean"]
    if mean < 0.02:
        print("[s2] 判定：过关。配方本身没问题，下一刀应是「先 MTQ 再导出」。")
    elif mean > 0.15:
        print("[s2] 判定：不过关。Linear FP8 假量化已经差了，下一刀应缩小量化范围。")
    else:
        print("[s2] 判定：中间带。可冒烟，但先不要跑 500 条 LIBERO。")


if __name__ == "__main__":
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    main()
