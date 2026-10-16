#!/usr/bin/env python3
"""Dump real intermediate tensors for ModelOpt FP8 calibration.

默认走完整 LIBERO input transform（tokenizer + 归一化）。若已有 PyTorch 评测视频，
会把第一帧填进观测，图像分布更接近仿真。

用法::

    .venv/bin/python deploy/tensorrt/pipeline/fp8/02_dump_calib.py --num-calib 32 --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from deploy.tensorrt.common import (  # noqa: E402
    ACTION_HORIZON,
    CALIB_DIR,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    NUM_LAYERS,
    cache_to_kv_tensor,
    inductor_cache_env,
    write_json,
)


def _maybe_video_frame(video_root: Path, index: int) -> np.ndarray | None:
    paths = sorted(video_root.rglob("*.mp4")) if video_root.is_dir() else []
    if not paths:
        return None
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio
    path = paths[index % len(paths)]
    reader = imageio.get_reader(path)
    try:
        frame = np.asarray(reader.get_data(0))
    finally:
        reader.close()
    return frame


@torch.no_grad()
def capture_one(
    model,
    observation,
    device: str,
    noise: torch.Tensor,
    *,
    denoise_every_step: bool = True,
) -> dict[str, np.ndarray]:
    """Run prefix once and the denoise loop; return numpy captures.

    ``denoise_every_step=True`` 时把 10 个 Euler 步都留下，校准覆盖完整时间表；
    否则只留 t≈1.0 / 0.5 / 0.1（旧行为，样本更少、更快）。
    """
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
        observation, train=False
    )
    pixel_list = [img.detach().float().cpu().numpy() for img in images]

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_mask_4d = model._prepare_attention_masks_4d(prefix_att_2d)

    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_mask_4d,
        position_ids=prefix_pos,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    past_kv = cache_to_kv_tensor(past_key_values, NUM_LAYERS)

    bsize = state.shape[0]
    x_t = noise
    dt = torch.tensor(-1.0 / 10.0, dtype=torch.float32, device=device)
    time = torch.tensor(1.0, dtype=torch.float32, device=device)

    suffix_list = []
    mask_list = []
    pos_list = []
    adarms_list = []
    step = 0
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(
            state, x_t, expanded_time
        )
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_mask_4d = model._prepare_attention_masks_4d(full_att_2d)

        keep = denoise_every_step or step in (0, 4, 9)
        if keep:
            suffix_list.append(suffix_embs.float().cpu().numpy()[0])
            mask_list.append(full_mask_4d.float().cpu().numpy()[0])
            pos_list.append(position_ids.cpu().numpy().astype(np.int32)[0])
            adarms_list.append(adarms_cond.float().cpu().numpy()[0])

        v_t = model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
        x_t = x_t + dt * v_t
        time = time + dt
        step += 1

    n_denoise = len(suffix_list)
    past_kv_np = past_kv.float().cpu().numpy().astype(np.float16)

    def _mask_fp16(arr: np.ndarray) -> np.ndarray:
        # -2.38e38 进不了 float16；转成 -inf 和 eager attention 的“屏蔽”语义一致。
        out = np.array(arr, dtype=np.float32)
        out = np.where(out < -1.0e4, np.float16(-np.inf), out.astype(np.float16))
        return out.astype(np.float16)

    return {
        "pixel_values": np.concatenate(pixel_list, axis=0).astype(np.float16),
        "inputs_embeds": prefix_embs.float().cpu().numpy().astype(np.float16)[0],
        "prefix_attention_mask": _mask_fp16(prefix_mask_4d.float().cpu().numpy()[0]),
        "prefix_position_ids": prefix_pos.cpu().numpy().astype(np.int32)[0],
        "suffix_embs": np.stack(suffix_list).astype(np.float16),
        "denoise_attention_mask": _mask_fp16(np.stack(mask_list)),
        "denoise_position_ids": np.stack(pos_list).astype(np.int32),
        "adarms_cond": np.stack(adarms_list).astype(np.float16),
        # Engine layout [B, L, 2, H, S, D]; dump 时 B=1，叠成 [N, L, 2, H, S, D]
        "past_kv": np.stack(
            [np.transpose(past_kv_np, (2, 0, 1, 3, 4, 5))[0]] * n_denoise, axis=0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump FP8 calibration tensors")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-calib", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=CALIB_DIR)
    parser.add_argument(
        "--video-root",
        type=Path,
        default=REPO_ROOT / "data/libero/videos_libero_spatial_pytorch",
    )
    parser.add_argument(
        "--denoise-every-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="校准 denoise 时保留全部 10 个 Euler 步（默认开）。--no-denoise-every-step 则只留 3 步。",
    )
    args = parser.parse_args()

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

    buckets: dict[str, list] = {k: [] for k in (
        "pixel_values",
        "inputs_embeds",
        "prefix_attention_mask",
        "prefix_position_ids",
        "suffix_embs",
        "denoise_attention_mask",
        "denoise_position_ids",
        "adarms_cond",
        "past_kv",
    )}
    rng = np.random.default_rng(0)

    for i in range(args.num_calib):
        example = make_libero_example()
        frame = _maybe_video_frame(args.video_root, i)
        if frame is not None:
            example["observation/image"] = frame
            example["observation/wrist_image"] = frame
        inputs = policy._input_transform(example)
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(args.device)[None, ...],
            inputs,
        )
        observation = openpi_model.Observation.from_dict(inputs)
        noise = torch.from_numpy(rng.standard_normal((1, ACTION_HORIZON, 32)).astype(np.float32)).to(
            args.device
        )
        cap = capture_one(
            model,
            observation,
            args.device,
            noise,
            denoise_every_step=args.denoise_every_step,
        )
        for key, value in cap.items():
            buckets[key].append(value)
        print(f"[02] captured {i + 1}/{args.num_calib}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stacked = {
        "pixel_values": np.concatenate(buckets["pixel_values"], axis=0),
        "inputs_embeds": np.stack(buckets["inputs_embeds"], axis=0),
        "prefix_attention_mask": np.stack(buckets["prefix_attention_mask"], axis=0),
        "prefix_position_ids": np.stack(buckets["prefix_position_ids"], axis=0),
        "suffix_embs": np.concatenate(buckets["suffix_embs"], axis=0),
        "denoise_attention_mask": np.concatenate(buckets["denoise_attention_mask"], axis=0),
        "denoise_position_ids": np.concatenate(buckets["denoise_position_ids"], axis=0),
        "adarms_cond": np.concatenate(buckets["adarms_cond"], axis=0),
        "past_kv": np.concatenate(buckets["past_kv"], axis=0),
    }
    for name, arr in stacked.items():
        path = args.out_dir / f"{name}.npy"
        np.save(path, arr)
        print(f"  {path.name:32s} {arr.shape} {arr.dtype}")

    write_json(
        args.out_dir / "calib_manifest.json",
        {
            "num_observations": args.num_calib,
            "pixel_values": "[N_views, 3, 224, 224] where N_views = num_calib * 3",
            "prefix": "one row per observation",
            "denoise": (
                "10 Euler timesteps per observation"
                if args.denoise_every_step
                else "3 timesteps (t≈1.0, 0.5, 0.1) per observation"
            ),
            "denoise_every_step": args.denoise_every_step,
            "video_root": str(args.video_root),
            "used_videos": args.video_root.is_dir(),
        },
    )
    print(f"[02] wrote calibration tensors under {args.out_dir}")


if __name__ == "__main__":
    main()
