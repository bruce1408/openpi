"""π₀.₅ policy that swaps Vision / Prefix / Denoise for TensorRT engines.

没编出来的段会自动退回 PyTorch（启动时会打印用了哪些 engine）。
time MLP、action_in/out_proj、Euler 积分始终在 Python 里做。

当前可用方案是 **MTQ 后再导出** 的 ``*_mtqfp8.plan``
（``OPENPI_TRT_PRECISION=mtqfp8``，三段全开 RMSE ~0.011）。

不要用旧的 ONNX PTQ ``*_fp8.plan`` 跑 Prefix/Denoise（动作 RMSE ~0.26，LIBERO 0%）。
默认 ``OPENPI_TRT_PRECISION=mtqfp8``、``OPENPI_FP8_SEGMENTS=vision,prefix,denoise``。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

from deploy.tensorrt.common import (
    ARTIFACTS,
    ENGINE_DIR,
    NUM_LAYERS,
    cache_to_kv_tensor,
    kv_internal_to_engine,
    kv_tensor_to_cache,
)
from deploy.tensorrt.runtime.trt_runner import TensorRTRunner

logger = logging.getLogger("openpi.fp8")

# PyTorch 的 additive mask 是 -2.38e38（bf16/fp32 能表示）。
# TensorRT engine 输入是 fp16，这个数会溢出成 -inf。
# Prefix 里 padding 行整行被屏蔽 → softmax(-inf)=NaN → KV 污染后面的 denoise。
_FP16_MASK_NEG = -1.0e4


def _engine_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    """把 additive attention mask 收成 fp16 能表示的有限值。"""
    mask_f = mask.float()
    return torch.where(
        mask_f >= 0,
        torch.zeros_like(mask_f),
        torch.full_like(mask_f, _FP16_MASK_NEG),
    ).to(dtype=torch.float16)


class Fp8ActionSampler:
    def __init__(
        self,
        pytorch_model,
        artifact_dir: str | Path | None = None,
        device: str = "cuda:0",
    ) -> None:
        self.model = pytorch_model
        self.device = device
        engines = Path(artifact_dir) / "engines" if artifact_dir else ENGINE_DIR
        precision = os.environ.get("OPENPI_TRT_PRECISION", "mtqfp8").strip() or "mtqfp8"
        wanted = {
            s.strip()
            for s in (os.environ.get("OPENPI_FP8_SEGMENTS", "vision,prefix,denoise")).split(",")
            if s.strip()
        }
        self.precision = precision
        self.vision = _maybe_runner(engines / f"vision_{precision}.plan", device) if "vision" in wanted else None
        self.prefix = _maybe_runner(engines / f"prefix_{precision}.plan", device) if "prefix" in wanted else None
        self.denoise = _maybe_runner(engines / f"denoise_{precision}.plan", device) if "denoise" in wanted else None
        logger.info(
            "TRT engines (%s): vision=%s prefix=%s denoise=%s",
            precision,
            bool(self.vision),
            bool(self.prefix),
            bool(self.denoise),
        )
        if not any((self.vision, self.prefix, self.denoise)):
            logger.warning("没有找到任何 %s engine，完全走 PyTorch。", precision)

    def _embed_images(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.vision is None:
            return [self.model.paligemma_with_expert.embed_image(img) for img in images]
        tokens = []
        for img in images:
            out = self.vision({"pixel_values": img})["image_tokens"]
            tokens.append(out.to(dtype=img.dtype if img.dtype.is_floating_point else torch.bfloat16))
        return tokens

    def _embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """Same as PI0Pytorch.embed_prefix, but vision may be TensorRT."""
        import math

        embs = []
        pad_masks = []
        att_masks = []
        img_embs = self._embed_images(images)
        for img_emb, img_mask in zip(img_embs, img_masks, strict=True):
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self.model.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks_t = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        bsize = pad_masks.shape[0]
        att_masks_t = att_masks_t[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks_t

    def _prefix_kv(self, prefix_embs, prefix_mask_4d, prefix_pos):
        if self.prefix is None:
            paligemma = self.model.paligemma_with_expert
            paligemma.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            _, past = paligemma.forward(
                attention_mask=prefix_mask_4d,
                position_ids=prefix_pos,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            internal = cache_to_kv_tensor(past, NUM_LAYERS)
            return past, kv_internal_to_engine(internal)
        out = self.prefix(
            {
                "inputs_embeds": prefix_embs,
                "attention_mask": _engine_attention_mask(prefix_mask_4d),
                "position_ids": prefix_pos.int(),
            }
        )["past_kv"]
        expert_dtype = self.model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
        return kv_tensor_to_cache(out, dtype=expert_dtype), out

    def _denoise_hidden(
        self,
        suffix_embs,
        full_mask_4d,
        position_ids,
        adarms_cond,
        past_cache,
        past_kv_tensor,
    ):
        if self.denoise is None:
            paligemma = self.model.paligemma_with_expert
            paligemma.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
            outputs_embeds, _ = paligemma.forward(
                attention_mask=full_mask_4d,
                position_ids=position_ids,
                past_key_values=past_cache,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return outputs_embeds[1]
        out = self.denoise(
            {
                "suffix_embs": suffix_embs,
                "attention_mask": _engine_attention_mask(full_mask_4d),
                "position_ids": position_ids.int(),
                "adarms_cond": adarms_cond,
                "past_kv": past_kv_tensor,
            }
        )["suffix_hidden"]
        return out

    @torch.no_grad()
    def __call__(self, device, observation, noise=None, num_steps=10):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        model = self.model
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, model.config.action_horizon, model.config.action_dim)
            noise = model.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
            observation, train=False
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_mask_4d = model._prepare_attention_masks_4d(prefix_att_2d)
        past_cache, past_kv_tensor = self._prefix_kv(prefix_embs, prefix_mask_4d, prefix_pos)

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
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

            suffix_out = self._denoise_hidden(
                suffix_embs,
                full_mask_4d,
                position_ids,
                adarms_cond,
                past_cache,
                past_kv_tensor,
            )
            suffix_out = suffix_out[:, -model.config.action_horizon :].to(dtype=torch.float32)
            v_t = model.action_out_proj(suffix_out)
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t


def _maybe_runner(path: Path, device: str) -> TensorRTRunner | None:
    if path.is_file():
        logger.info("loading engine %s", path)
        return TensorRTRunner(path, device=device)
    logger.info("engine not found: %s", path)
    return None


def create_fp8_policy(
    train_config,
    checkpoint_dir: str,
    artifact_dir: str | Path | None = None,
    pytorch_device: str | None = None,
):
    from openpi.policies import policy_config as _policy_config

    if pytorch_device is None:
        pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        pytorch_device=pytorch_device,
    )
    sampler = Fp8ActionSampler(
        policy._model,
        artifact_dir=artifact_dir or ARTIFACTS,
        device=pytorch_device,
    )
    policy._sample_actions = sampler
    return policy
