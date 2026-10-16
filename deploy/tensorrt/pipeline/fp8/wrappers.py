"""ONNX-friendly subgraphs for π₀.₅ (no HuggingFace DynamicCache).

为什么要自己写一层循环
----------------------
官方 ``GemmaModel.forward`` 用 ``DynamicCache`` 存 KV。那是一个 Python 对象
（里面是 list），ONNX 追踪时经常失败。

我们改成：

* Prefix：每层算出 RoPE 之后的 K/V，stack 成一个张量 ``[L, 2, B, H, S, D]``
* Denoise：把这个张量按层拆开，和当前 suffix 的 K/V 在 seq 维拼起来

数学上和 ``sample_actions`` / ``denoise_step`` 一致，只是存储格式换成了纯 Tensor。
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.gemma.modeling_gemma import (
    _gated_residual,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


def _one_decoder_layer(
    layer,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    adarms_cond: torch.Tensor | None,
    past_k: torch.Tensor | None,
    past_v: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Gemma layer. Returns (hidden, k_now, v_now) where k/v are *this* token's cache."""
    residual = hidden_states
    # adaRMS 的 dense 在 to_bfloat16_for_selected_params 里因名字含 input_layernorm
    # 被留在 float32；条件向量必须跟它同 dtype，否则 Linear 会报错。
    if adarms_cond is not None and getattr(layer.input_layernorm, "dense", None) is not None:
        adarms_cond = adarms_cond.to(dtype=layer.input_layernorm.dense.weight.dtype)
    hidden_states, gate = layer.input_layernorm(hidden_states, adarms_cond)

    attn = layer.self_attn
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    query_states = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Prefix 要存的是「当前 token 自己的 K/V」（RoPE 之后、和 past 拼接之前）。
    k_now, v_now = key_states, value_states
    if past_k is not None:
        key_states = torch.cat([past_k.to(dtype=k_now.dtype), k_now], dim=2)
        value_states = torch.cat([past_v.to(dtype=v_now.dtype), v_now], dim=2)

    attn_output, _ = eager_attention_forward(
        attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=attn.scaling,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    if attn_output.dtype != attn.o_proj.weight.dtype:
        attn_output = attn_output.to(attn.o_proj.weight.dtype)
    attn_output = attn.o_proj(attn_output)
    hidden_states = _gated_residual(residual, attn_output, gate)

    residual = hidden_states
    hidden_states, gate = layer.post_attention_layernorm(hidden_states, adarms_cond)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = _gated_residual(residual, hidden_states, gate)
    return hidden_states, k_now, v_now


def run_gemma_layers(
    gemma_model,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    adarms_cond: torch.Tensor | None = None,
    past_kv: torch.Tensor | None = None,
    *,
    return_kv: bool = False,
    apply_final_norm: bool = True,
):
    """Run all decoder layers with optional stacked past KV.

    past_kv 形状: ``[num_layers, 2, batch, kv_heads, prefix_len, head_dim]``
    """
    weight_dtype = gemma_model.layers[0].self_attn.q_proj.weight.dtype
    hidden_states = hidden_states.to(weight_dtype)
    attention_mask = attention_mask.to(hidden_states.dtype)
    position_ids = position_ids.long()

    position_embeddings = gemma_model.rotary_emb(hidden_states, position_ids)

    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for layer_idx, layer in enumerate(gemma_model.layers):
        past_k = past_kv[layer_idx, 0] if past_kv is not None else None
        past_v = past_kv[layer_idx, 1] if past_kv is not None else None
        hidden_states, k_now, v_now = _one_decoder_layer(
            layer,
            hidden_states,
            attention_mask,
            position_embeddings,
            adarms_cond,
            past_k,
            past_v,
        )
        if return_kv:
            keys.append(k_now)
            values.append(v_now)

    if apply_final_norm:
        hidden_states, _ = gemma_model.norm(hidden_states, adarms_cond)

    if return_kv:
        kv = torch.stack([torch.stack(keys, dim=0), torch.stack(values, dim=0)], dim=1)
        return hidden_states, kv
    return hidden_states, None


class VisionWrapper(nn.Module):
    """SigLIP + multimodal projector. Input ``[B, 3, 224, 224]`` → ``[B, 256, 2048]``."""

    def __init__(self, paligemma_with_expert: nn.Module):
        super().__init__()
        self.inner = paligemma_with_expert

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.inner.embed_image(pixel_values)


class PrefixWrapper(nn.Module):
    """Gemma-2B prefix: inputs_embeds → stacked KV cache."""

    def __init__(self, language_model: nn.Module):
        super().__init__()
        self.language_model = language_model
        language_model.config._attn_implementation = "eager"  # noqa: SLF001

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        _, past_kv = run_gemma_layers(
            self.language_model,
            inputs_embeds,
            attention_mask,
            position_ids,
            adarms_cond=None,
            past_kv=None,
            return_kv=True,
            apply_final_norm=False,
        )
        # [L, 2, B, H, S, D] → [B, L, 2, H, S, D]，方便 ModelOpt 把第 0 维当 batch。
        return past_kv.permute(2, 0, 1, 3, 4, 5).contiguous()


class DenoiseWrapper(nn.Module):
    """Gemma-300M action expert: suffix tokens + prefix KV + adaRMS cond → hidden states."""

    def __init__(self, expert_model: nn.Module):
        super().__init__()
        self.expert_model = expert_model
        expert_model.config._attn_implementation = "eager"  # noqa: SLF001

    def forward(
        self,
        suffix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        adarms_cond: torch.Tensor,
        past_kv: torch.Tensor,
    ) -> torch.Tensor:
        # Engine layout [B, L, 2, H, S, D] → internal [L, 2, B, H, S, D]
        past_kv = past_kv.permute(1, 2, 0, 3, 4, 5).contiguous()
        hidden, _ = run_gemma_layers(
            self.expert_model,
            suffix_embs,
            attention_mask,
            position_ids,
            adarms_cond=adarms_cond,
            past_kv=past_kv,
            return_kv=False,
            apply_final_norm=True,
        )
        return hidden
