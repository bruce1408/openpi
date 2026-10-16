"""Shared paths, tensor shapes, and helpers for the π₀.₅ TensorRT FP8 pipeline.

π₀.₅-LIBERO 推理可以想成三条流水线：

1. **Vision（SigLIP）**：三张 224×224 图 → 每张 256 个 2048 维 token
2. **Prefix Gemma-2B**：图像 token + 语言 token → 18 层 KV cache
3. **Denoise expert（Gemma-300M）**：在 KV cache 上做 10 步 flow matching

时间 MLP、action 投影、Euler 积分、反归一化都留在 Python 里（算量很小，
不必进 TensorRT）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = Path("/share_data/public/models/openpi/pi05_libero_pytorch")
DEFAULT_CONFIG = "pi05_libero"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ONNX_DIR = ARTIFACTS / "onnx"
CALIB_DIR = ARTIFACTS / "calib"
ENGINE_DIR = ARTIFACTS / "engines"
TRT_ROOT = Path(os.environ.get("TRT_ROOT", "/share_data/bruce/software/TensorRT-11.2.1.2"))

# π₀.₅-LIBERO 固定形状（评测时 batch=1，三相机，语言 200 token，动作 horizon=10）
NUM_VIEWS = 3
IMAGE_SIZE = 224
VISION_TOKENS = 256  # SigLIP 224/14 的 patch 数（含 CLS 后 PaliGemma 用 256）
LANG_LEN = 200
PREFIX_LEN = NUM_VIEWS * VISION_TOKENS + LANG_LEN  # 768 + 200 = 968
ACTION_HORIZON = 10
ACTION_DIM = 32
SUFFIX_LEN = ACTION_HORIZON  # pi05 没有连续 state token，suffix 就是 10 个动作 token
NUM_LAYERS = 18
NUM_KV_HEADS = 1
NUM_ATTN_HEADS = 8
HEAD_DIM = 256
VLM_WIDTH = 2048
EXPERT_WIDTH = 1024
DENOISE_KV_LEN = PREFIX_LEN + SUFFIX_LEN  # 978


def ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def inductor_cache_env() -> None:
    cache = Path.home() / ".cache" / "torchinductor"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache))


def load_pi05_pytorch(
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT,
    device: str = "cuda:0",
    config_name: str = DEFAULT_CONFIG,
):
    """Load the converted PyTorch π₀.₅-LIBERO weights. Compile is always off."""
    ensure_repo_on_path()
    inductor_cache_env()
    from openpi.training import config as train_config

    checkpoint_dir = Path(checkpoint_dir)
    weight_path = checkpoint_dir / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"找不到 {weight_path}。请先把 JAX checkpoint 转成 PyTorch safetensors。"
        )

    cfg = train_config.get_config(config_name)
    object.__setattr__(cfg.model, "pytorch_compile_mode", None)
    model = cfg.model.load_pytorch(cfg, str(weight_path))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.eval()
    model.to(device)
    return model, cfg


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def export_onnx(
    module: torch.nn.Module,
    dummy_inputs: tuple,
    out_path: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int = 19,
) -> None:
    """Export a static-shape ONNX graph. Large weights go to an adjacent ``.data`` file.

    ONNX protobuf 单文件上限约 2GB，Gemma-2B 的 fp16 权重大于这个限制，
    所以一律写成 external data。追踪过程在临时目录里完成，避免把成百个
    ``onnx__MatMul_*`` 碎片留在 artifacts 目录。
    """
    import tempfile

    import onnx

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_file = out_path.name + ".data"
    for leftover in (out_path, out_path.parent / data_file):
        if leftover.exists():
            leftover.unlink()

    module.eval()
    with tempfile.TemporaryDirectory(prefix="openpi_onnx_") as tmp:
        tmp_path = Path(tmp) / "model.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                module,
                dummy_inputs if len(dummy_inputs) > 1 else dummy_inputs[0],
                str(tmp_path),
                input_names=input_names,
                output_names=output_names,
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,
            )
        model_proto = onnx.load(str(tmp_path), load_external_data=True)
        onnx.save_model(
            model_proto,
            str(out_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file,
            size_threshold=1024,
        )
    print(f"[export] wrote {out_path} (+ {data_file})")


def cache_to_kv_tensor(cache, num_layers: int = NUM_LAYERS) -> torch.Tensor:
    """HuggingFace DynamicCache → internal layout ``[L, 2, B, H, S, D]``."""
    keys = torch.stack([cache.key_cache[i] for i in range(num_layers)], dim=0)
    values = torch.stack([cache.value_cache[i] for i in range(num_layers)], dim=0)
    return torch.stack([keys, values], dim=1)


def kv_internal_to_engine(past_kv: torch.Tensor) -> torch.Tensor:
    """``[L, 2, B, H, S, D]`` → engine/ONNX layout ``[B, L, 2, H, S, D]``.

    ModelOpt 校准把输入第 0 维当 batch。内部 layout 第 0 维是层数，必须换轴。
    """
    return past_kv.permute(2, 0, 1, 3, 4, 5).contiguous()


def kv_engine_to_internal(past_kv: torch.Tensor) -> torch.Tensor:
    """``[B, L, 2, H, S, D]`` → ``[L, 2, B, H, S, D]``."""
    return past_kv.permute(1, 2, 0, 3, 4, 5).contiguous()


def kv_tensor_to_cache(past_kv: torch.Tensor, dtype: torch.dtype | None = None):
    """Engine or internal KV tensor → DynamicCache (for hybrid TRT prefix + PyTorch denoise)."""
    from transformers.cache_utils import DynamicCache

    if past_kv.ndim != 6:
        raise ValueError(f"past_kv should be 6D, got {tuple(past_kv.shape)}")
    # Engine layout has batch at dim 0 and a small batch (1). Internal has layers at dim 0 (18).
    if past_kv.shape[0] == NUM_LAYERS:
        internal = past_kv
    else:
        internal = kv_engine_to_internal(past_kv)
    cache = DynamicCache()
    for i in range(internal.shape[0]):
        key = internal[i, 0].contiguous()
        value = internal[i, 1].contiguous()
        # TRT 输出是 fp16；PyTorch expert 是 bf16。混用时必须对齐。
        if dtype is not None:
            key = key.to(dtype)
            value = value.to(dtype)
        cache.update(key, value, layer_idx=i)
    return cache
