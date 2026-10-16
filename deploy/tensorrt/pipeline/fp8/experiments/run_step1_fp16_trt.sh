#!/usr/bin/env bash
# 第 1 步：用 FP16 ONNX 编 TensorRT engine（不量化），再分段测 RMSE。
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
TRT_ROOT="${TRT_ROOT:-/share_data/bruce/software/TensorRT-11.2.1.2}"
export LD_LIBRARY_PATH="${TRT_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/share_data/public/models/openpi/pi05_libero_pytorch}"
OUT_DIR="${PWD}/deploy/tensorrt/pipeline/fp8/experiments"
PY="${PWD}/.venv/bin/python"

echo "[step1] build FP16 engines: vision,prefix,denoise"
PRECISION=fp16 SEGMENTS=vision,prefix,denoise \
  bash deploy/tensorrt/pipeline/fp8/04_build_engines.sh

for segs in vision prefix denoise; do
  echo "[step1] RMSE segments=${segs} precision=fp16"
  OPENPI_TRT_PRECISION=fp16 OPENPI_FP8_SEGMENTS="${segs}" \
  "${PY}" deploy/tensorrt/pipeline/fp8/05_compare_actions.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --device "${DEVICE}" \
    --segments "${segs}" \
    --precision fp16 \
    --out "${OUT_DIR}/s1_${segs}_fp16.json"
done
echo "[step1] done. json under ${OUT_DIR}/s1_*_fp16.json"
