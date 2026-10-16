#!/usr/bin/env bash
# 第 2 步：PyTorch MTQ 假量化（先 Prefix Linear，再 Denoise Linear）。
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"

ARCH_TORCH="${ARCH_TORCH:-/tmp/cursor-sandbox-cache/6dd47ce2163efa45a1a505b8b9ebfcec/uv/archive-v0/P7L08tdOJkuexWFa}"
if [[ -d "${ARCH_TORCH}/torch" ]]; then
  export PYTHONPATH="${ARCH_TORCH}:${PWD}${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
fi
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
TRT_ROOT="${TRT_ROOT:-/share_data/bruce/software/TensorRT-11.2.1.2}"
NVLIBS="$(find .venv/lib/python3.11/site-packages/nvidia -type d -name lib 2>/dev/null | paste -sd:)"
export LD_LIBRARY_PATH="${TRT_ROOT}/lib${NVLIBS:+:$NVLIBS}"

DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/share_data/public/models/openpi/pi05_libero_pytorch}"
OUT_DIR="${PWD}/deploy/tensorrt/pipeline/fp8/experiments"
PY="${PWD}/.venv/bin/python"
SCOPE="${1:-prefix}"

echo "[step2] torch=$("${PY}" -c 'import torch; print(torch.__version__)')"
echo "[step2] scope=${SCOPE} device=${DEVICE}"
"${PY}" deploy/tensorrt/pipeline/fp8/experiments/run_step2_mtq_fakequant.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --device "${DEVICE}" \
  --scope "${SCOPE}" \
  --out "${OUT_DIR}/s2_${SCOPE}_mtq.json" \
  2>&1 | tee "${OUT_DIR}/s2_${SCOPE}_mtq.log"
echo "[step2] wrote ${OUT_DIR}/s2_${SCOPE}_mtq.json"
