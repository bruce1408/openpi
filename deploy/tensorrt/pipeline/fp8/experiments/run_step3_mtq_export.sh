#!/usr/bin/env bash
# 第 3 步：MTQ → 导出 {seg}_mtqfp8.onnx → trtexec → RMSE。默认只做 Prefix。
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

echo "[step3] export ${SCOPE} after MTQ"
"${PY}" deploy/tensorrt/pipeline/fp8/experiments/run_step3_mtq_export.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --device "${DEVICE}" \
  --scope "${SCOPE}" \
  2>&1 | tee "${OUT_DIR}/s3_${SCOPE}_export.log"

echo "[step3] build ${SCOPE}_mtqfp8.plan"
PRECISION=mtqfp8 SEGMENTS="${SCOPE}" \
  bash deploy/tensorrt/pipeline/fp8/04_build_engines.sh \
  2>&1 | tee -a "${OUT_DIR}/s3_${SCOPE}_export.log"

echo "[step3] RMSE ${SCOPE} mtqfp8"
"${PY}" deploy/tensorrt/pipeline/fp8/05_compare_actions.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --device "${DEVICE}" \
  --segments "${SCOPE}" \
  --precision mtqfp8 \
  --out "${OUT_DIR}/s3_${SCOPE}_mtqfp8.json" \
  2>&1 | tee "${OUT_DIR}/s3_${SCOPE}_mtqfp8.log"
echo "[step3] done"
