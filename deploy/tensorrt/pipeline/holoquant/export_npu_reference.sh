#!/usr/bin/env bash
# 给 HoloQuant NPU 仿真导出 config / amax / 参考动作。
# 只写 json / npz，不 import HoloQuant。
#
# 用法（在 openpi 仓库根，或任意目录）:
#   ./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh
#   ./deploy/tensorrt/pipeline/holoquant/export_npu_reference.sh --trt

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
SCRIPT="${ROOT}/deploy/tensorrt/pipeline/holoquant/export_npu_reference.py"
TRT_ROOT="${TRT_ROOT:-/share_data/bruce/software/TensorRT-11.2.1.2}"

if [[ ! -x "$PY" ]]; then
  echo "找不到 OpenPI 解释器: $PY" >&2
  echo "请先在本仓库建好 .venv（Python 3.11 / torch 2.7）" >&2
  exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
  echo "找不到导出脚本: $SCRIPT" >&2
  exit 1
fi

NVLIBS="$(find "${ROOT}/.venv/lib" -type d -name lib -path '*/nvidia/*' 2>/dev/null | paste -sd: || true)"
export LD_LIBRARY_PATH="${TRT_ROOT}/lib${NVLIBS:+:$NVLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

echo "[export] python=$PY"
echo "[export] OPENPI_ROOT=$ROOT"
exec "$PY" "$SCRIPT" "$@"
