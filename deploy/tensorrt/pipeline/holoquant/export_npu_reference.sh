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

if [[ ! -x "$PY" ]]; then
  echo "找不到 OpenPI 解释器: $PY" >&2
  echo "请先在本仓库建好 .venv（Python 3.11 / torch 2.7）" >&2
  exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
  echo "找不到导出脚本: $SCRIPT" >&2
  exit 1
fi

# TensorRT：只认已设置的 TRT_ROOT，或 PATH 上的 trtexec，不写死本机目录。
TRT_LIB=""
if [[ -n "${TRT_ROOT:-}" && -d "${TRT_ROOT}/lib" ]]; then
  TRT_LIB="${TRT_ROOT}/lib"
elif command -v trtexec >/dev/null 2>&1; then
  _trt_guess="$(cd "$(dirname "$(command -v trtexec)")/../lib" && pwd)"
  if [[ -d "$_trt_guess" ]]; then
    TRT_LIB="$_trt_guess"
    export TRT_ROOT="$(cd "${TRT_LIB}/.." && pwd)"
  fi
fi

NVLIBS="$(find "${ROOT}/.venv/lib" -type d -name lib -path '*/nvidia/*' 2>/dev/null | paste -sd: || true)"
_ld_parts=()
[[ -n "$TRT_LIB" ]] && _ld_parts+=("$TRT_LIB")
[[ -n "$NVLIBS" ]] && _ld_parts+=("$NVLIBS")
[[ -n "${LD_LIBRARY_PATH:-}" ]] && _ld_parts+=("$LD_LIBRARY_PATH")
if ((${#_ld_parts[@]})); then
  export LD_LIBRARY_PATH="$(IFS=:; echo "${_ld_parts[*]}")"
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

echo "[export] python=$PY"
echo "[export] OPENPI_ROOT=$ROOT"
echo "[export] TRT_ROOT=${TRT_ROOT:-（未设置：export TRT_ROOT 或把 trtexec 加入 PATH）}"
exec "$PY" "$SCRIPT" "$@"
