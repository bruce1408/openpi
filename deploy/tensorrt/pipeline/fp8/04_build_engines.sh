#!/usr/bin/env bash
# Build TensorRT engines from ONNX.
# PRECISION=fp8（默认）：读 {seg}_fp8.onnx（含 Q/DQ）。
# PRECISION=fp16：读 {seg}_fp16.onnx，不插量化，用来对照「TRT/接口 vs FP8」。
# TensorRT 11 是 strongly-typed：不要传 --fp8；精度跟 ONNX 走。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
trt_root="${TRT_ROOT:-/share_data/bruce/software/TensorRT-11.2.1.2}"
onnx_dir="${ONNX_DIR:-${repo_root}/deploy/tensorrt/artifacts/onnx}"
engine_dir="${ENGINE_DIR:-${repo_root}/deploy/tensorrt/artifacts/engines}"
segments="${SEGMENTS:-vision,prefix,denoise}"
precision="${PRECISION:-fp8}"

if [[ -x "${trt_root}/bin/trtexec" ]]; then
  trtexec_bin="${trt_root}/bin/trtexec"
  export LD_LIBRARY_PATH="${trt_root}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif command -v trtexec >/dev/null 2>&1; then
  trtexec_bin="$(command -v trtexec)"
else
  echo "找不到 trtexec。请设置 TRT_ROOT=/share_data/bruce/software/TensorRT-11.2.1.2" >&2
  exit 1
fi

precision_flags=()
if [[ "${precision}" == "fp8" ]] && "${trtexec_bin}" --help 2>&1 | grep -qE '^[[:space:]]+--fp16'; then
  echo "[04] TensorRT 10.x style flags detected; adding --fp8 --fp16"
  precision_flags=(--fp8 --fp16)
else
  echo "[04] TensorRT 11+ strongly-typed: engine dtype follows ONNX (${precision})"
fi

mkdir -p "${engine_dir}"
IFS=',' read -r -a segs <<< "${segments}"
for seg in "${segs[@]}"; do
  seg="$(echo "${seg}" | xargs)"
  onnx="${onnx_dir}/${seg}_${precision}.onnx"
  engine="${engine_dir}/${seg}_${precision}.plan"
  if [[ ! -f "${onnx}" ]]; then
    echo "[04] skip ${seg}: missing ${onnx}" >&2
    continue
  fi
  echo "[04] building ${seg}: ${onnx} -> ${engine}"
  cmd=("${trtexec_bin}" --onnx="${onnx}" --saveEngine="${engine}" --skipInference --builderOptimizationLevel=5)
  if ((${#precision_flags[@]})); then
    cmd+=("${precision_flags[@]}")
  fi
  "${cmd[@]}"
  echo "[04] wrote ${engine}"
done

echo "[04] done"
