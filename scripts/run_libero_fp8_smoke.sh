#!/usr/bin/env bash
# FP8 冒烟：1 个 LIBERO spatial 任务 × 2 条 episode，单卡。
set -euo pipefail

cd /share_data/bruce/workspace/ai/openpi

TRT_ROOT="${TRT_ROOT:-/share_data/bruce/software/TensorRT-11.2.1.2}"
export LD_LIBRARY_PATH="${TRT_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

PY="${PWD}/.venv/bin/python"
DEVICE="${DEVICE:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/share_data/public/models/openpi/pi05_libero_pytorch}"
ARTIFACT="${FP8_ARTIFACT_DIR:-${PWD}/deploy/tensorrt/artifacts}"

# 当前方案：MTQ 导出的 mtqfp8 engine（三段全开）。旧 ONNX PTQ 的 prefix/denoise 不要用。
export OPENPI_TRT_PRECISION="${OPENPI_TRT_PRECISION:-mtqfp8}"
export OPENPI_FP8_SEGMENTS="${OPENPI_FP8_SEGMENTS:-vision,prefix,denoise}"
echo "[fp8-smoke] device=${DEVICE} artifact=${ARTIFACT} precision=${OPENPI_TRT_PRECISION} segments=${OPENPI_FP8_SEGMENTS}"
CUDA_VISIBLE_DEVICES="${DEVICE}" \
MUJOCO_EGL_DEVICE_ID="${DEVICE}" \
"${PY}" examples/libero/main_local.py \
  --args.config-name pi05_libero \
  --args.checkpoint-dir "${CHECKPOINT_DIR}" \
  --args.backend fp8 \
  --args.fp8-artifact-dir "${ARTIFACT}" \
  --args.task-suite-name libero_spatial \
  --args.max-tasks 1 \
  --args.num-trials-per-task 2 \
  --args.video-out-path data/libero/videos_fp8_smoke \
  --args.results-json data/libero/logs_fp8_smoke.json

echo "[fp8-smoke] 看 data/libero/logs_fp8_smoke.json"
