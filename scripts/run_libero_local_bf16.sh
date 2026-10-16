#!/usr/bin/env bash
# In-process LIBERO bf16 eval (no policy server).
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="${PWD}/third_party/libero:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/share_data/bruce/openpi_cache}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/share_data/public/models/openpi/pi05_libero}"
SUITE="${SUITE:-libero_spatial}"
TRIALS="${TRIALS:-2}"

exec uv run python examples/libero/main_local.py \
  --args.config-name pi05_libero \
  --args.checkpoint-dir "${CHECKPOINT_DIR}" \
  --args.task-suite-name "${SUITE}" \
  --args.num-trials-per-task "${TRIALS}" \
  "$@"
