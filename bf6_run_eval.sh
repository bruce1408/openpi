#!/usr/bin/env bash
# Local LIBERO bf16 eval on 8 GPUs (no policy server).
# Each GPU loads its own copy of pi05_libero and runs a shard of episodes.
set -euo pipefail

cd /share_data/bruce/workspace/ai/openpi

export PYTHONPATH="/share_data/bruce/workspace/ai/openpi/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"   # if EGL fails: MUJOCO_GL=glx bash bf6_run.sh
# Shared /tmp/torchinductor_bruce is owned by another user; keep torch.compile cache in $HOME.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
# Triton/torch.compile looks for libcuda.so (not libcuda.so.1). CUDA stubs provide it.
if [[ -d /usr/local/cuda/lib64/stubs ]]; then
  export LIBRARY_PATH="/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi

NUM_GPUS="${NUM_GPUS:-8}"
SUITE="${SUITE:-libero_spatial}"
TRIALS="${TRIALS:-50}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/share_data/public/models/openpi/pi05_libero}"
RUN_TAG="${RUN_TAG:-}"
BACKEND="${BACKEND:-}"
FP8_ARTIFACT_DIR="${FP8_ARTIFACT_DIR:-}"

if [[ "${BACKEND}" == "fp8" ]]; then
  TRT_ROOT="${TRT_ROOT:-/share_data/bruce/software/TensorRT-11.2.1.2}"
  export LD_LIBRARY_PATH="${TRT_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
fi

EXTRA_ARGS=()
if [[ -n "${BACKEND}" ]]; then
  EXTRA_ARGS+=(--args.backend "${BACKEND}")
fi
if [[ -n "${FP8_ARTIFACT_DIR}" ]]; then
  EXTRA_ARGS+=(--args.fp8-artifact-dir "${FP8_ARTIFACT_DIR}")
fi
if [[ -n "${OPENPI_FP8_SEGMENTS:-}" ]]; then
  export OPENPI_FP8_SEGMENTS
fi
if [[ -n "${OPENPI_TRT_PRECISION:-}" ]]; then
  export OPENPI_TRT_PRECISION
fi

PY="${PWD}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[bf6_run] missing $PY ; run: GIT_LFS_SKIP_SMUDGE=1 uv sync" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PWD}/data/libero/logs_${SUITE}${RUN_TAG:+_${RUN_TAG}}_${STAMP}"
VIDEO_ROOT="${PWD}/data/libero/videos_${SUITE}${RUN_TAG:+_${RUN_TAG}}"
mkdir -p "${LOG_DIR}"

echo "[bf6_run] python=$PY"
echo "[bf6_run] gpus=${NUM_GPUS} suite=${SUITE} trials=${TRIALS}"
echo "[bf6_run] checkpoint=${CHECKPOINT_DIR}"
echo "[bf6_run] logs=${LOG_DIR}"
if [[ "${BACKEND}" == "fp8" ]]; then
  echo "[bf6_run] backend=fp8 segments=${OPENPI_FP8_SEGMENTS:-vision,prefix,denoise} precision=${OPENPI_TRT_PRECISION:-mtqfp8}"
fi
"$PY" -c "from libero.libero import benchmark; print('[bf6_run] libero ok')"

pids=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  echo "[bf6_run] launching shard ${gpu}/${NUM_GPUS} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  MUJOCO_EGL_DEVICE_ID="${gpu}" \
  "${PY}" examples/libero/main_local.py \
    --args.config-name pi05_libero \
    --args.checkpoint-dir "${CHECKPOINT_DIR}" \
    --args.task-suite-name "${SUITE}" \
    --args.num-trials-per-task "${TRIALS}" \
    --args.shard-id "${gpu}" \
    --args.num-shards "${NUM_GPUS}" \
    --args.video-out-path "${VIDEO_ROOT}/rank${gpu}" \
    --args.results-json "${LOG_DIR}/rank${gpu}.json" \
    "${EXTRA_ARGS[@]}" \
    > "${LOG_DIR}/rank${gpu}.log" 2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    echo "[bf6_run] worker pid ${pid} failed"
    fail=1
  fi
done

echo "[bf6_run] aggregating results from ${LOG_DIR}"
"${PY}" - <<PY
import json, glob, pathlib
log_dir = pathlib.Path("${LOG_DIR}")
files = sorted(log_dir.glob("rank*.json"))
episodes = successes = 0
print("shard  episodes  successes  rate")
for f in files:
    d = json.loads(f.read_text())
    episodes += d["episodes"]
    successes += d["successes"]
    print(f"{d['shard_id']:>5}  {d['episodes']:>8}  {d['successes']:>9}  {d['success_rate']:.4f}")
rate = (successes / episodes) if episodes else 0.0
summary = {
    "suite": "${SUITE}",
    "trials_per_task": ${TRIALS},
    "num_gpus": ${NUM_GPUS},
    "checkpoint_dir": "${CHECKPOINT_DIR}",
    "run_tag": "${RUN_TAG}",
    "backend": "${BACKEND}",
    "openpi_fp8_segments": "${OPENPI_FP8_SEGMENTS:-}",
    "openpi_trt_precision": "${OPENPI_TRT_PRECISION:-}",
    "episodes": episodes,
    "successes": successes,
    "success_rate": rate,
}
(log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print("---")
print(f"Total episodes: {episodes}")
print(f"Total successes: {successes}")
print(f"Total success rate: {rate:.4f} ({100*rate:.1f}%)")
print(f"Wrote {log_dir / 'summary.json'}")
PY

if [[ "${fail}" -ne 0 ]]; then
  echo "[bf6_run] some workers failed; check ${LOG_DIR}/rank*.log" >&2
  exit 1
fi
