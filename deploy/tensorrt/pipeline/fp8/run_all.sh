#!/usr/bin/env bash
# 旧 ONNX PTQ 流水线（01→05）。Prefix/Denoise 这条路径 RMSE ~0.26，不要当当前部署。
# 当前可用方案：deploy/tensorrt/pipeline/fp8/experiments/run_step3_mtq_export.sh
#               → *_mtqfp8.plan，OPENPI_TRT_PRECISION=mtqfp8
#
# 任一步失败会停；可以 SKIP_STEPS="03,04" 之类跳过。
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

TRT_ROOT="${TRT_ROOT:-/share_data/bruce/software/TensorRT-11.2.1.2}"
export TRT_ROOT
export LD_LIBRARY_PATH="${TRT_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

PY="${PY:-${PWD}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/share_data/public/models/openpi/pi05_libero_pytorch}"
SKIP_STEPS="${SKIP_STEPS:-}"
NUM_CALIB="${NUM_CALIB:-32}"

should_run() {
  local step="$1"
  [[ ",${SKIP_STEPS}," != *",${step},"* ]]
}

echo "[run_all] python=$PY"
echo "[run_all] device=$DEVICE"
echo "[run_all] checkpoint=$CHECKPOINT_DIR"

if should_run 01; then
  echo ""
  echo "======== 01 导出三个 ONNX（vision / prefix / denoise）========"
  "$PY" deploy/tensorrt/pipeline/fp8/01_export_onnx.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --device "${DEVICE}"
fi

if should_run 02; then
  echo ""
  echo "======== 02 导出 FP8 校准张量 ========"
  "$PY" deploy/tensorrt/pipeline/fp8/02_dump_calib.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --device "${DEVICE}" \
    --num-calib "${NUM_CALIB}"
fi

if should_run 03; then
  echo ""
  echo "======== 03 ModelOpt FP8 PTQ ========"
  QUANT_PY="${QUANT_PY:-}"
  if [[ -z "${QUANT_PY}" ]]; then
    if "$PY" -c "import modelopt.onnx.quantization" >/dev/null 2>&1; then
      QUANT_PY="$PY"
    else
      for cand in \
        "${HOME}/miniconda3/envs/torch270_128/bin/python" \
        "/share_data/bruce/miniconda3/envs/torch270_128/bin/python" \
        "/opt/conda/envs/torch270_128/bin/python"; do
        if [[ -x "$cand" ]] && "$cand" -c "import modelopt.onnx.quantization" >/dev/null 2>&1; then
          QUANT_PY="$cand"
          break
        fi
      done
    fi
  fi
  if [[ -z "${QUANT_PY}" ]]; then
    echo "[run_all] 找不到 nvidia-modelopt。请设置 QUANT_PY=.../python 后重跑，或 SKIP_STEPS=03,04,05" >&2
    exit 1
  fi
  echo "[run_all] quantize python=${QUANT_PY}"
  PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}" \
  LD_LIBRARY_PATH="${TRT_ROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "${QUANT_PY}" deploy/tensorrt/pipeline/fp8/03_quantize_fp8.py \
      --calibration-eps cuda:0 cpu
fi

if should_run 04; then
  echo ""
  echo "======== 04 trtexec 编 engine ========"
  bash deploy/tensorrt/pipeline/fp8/04_build_engines.sh
fi

if should_run 05; then
  echo ""
  echo "======== 05 对照 PyTorch 动作 RMSE ========"
  # 这条流水线编的是旧 *_fp8.plan，必须显式 --precision fp8，不要落到 mtqfp8。
  COMPARE_SEGMENTS="${COMPARE_SEGMENTS:-vision,prefix,denoise}"
  echo "[run_all] compare segments=${COMPARE_SEGMENTS} precision=fp8"
  OPENPI_FP8_SEGMENTS="${COMPARE_SEGMENTS}" \
  OPENPI_TRT_PRECISION=fp8 \
  "$PY" deploy/tensorrt/pipeline/fp8/05_compare_actions.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --device "${DEVICE}" \
    --segments "${COMPARE_SEGMENTS}" \
    --precision fp8
fi

echo ""
echo "[run_all] 注意：03 ONNX PTQ 是旧路径（Prefix/Denoise RMSE~0.26）。"
echo "当前可用方案是 MTQ 导出的 mtqfp8 engine。冒烟："
echo "  OPENPI_TRT_PRECISION=mtqfp8 OPENPI_FP8_SEGMENTS=vision,prefix,denoise \\"
echo "  bash scripts/run_libero_fp8_smoke.sh"
echo "8 卡评测："
echo "  BACKEND=fp8 OPENPI_TRT_PRECISION=mtqfp8 OPENPI_FP8_SEGMENTS=vision,prefix,denoise \\"
echo "  FP8_ARTIFACT_DIR=deploy/tensorrt/artifacts CHECKPOINT_DIR=${CHECKPOINT_DIR} \\"
echo "  SUITE=libero_spatial TRIALS=50 RUN_TAG=mtqfp8 bash bf6_run_eval.sh"
