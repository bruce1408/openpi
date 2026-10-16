#!/usr/bin/env bash
# Full official LIBERO eval of the converted PyTorch pi05_libero checkpoint.
set -euo pipefail
cd /share_data/bruce/workspace/ai/openpi
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/share_data/public/models/openpi/pi05_libero_pytorch}"
export RUN_TAG="${RUN_TAG:-pytorch}"
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  echo "===== $(date '+%F %T') START ${SUITE} ====="
  SUITE="${SUITE}" bash bf6_run_eval.sh
  echo "===== $(date '+%F %T') DONE ${SUITE} ====="
done
echo "===== $(date '+%F %T') ALL SUITES DONE ====="
