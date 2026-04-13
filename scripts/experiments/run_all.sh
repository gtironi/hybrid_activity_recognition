#!/usr/bin/env bash
# Master script: runs all experiments sequentially.
#
# For remote execution via SSH:
#   nohup bash scripts/experiments/run_all.sh > logs/run_all.log 2>&1 &
#   # or: screen -dmS exp bash scripts/experiments/run_all.sh
#   # or: tmux new -d -s exp 'bash scripts/experiments/run_all.sh'
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Full experiment pipeline ==="
echo "Start: $(date)"
echo ""

bash "${DIR}/run_cnn_lstm.sh"
bash "${DIR}/run_robust.sh"
bash "${DIR}/run_patchtst.sh"
bash "${DIR}/run_tsfel_baseline.sh"

echo ""
echo "=== All experiments complete: $(date) ==="
