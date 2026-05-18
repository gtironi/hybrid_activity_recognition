#!/usr/bin/env bash
# Pipeline: pretrain MAE no raw → patchtst_hybrid_frompretrain.
#
# Uso:
#   nohup bash scripts/experiments/run_patchtst_raw_pipeline.sh > logs/patchtst_raw_pipeline.log 2>&1 &
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== PatchTST raw pipeline start: $(date) ==="

echo ""
echo "--- [1/2] Pretrain MAE sobre RAW (Time_Adj_Raw_Data.csv) ---"
bash "${DIR}/pretrain_patchtst_raw.sh"

echo ""
echo "--- [2/2] patchtst_hybrid_frompretrain ---"
bash "${DIR}/run_patchtst_hybrid_frompretrain.sh"

echo ""
echo "=== PatchTST raw pipeline done: $(date) ==="
