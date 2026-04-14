#!/usr/bin/env bash
# Smoke test: runs every experiment for 2 epochs to verify everything connects.
# No real training — just validates that imports, shapes, and I/O work end-to-end.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# Override for quick test
export EPOCHS=2
export PRETRAIN_EPOCHS=2
export BATCH_SIZE=16

echo "=== Smoke test (2 epochs, batch_size=16) ==="
echo "Start: $(date)"
echo ""

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CNN+LSTM
bash "${DIR}/run_cnn_lstm.sh"

# Robust
bash "${DIR}/run_robust.sh"

# PatchTST (pretrain + supervised)
bash "${DIR}/run_patchtst.sh"

# TSFEL baseline
bash "${DIR}/run_tsfel_baseline.sh"

# TSFEL + MLP head baseline
bash "${DIR}/run_tsfel_mlp.sh"

echo ""
echo "=== Smoke test complete: $(date) ==="
echo "Check experiments/ for DONE markers."
