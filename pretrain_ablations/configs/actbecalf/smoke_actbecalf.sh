#!/usr/bin/env bash
# Smoke test for AcTBeCalf SSL ablation — 2 epochs, small model, all methods.
# Validates imports, data loading, and shapes end-to-end without real training.
#
# Usage:
#   bash pretrain_ablations/configs/actbecalf/smoke_actbecalf.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

PY="python -m pretrain_ablations.experiment"
BASE="--config pretrain_ablations/configs/actbecalf/actbecalf_smoke.yaml"

# Ensure data is exported.
if [ ! -f "pretrain_ablations/processed/actbecalf_windowed/train.pt" ]; then
    echo ">>> Exporting AcTBeCalf data..."
    python -m pretrain_ablations.preprocess.export_actbecalf
fi

echo "=== AcTBeCalf SSL smoke test start: $(date) ==="

echo ""
echo "=== P0: Supervised baselines ==="
$PY $BASE --override encoder.name=cnn_tfc      pretext.method=supervised finetune.mode=full run_name=smoke_sup_cnn
$PY $BASE --override encoder.name=resnet1d     pretext.method=supervised finetune.mode=full run_name=smoke_sup_resnet
$PY $BASE --override encoder.name=patchtst     pretext.method=supervised finetune.mode=full run_name=smoke_sup_ptst
$PY $BASE --override encoder.name=patchtsmixer pretext.method=supervised finetune.mode=full run_name=smoke_sup_pmix

echo ""
echo "=== P1: SSL methods ==="
$PY $BASE --override encoder.name=patchtst pretext.method=mae    finetune.mode=freeze run_name=smoke_mae_ptst
$PY $BASE --override encoder.name=cnn_tfc  pretext.method=simclr finetune.mode=freeze run_name=smoke_simclr_cnn
$PY $BASE --override encoder.name=cnn_tfc  pretext.method=tfc    finetune.mode=freeze run_name=smoke_tfc_cnn
$PY $BASE --override encoder.name=cnn_tfc  pretext.method=tstcc  finetune.mode=full   run_name=smoke_tstcc_cnn

echo ""
echo "=== P2: Frozen variants ==="
$PY $BASE --override encoder.name=cnn_tfc  pretext.method=tstcc  finetune.mode=freeze run_name=smoke_tstcc_cnn_freeze
$PY $BASE --override encoder.name=cnn_tfc  pretext.method=simclr finetune.mode=full   run_name=smoke_simclr_cnn_full

echo ""
echo "=== AcTBeCalf SSL smoke test done: $(date) ==="
