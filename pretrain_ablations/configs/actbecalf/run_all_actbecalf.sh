#!/usr/bin/env bash
# AcTBeCalf ablation: supervised baselines + SSL methods.
# Mirrors configs/smoke/run_all_smoke.sh but on the calf dataset with real epochs.
#
# Prerequisite: export the dataset once with
#   python -m pretrain_ablations.preprocess.export_actbecalf
# (writes pretrain_ablations/processed/actbecalf_windowed/{train,val,test}.pt)
#
# Usage:
#   bash pretrain_ablations/configs/actbecalf/run_all_actbecalf.sh
#   nohup bash pretrain_ablations/configs/actbecalf/run_all_actbecalf.sh > logs/actbecalf_ssl.log 2>&1 &
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

PY="python -m pretrain_ablations.experiment"
BASE="--config pretrain_ablations/configs/actbecalf_base.yaml"

# Ensure data is exported.
if [ ! -f "pretrain_ablations/processed/actbecalf_windowed/train.pt" ]; then
    echo ">>> Exporting AcTBeCalf data..."
    python -m pretrain_ablations.preprocess.export_actbecalf
fi

echo "=== AcTBeCalf ablation start: $(date) ==="

echo ""
echo "=== P0: Supervised baselines (full finetune) ==="
$PY $BASE --override encoder.name=cnn_tfc      pretext.method=supervised finetune.mode=full run_name=actbecalf_sup_cnn
$PY $BASE --override encoder.name=resnet1d     pretext.method=supervised finetune.mode=full run_name=actbecalf_sup_resnet
$PY $BASE --override encoder.name=patchtst     pretext.method=supervised finetune.mode=full run_name=actbecalf_sup_ptst
$PY $BASE --override encoder.name=patchtsmixer pretext.method=supervised finetune.mode=full run_name=actbecalf_sup_pmix

echo ""
echo "=== P1: SSL pretraining methods ==="
# MAE on PatchTST (frozen) — known to be weak on short windows, kept for comparison.
$PY $BASE --override encoder.name=patchtst pretext.method=mae    finetune.mode=freeze run_name=actbecalf_mae_ptst
# SimCLR on CNN-TFC (frozen) — strong on HAR.
$PY $BASE --override encoder.name=cnn_tfc  pretext.method=simclr finetune.mode=freeze run_name=actbecalf_simclr_cnn
# TFC on CNN-TFC (frozen).
$PY $BASE --override encoder.name=cnn_tfc  pretext.method=tfc    finetune.mode=freeze run_name=actbecalf_tfc_cnn
# TSTCC on CNN-TFC (full finetune) — dominated HAR.
$PY $BASE --override encoder.name=cnn_tfc  pretext.method=tstcc  finetune.mode=full   run_name=actbecalf_tstcc_cnn

echo ""
echo "=== P2: SSL + frozen variants for SimCLR/TFC/TSTCC consistency ==="
# Also test TSTCC frozen — does the contrastive representation hold without finetune?
$PY $BASE --override encoder.name=cnn_tfc  pretext.method=tstcc  finetune.mode=freeze run_name=actbecalf_tstcc_cnn_freeze
# SimCLR full finetune — does fine-tuning add over frozen?
$PY $BASE --override encoder.name=cnn_tfc  pretext.method=simclr finetune.mode=full   run_name=actbecalf_simclr_cnn_full

echo ""
echo "=== Aggregation ==="
python -m pretrain_ablations.results.summarize --filter actbecalf_

echo ""
echo "=== AcTBeCalf ablation done: $(date) ==="
