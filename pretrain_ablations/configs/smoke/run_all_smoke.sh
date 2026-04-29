#!/usr/bin/env bash
# One-shot smoke runner — validates every component in ~5min on CPU.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"
PY=".venv/bin/python -m pretrain_ablations.experiment"
SMOKE="--config pretrain_ablations/configs/smoke/smoke_base.yaml"

echo "=== P0: Supervised baselines ==="
$PY $SMOKE --override encoder.name=cnn_tfc      pretext.method=supervised finetune.mode=full     run_name=smoke_sup_cnn
$PY $SMOKE --override encoder.name=resnet1d     pretext.method=supervised finetune.mode=full     run_name=smoke_sup_resnet
$PY $SMOKE --override encoder.name=patchtst     pretext.method=supervised finetune.mode=full     run_name=smoke_sup_ptst
$PY $SMOKE --override encoder.name=patchtsmixer pretext.method=supervised finetune.mode=full     run_name=smoke_sup_pmix

echo "=== P1: Pretraining methods ==="
$PY $SMOKE --override encoder.name=patchtst     pretext.method=mae        finetune.mode=freeze   run_name=smoke_mae_ptst
$PY $SMOKE --override encoder.name=cnn_tfc      pretext.method=simclr     finetune.mode=freeze   run_name=smoke_simclr_cnn
$PY $SMOKE --override encoder.name=cnn_tfc      pretext.method=tfc        finetune.mode=freeze   run_name=smoke_tfc_cnn
$PY $SMOKE --override encoder.name=cnn_tfc      pretext.method=tstcc      finetune.mode=full     run_name=smoke_tstcc_cnn

echo "=== Compatibility guards (must fail) ==="
! $PY $SMOKE --override encoder.name=cnn_tfc pretext.method=mae        run_name=neg_mae_cnn       2>&1 | grep -q ValueError
! $PY $SMOKE --override              pretext.method=supervised finetune.mode=freeze run_name=neg_sup_freeze 2>&1 | grep -q ValueError
! $PY $SMOKE --override data.dataset_id=etth run_name=neg_etth        2>&1 | grep -q ValueError

echo "=== Aggregation ==="
.venv/bin/python -m pretrain_ablations.results.summarize --filter smoke_

echo ""
echo "ALL SMOKE TESTS PASSED"
