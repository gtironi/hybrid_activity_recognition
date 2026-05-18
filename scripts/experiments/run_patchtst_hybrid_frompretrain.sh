#!/usr/bin/env bash
# Roda APENAS patchtst_hybrid_frompretrain — supervisionado + finetune.
# Requer: checkpoint do pretrain raw em PATCHTST_CHECKPOINT (ou em make_run_dir patchtst pretrain com RUN_SUFFIX=raw).
set -euo pipefail
source "$(dirname "$0")/_common.sh"
export BATCH_SIZE="${PATCHTST_BATCH_SIZE:-128}"

# Resolve o checkpoint do pretrain raw se não passado.
PATCHTST_CHECKPOINT="${PATCHTST_CHECKPOINT:-}"
if [ -z "$PATCHTST_CHECKPOINT" ]; then
    RUN_SUFFIX="raw" PRETRAIN_DIR=$(RUN_SUFFIX="raw" make_run_dir "patchtst" "pretrain")
    PATCHTST_CHECKPOINT="${PRETRAIN_DIR}/best.pt"
fi

if [ ! -f "$PATCHTST_CHECKPOINT" ]; then
    echo "ERROR: PatchTST checkpoint não encontrado: $PATCHTST_CHECKPOINT" >&2
    exit 1
fi

echo ">>> Usando pretrain checkpoint: $PATCHTST_CHECKPOINT"
echo ">>> Rodando patchtst_hybrid_frompretrain"

export RUN_SUFFIX=frompretrain
unset FREEZE_ENCODER || true
run_experiment "patchtst" "hybrid" --patchtst_checkpoint "$PATCHTST_CHECKPOINT"
run_finetune   "patchtst" "hybrid"
unset RUN_SUFFIX

echo ">>> Done."
