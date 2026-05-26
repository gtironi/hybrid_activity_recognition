#!/usr/bin/env bash
# PatchTST experiments (uses smaller batch via PATCHTST_BATCH_SIZE in _common).
#
# Always runs (random init, no MAE checkpoint):
#   - supervised deep_only + hybrid
#   - deep_only + --head patchtst_hf
#
# If SKIP_PRETRAIN=1 and PATCHTST_CHECKPOINT is unset: stops after the above.
# Otherwise runs MAE pretrain (unless PATCHTST_CHECKPOINT is already set), then:
#   - supervised from pretrained encoder (deep_only + hybrid)
#   - deep_only + patchtst_hf + pretrained backbone
#   - same two modes with --freeze_encoder (head / TSFEL / fusion only)
#
# Env: SKIP_PRETRAIN=1 | PATCHTST_CHECKPOINT=/path/to/best.pt
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${DIR}/_common.sh"
export BATCH_SIZE="${PATCHTST_BATCH_SIZE:-128}"

SKIP_PRETRAIN="${SKIP_PRETRAIN:-0}"
PATCHTST_CHECKPOINT="${PATCHTST_CHECKPOINT:-}"

echo ">>> PatchTST: from scratch (no MAE checkpoint)"
export RUN_SUFFIX=fromscratch
unset FREEZE_ENCODER || true
for MODE in deep_only hybrid; do
    run_experiment "patchtst" "$MODE"
    run_finetune   "patchtst" "$MODE"
done

export RUN_SUFFIX=fromscratch_hf
run_experiment "patchtst" "deep_only" --head patchtst_hf
unset RUN_SUFFIX

if [ -z "$PATCHTST_CHECKPOINT" ] && [ "$SKIP_PRETRAIN" = "1" ]; then
    echo ">>> SKIP_PRETRAIN=1 — skipping MAE and all runs that need a checkpoint."
    exit 0
fi

if [ -z "$PATCHTST_CHECKPOINT" ]; then
    unset RUN_SUFFIX || true
    bash "${DIR}/pretrain_patchtst.sh"
    PRETRAIN_DIR="$(make_run_dir "patchtst" "pretrain")"
    PATCHTST_CHECKPOINT="${PRETRAIN_DIR}/best.pt"
fi

if [ ! -f "$PATCHTST_CHECKPOINT" ]; then
    echo "ERROR: PatchTST checkpoint not found: $PATCHTST_CHECKPOINT" >&2
    exit 1
fi

echo ">>> PatchTST: supervised from MAE checkpoint (full encoder training)"
export RUN_SUFFIX=frompretrain
unset FREEZE_ENCODER || true
for MODE in deep_only hybrid; do
    run_experiment "patchtst" "$MODE" --patchtst_checkpoint "$PATCHTST_CHECKPOINT"
    run_finetune   "patchtst" "$MODE"
done

echo ">>> PatchTST: HF classification head + MAE checkpoint"
export RUN_SUFFIX=frompretrain_hf
run_experiment "patchtst" "deep_only" --head patchtst_hf --patchtst_checkpoint "$PATCHTST_CHECKPOINT"
unset RUN_SUFFIX

echo ">>> PatchTST: supervised from MAE with encoder frozen"
export RUN_SUFFIX=frompretrain_encfrozen
export FREEZE_ENCODER=1
for MODE in deep_only hybrid; do
    run_experiment "patchtst" "$MODE" --patchtst_checkpoint "$PATCHTST_CHECKPOINT"
done
unset FREEZE_ENCODER
unset RUN_SUFFIX

# ---------------------------------------------------------------------------
# Extra runs: MAE pretrained on RAW (unlabeled) data.
#   - Full encoder training (deep_only + hybrid, stage1 + finetune): ep10, ep25, ep40
#   - HF head and encoder-frozen variants: ep40 only
# ---------------------------------------------------------------------------
RAW_PRETRAIN_DIR="${REPO_ROOT}/experiments/patchtst_pretrain_raw_AcTBeCalf_ep40_bs1536_lr1e-3_s2026"

for EP in 10 25 40; do
    RAW_CKPT="${RAW_PRETRAIN_DIR}/pretrain_ep${EP}.pt"
    if [ ! -f "$RAW_CKPT" ]; then
        echo ">>> SKIP raw_ep${EP}: checkpoint not found at ${RAW_CKPT}"
        continue
    fi

    echo ">>> PatchTST: from RAW MAE checkpoint ep${EP} (full encoder training)"
    export RUN_SUFFIX="frompretrain_raw_ep${EP}"
    unset FREEZE_ENCODER || true
    for MODE in deep_only hybrid; do
        run_experiment "patchtst" "$MODE" --patchtst_checkpoint "$RAW_CKPT"
        run_finetune   "patchtst" "$MODE"
    done
done

# HF head and encoder-frozen variants: only with the last raw checkpoint (ep40).
RAW_CKPT_LAST="${RAW_PRETRAIN_DIR}/pretrain_ep40.pt"
if [ -f "$RAW_CKPT_LAST" ]; then
    echo ">>> PatchTST: HF head + RAW MAE ep40"
    export RUN_SUFFIX="frompretrain_raw_ep40_hf"
    unset FREEZE_ENCODER || true
    run_experiment "patchtst" "deep_only" --head patchtst_hf --patchtst_checkpoint "$RAW_CKPT_LAST"

    echo ">>> PatchTST: encoder frozen + RAW MAE ep40"
    export RUN_SUFFIX="frompretrain_raw_ep40_encfrozen"
    export FREEZE_ENCODER=1
    for MODE in deep_only hybrid; do
        run_experiment "patchtst" "$MODE" --patchtst_checkpoint "$RAW_CKPT_LAST"
    done
    unset FREEZE_ENCODER
else
    echo ">>> SKIP raw_ep40 hf/encfrozen: checkpoint not found at ${RAW_CKPT_LAST}"
fi
unset RUN_SUFFIX
