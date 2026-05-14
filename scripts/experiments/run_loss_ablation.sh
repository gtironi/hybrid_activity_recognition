#!/usr/bin/env bash
# Loss/sampler ablation: 4 combinations (CE, CE+sampler, Focal, Focal+sampler)
# applied to a fixed set of models. Uses RUN_SUFFIX to keep output dirs distinct
# from the existing baselines.
#
# Usage:
#   bash scripts/experiments/run_loss_ablation.sh
#   # restrict models:
#   MODELS="cnn_lstm:hybrid" bash scripts/experiments/run_loss_ablation.sh
#   # background:
#   nohup bash scripts/experiments/run_loss_ablation.sh > logs/loss_ablation.log 2>&1 &
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# model:mode pairs to ablate. Format "model1:mode1 model2:mode2 ...".
MODELS="${MODELS:-cnn_lstm:hybrid patchtst:hybrid robust:hybrid tsfel_mlp:tsfel_only}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"

run_combo() {
    # Args: model mode loss_type use_sampler
    local MODEL="$1" MODE="$2" LOSS="$3" SAMPLER="$4"

    local suffix="loss${LOSS}"
    [ "$SAMPLER" = "1" ] && suffix="${suffix}_balsamp"
    export RUN_SUFFIX="${suffix}"

    # Pick batch size: PatchTST needs smaller batches.
    if [ "$MODEL" = "patchtst" ]; then
        export BATCH_SIZE="${PATCHTST_BATCH_SIZE:-128}"
    else
        export BATCH_SIZE="${BATCH_SIZE_LARGE:-512}"
    fi

    local extra_args=(--loss_type "$LOSS" --focal_gamma "$FOCAL_GAMMA")
    [ "$SAMPLER" = "1" ] && extra_args+=(--balanced_sampler)

    echo "=== ${MODEL}/${MODE} | loss=${LOSS} sampler=${SAMPLER} (suffix=${suffix}) ==="
    run_experiment "$MODEL" "$MODE" "${extra_args[@]}"
    unset RUN_SUFFIX
}

echo "=== Loss/sampler ablation start: $(date) ==="

for pair in $MODELS; do
    MODEL="${pair%%:*}"
    MODE="${pair##*:}"

    # 4 combos: (ce, no-sampler) is the existing baseline — skip if you already
    # have it in experiments/. Comment out the line below to re-run for parity.
    run_combo "$MODEL" "$MODE" "ce"    "0"
    run_combo "$MODEL" "$MODE" "ce"    "1"
    run_combo "$MODEL" "$MODE" "focal" "0"
    run_combo "$MODEL" "$MODE" "focal" "1"
done

echo "=== Loss/sampler ablation done: $(date) ==="
