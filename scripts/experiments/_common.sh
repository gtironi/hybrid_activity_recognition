#!/usr/bin/env bash
# Shared variables and helpers for all experiment scripts.
# Source this file; do not execute it directly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src"

# --- Default paths (override from environment if needed) ---
TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/dataset/processed/AcTBeCalf/windowed_train.parquet}"
TEST_PARQUET="${TEST_PARQUET:-${REPO_ROOT}/dataset/processed/AcTBeCalf/windowed_test.parquet}"
PRETRAIN_PARQUET="${PRETRAIN_PARQUET:-${TRAIN_PARQUET}}"
DATASET_ID="${DATASET_ID:-AcTBeCalf}"

# Raw (unlabeled) data used for pretraining encoders.
RAW_CSV="${RAW_CSV:-${REPO_ROOT}/dataset/Time_Adj_Raw_Data.csv}"
RAW_PARQUET="${RAW_PARQUET:-${REPO_ROOT}/dataset/processed/pretrain_raw_windowed.parquet}"
WINDOW_LEN="${WINDOW_LEN:-75}"
WINDOW_STRIDE="${WINDOW_STRIDE:-37}"
# Base dir for supervised run dirs. run_all.sh overrides this with a
# timestamped per-run folder; per-encoder scripts append their own sub-dir.
EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-${REPO_ROOT}/experiments}"
# Pretraining checkpoints live in a fixed shared location so they are reused
# across every run regardless of which run_all invocation produced them.
PRETRAIN_BASE="${PRETRAIN_BASE:-${REPO_ROOT}/experiments/pretrain}"

# --- Default hyperparameters ---
SEED="${SEED:-2026}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-200}"          # Stage 1: balanced CE, ES patience=25 will stop early
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"  # Stage 2: plain CE, low LR, no early stop
VAL_FRACTION="${VAL_FRACTION:-0.1}"
LR="${LR:-1e-3}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-40}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
# Larger batches for CNN/LSTM/robust/TSFEL-MLP; PatchTST scripts set PATCHTST_BATCH_SIZE.
BATCH_SIZE_LARGE="${BATCH_SIZE_LARGE:-512}"
PATCHTST_BATCH_SIZE="${PATCHTST_BATCH_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-${BATCH_SIZE_LARGE}}"

# PatchTST pretrain uses a much larger batch + fewer epochs than supervised
# training. Kept separate from PATCHTST_BATCH_SIZE / PRETRAIN_EPOCHS so we can
# tune pretraining independently.
PATCHTST_PRETRAIN_EPOCHS="${PATCHTST_PRETRAIN_EPOCHS:-40}"
PATCHTST_PRETRAIN_BATCH_SIZE="${PATCHTST_PRETRAIN_BATCH_SIZE:-1536}"

# --- Helpers ---

make_run_dir() {
    # Usage: make_run_dir MODEL MODE
    # Hyperparameters (epochs, batch size, lr) live in the parent run folder
    # manifest.json — keep individual experiment names short and stable.
    # Optional RUN_SUFFIX (e.g. fromscratch, frompretrain_raw_ep40) disambiguates
    # variants within the same encoder × mode.
    local MODEL="$1" MODE="$2"
    local suf="${RUN_SUFFIX:-}"
    if [ -n "$suf" ]; then
        suf="_${suf}"
    fi
    echo "${EXPERIMENTS_BASE}/${MODEL}_${MODE}${suf}_${DATASET_ID}_s${SEED}"
}

run_experiment() {
    # Usage: run_experiment MODEL MODE [-- extra CLI args ...]
    # Remaining words after MODEL MODE are passed verbatim to main.py (quoted-safe for paths with spaces).
    local MODEL="$1" MODE="$2"
    shift 2
    local OUT
    OUT=$(make_run_dir "$MODEL" "$MODE")

    if [ -f "${OUT}/DONE" ]; then
        echo ">>> ${MODEL}_${MODE}: already complete, skipping"
        return 0
    fi
    mkdir -p "${OUT}"

    echo ">>> Starting ${MODEL}_${MODE} at $(date)"
    local FREEZE_ARGS=()
    if [ "${FREEZE_ENCODER:-0}" = "1" ]; then
        FREEZE_ARGS+=(--freeze_encoder)
    fi
    if [ -f "${OUT}/checkpoint.pt" ]; then
        echo ">>> Checkpoint found at ${OUT}/checkpoint.pt, resuming..."
        python -m hybrid_activity_recognition.main \
            --mode supervised \
            --model "$MODEL" \
            --input_mode "$MODE" \
            --labeled_parquet_train "$TRAIN_PARQUET" \
            --labeled_parquet_test "$TEST_PARQUET" \
            --output_dir "$OUT" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --seed "$SEED" \
            --device "$DEVICE" \
            --val_fraction "$VAL_FRACTION" \
            --checkpoint "${OUT}/checkpoint.pt" \
            "${FREEZE_ARGS[@]}" \
            "$@" \
            2>&1 | tee -a "${OUT}/train.log"
    else
        python -m hybrid_activity_recognition.main \
            --mode supervised \
            --model "$MODEL" \
            --input_mode "$MODE" \
            --labeled_parquet_train "$TRAIN_PARQUET" \
            --labeled_parquet_test "$TEST_PARQUET" \
            --output_dir "$OUT" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --seed "$SEED" \
            --device "$DEVICE" \
            --val_fraction "$VAL_FRACTION" \
            "${FREEZE_ARGS[@]}" \
            "$@" \
            2>&1 | tee -a "${OUT}/train.log"
    fi

    touch "${OUT}/DONE"
    echo ">>> ${MODEL}_${MODE} done at $(date)"
}

ensure_raw_parquet() {
    # Window the raw CSV into a parquet if it doesn't exist yet.
    if [ ! -f "$RAW_PARQUET" ]; then
        echo ">>> Windowing $RAW_CSV → $RAW_PARQUET" >&2
        python "${REPO_ROOT}/scripts/window_raw_for_pretrain.py" \
            --input  "$RAW_CSV" \
            --output "$RAW_PARQUET" \
            --window_len "$WINDOW_LEN" \
            --stride "$WINDOW_STRIDE" >&2
    else
        echo ">>> $RAW_PARQUET already exists, skipping windowing" >&2
    fi
}

ts2vec_pretrain_raw_dir() {
    # Echo the directory where TS2Vec pretrain checkpoints live for a given
    # encoder. TS2Vec is always raw-data in the new pipeline; the "raw"
    # qualifier is omitted from the directory name.
    local MODEL="$1"
    echo "${PRETRAIN_BASE}/ts2vec_pretrain_${MODEL}_${DATASET_ID}_ep${PRETRAIN_EPOCHS}_s${SEED}"
}

patchtst_mae_raw_dir() {
    # Echo the directory where PatchTST raw MAE checkpoints live.
    # Uses PATCHTST_PRETRAIN_EPOCHS / PATCHTST_PRETRAIN_BATCH_SIZE so pretrain
    # hyperparameters are independent of supervised ones.
    echo "${PRETRAIN_BASE}/patchtst_pretrain_raw_${DATASET_ID}_ep${PATCHTST_PRETRAIN_EPOCHS}_bs${PATCHTST_PRETRAIN_BATCH_SIZE}_lr${PRETRAIN_LR}_s${SEED}"
}

run_ts2vec_pretrain_raw() {
    # Usage: run_ts2vec_pretrain_raw MODEL
    # TS2Vec pretraining on the unlabeled raw parquet. Idempotent: skipped if
    # ep20/ep50/ep100 snapshots already exist. Prints the output_dir to stdout.
    local MODEL="$1"
    local OUT
    OUT=$(ts2vec_pretrain_raw_dir "$MODEL")

    if [ -f "${OUT}/ts2vec_best.pt" ]; then
        echo ">>> TS2Vec raw pretrain ${MODEL}: best checkpoint found at ${OUT}, skipping" >&2
        echo "${OUT}"
        return 0
    fi
    ensure_raw_parquet
    mkdir -p "${OUT}"

    echo ">>> TS2Vec raw pretraining ${MODEL} at $(date)" >&2
    python -m hybrid_activity_recognition.main \
        --mode pretrain_ts2vec \
        --model "$MODEL" \
        --pretrain_parquet "$RAW_PARQUET" \
        --output_dir "$OUT" \
        --pretrain_epochs "$PRETRAIN_EPOCHS" \
        --batch_size "${BATCH_SIZE_LARGE}" \
        --pretrain_lr "$PRETRAIN_LR" \
        --seed "$SEED" \
        --device "$DEVICE" \
        2>&1 | tee -a "${OUT}/pretrain.log" >&2

    echo ">>> TS2Vec raw pretrain ${MODEL} done at $(date) → ${OUT}" >&2
    echo "${OUT}"
}

run_patchtst_mae_raw() {
    # PatchTST MAE pretraining on the unlabeled raw parquet. Idempotent: skipped
    # if a DONE marker is present. Prints the output_dir to stdout.
    local OUT
    OUT=$(patchtst_mae_raw_dir)

    if [ -f "${OUT}/DONE" ]; then
        echo ">>> PatchTST raw MAE pretrain: already complete in ${OUT}, skipping" >&2
        echo "${OUT}"
        return 0
    fi
    ensure_raw_parquet
    mkdir -p "${OUT}"

    echo ">>> PatchTST raw MAE pretraining at $(date)" >&2
    local RESUME_ARGS=()
    if [ -f "${OUT}/checkpoint.pt" ]; then
        echo ">>> Resuming from ${OUT}/checkpoint.pt" >&2
        RESUME_ARGS+=(--checkpoint "${OUT}/checkpoint.pt")
    fi

    python -m hybrid_activity_recognition.main \
        --mode pretrain \
        --pretrain_parquet "$RAW_PARQUET" \
        --output_dir "$OUT" \
        --pretrain_epochs "$PATCHTST_PRETRAIN_EPOCHS" \
        --pretrain_lr "$PRETRAIN_LR" \
        --batch_size "${PATCHTST_PRETRAIN_BATCH_SIZE}" \
        --seed "$SEED" \
        --device "$DEVICE" \
        "${RESUME_ARGS[@]}" \
        2>&1 | tee -a "${OUT}/train.log" >&2

    touch "${OUT}/DONE"
    echo ">>> PatchTST raw MAE pretrain done at $(date) → ${OUT}" >&2
    echo "${OUT}"
}

run_ts2vec_pretrain() {
    # Usage: run_ts2vec_pretrain MODEL
    # Runs TS2Vec pretraining once; skips if all snapshots already exist.
    # Prints the output_dir to stdout (caller resolves individual checkpoints).
    local MODEL="$1"
    local OUT="${EXPERIMENTS_BASE}/ts2vec_pretrain_${MODEL}_${DATASET_ID}_ep${PRETRAIN_EPOCHS}_s${SEED}"

    if [ -f "${OUT}/ts2vec_best.pt" ]; then
        echo ">>> TS2Vec pretrain ${MODEL}: best checkpoint found, skipping" >&2
        echo "${OUT}"
        return 0
    fi
    mkdir -p "${OUT}"

    echo ">>> TS2Vec pretraining ${MODEL} at $(date)" >&2
    python -m hybrid_activity_recognition.main \
        --mode pretrain_ts2vec \
        --model "$MODEL" \
        --pretrain_parquet "$PRETRAIN_PARQUET" \
        --output_dir "$OUT" \
        --pretrain_epochs "$PRETRAIN_EPOCHS" \
        --batch_size "${BATCH_SIZE_LARGE}" \
        --pretrain_lr "$PRETRAIN_LR" \
        --seed "$SEED" \
        --device "$DEVICE" \
        2>&1 | tee -a "${OUT}/pretrain.log" >&2

    echo ">>> TS2Vec pretrain ${MODEL} done at $(date)" >&2
    echo "${OUT}"
}

run_finetune() {
    # Stage 2: load best.pt from Stage 1, fine-tune with plain CE (no class weights).
    # Usage: run_finetune MODEL MODE [-- extra CLI args ...]
    local MODEL="$1" MODE="$2"
    shift 2
    local OUT
    OUT=$(make_run_dir "$MODEL" "$MODE")
    local STAGE1_CKPT="${OUT}/best.pt"
    local FINETUNE_DONE="${OUT}/DONE_finetune"

    if [ -f "${FINETUNE_DONE}" ]; then
        echo ">>> ${MODEL}_${MODE} finetune: already complete, skipping"
        return 0
    fi
    if [ ! -f "${STAGE1_CKPT}" ]; then
        echo ">>> ${MODEL}_${MODE} finetune: Stage 1 checkpoint not found at ${STAGE1_CKPT}, skipping"
        return 1
    fi

    echo ">>> Fine-tuning ${MODEL}_${MODE} at $(date)"
    python -m hybrid_activity_recognition.main \
        --mode finetune \
        --model "$MODEL" \
        --input_mode "$MODE" \
        --labeled_parquet_train "$TRAIN_PARQUET" \
        --labeled_parquet_test "$TEST_PARQUET" \
        --output_dir "$OUT" \
        --epochs "$FINETUNE_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr 1e-4 \
        --seed "$SEED" \
        --device "$DEVICE" \
        --val_fraction "$VAL_FRACTION" \
        --checkpoint "${STAGE1_CKPT}" \
        "$@" \
        2>&1 | tee -a "${OUT}/finetune.log"

    touch "${FINETUNE_DONE}"
    echo ">>> ${MODEL}_${MODE} finetune done at $(date)"
}
