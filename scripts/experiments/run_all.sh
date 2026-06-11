#!/usr/bin/env bash
# Master script: runs the full experiment matrix.
#
# Layout produced by a single invocation:
#   experiments/pretrain/                 ← shared across all runs (PRETRAIN_BASE)
#   experiments/runs/<RUN_NAME>/          ← this invocation's outputs
#     ├── manifest.json
#     ├── cnn_lstm/                       (last pretrain snapshot only)
#     ├── patchtst/                       (last pretrain snapshot only)
#     ├── patchtst_frozen/
#     ├── pretrain_ablation_cnn_lstm/     (all snapshots)
#     ├── pretrain_ablation_patchtst/     (all snapshots)
#     └── tsfel_baseline/
#
# Override the run name with RUN_NAME=my_label; defaults to a timestamp.
#
# For remote execution via SSH:
#   nohup bash scripts/experiments/run_all.sh > logs/run_all.log 2>&1 &
#   # or: screen -dmS exp bash scripts/experiments/run_all.sh
#   # or: tmux new -d -s exp 'bash scripts/experiments/run_all.sh'
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${DIR}/_common.sh"

# --- Per-invocation run directory ---
RUN_NAME="${RUN_NAME:-run_$(date +%Y-%m-%d_%H-%M-%S)}"
RUN_DIR="${REPO_ROOT}/experiments/runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
export EXPERIMENTS_BASE="${RUN_DIR}"

# Sub-experiments included in this run.
SUB_EXPERIMENTS=(
    cnn_lstm
    patchtst
    patchtst_frozen
    pretrain_ablation_cnn_lstm
    pretrain_ablation_patchtst
    tsfel_baseline
)

# --- Manifest ---
START_ISO="$(date -Iseconds)"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
MANIFEST="${RUN_DIR}/manifest.json"
{
    printf '{\n'
    printf '  "run_name": "%s",\n' "${RUN_NAME}"
    printf '  "start_time": "%s",\n' "${START_ISO}"
    printf '  "git_commit": "%s",\n' "${GIT_COMMIT}"
    printf '  "git_branch": "%s",\n' "${GIT_BRANCH}"
    printf '  "dataset_id": "%s",\n' "${DATASET_ID}"
    printf '  "pretrain_base": "%s",\n' "${PRETRAIN_BASE}"
    printf '  "experiments_base": "%s",\n' "${EXPERIMENTS_BASE}"
    printf '  "hyperparameters": {\n'
    printf '    "seed": %s,\n' "${SEED}"
    printf '    "device": "%s",\n' "${DEVICE}"
    printf '    "epochs": %s,\n' "${EPOCHS}"
    printf '    "pretrain_epochs": %s,\n' "${PRETRAIN_EPOCHS}"
    printf '    "patchtst_pretrain_epochs": %s,\n' "${PATCHTST_PRETRAIN_EPOCHS}"
    printf '    "batch_size_large": %s,\n' "${BATCH_SIZE_LARGE}"
    printf '    "patchtst_batch_size": %s,\n' "${PATCHTST_BATCH_SIZE}"
    printf '    "patchtst_pretrain_batch_size": %s,\n' "${PATCHTST_PRETRAIN_BATCH_SIZE}"
    printf '    "lr": "%s",\n' "${LR}"
    printf '    "pretrain_lr": "%s",\n' "${PRETRAIN_LR}"
    printf '    "val_fraction": %s\n' "${VAL_FRACTION}"
    printf '  },\n'
    printf '  "sub_experiments": [\n'
    for i in "${!SUB_EXPERIMENTS[@]}"; do
        sep=","
        [ "$i" -eq $(( ${#SUB_EXPERIMENTS[@]} - 1 )) ] && sep=""
        printf '    "%s"%s\n' "${SUB_EXPERIMENTS[$i]}" "$sep"
    done
    printf '  ]\n'
    printf '}\n'
} > "${MANIFEST}"

echo "=== Full experiment pipeline ==="
echo "Run name:  ${RUN_NAME}"
echo "Run dir:   ${RUN_DIR}"
echo "Manifest:  ${MANIFEST}"
echo "Pretrain:  ${PRETRAIN_BASE}"
echo "Start:     ${START_ISO}"
echo ""

# --- 1) Pretrain encoders on raw data (idempotent; lives in PRETRAIN_BASE) ---
echo "--- Pretrain (TS2Vec + MAE on raw data) ---"
ENCODER=cnn_lstm bash "${DIR}/pretrain_encoder.sh"
bash "${DIR}/pretrain_patchtst.sh"

# --- 2) Per-encoder experiments ---
echo ""
echo "--- Per-encoder experiments ---"
bash "${DIR}/run_cnn_lstm.sh"
bash "${DIR}/run_patchtst.sh"

# --- 3) PatchTST ablations ---
echo ""
echo "--- PatchTST ablations ---"
bash "${DIR}/run_patchtst_frozen.sh"

# --- 4) Pretrain checkpoint ablations (one folder per encoder) ---
echo ""
echo "--- Pretrain checkpoint ablations ---"
ENCODER=cnn_lstm bash "${DIR}/run_pretrain_ablation.sh"
ENCODER=patchtst bash "${DIR}/run_pretrain_ablation.sh"

# --- 5) TSFEL baseline ---
echo ""
echo "--- TSFEL baseline ---"
bash "${DIR}/run_tsfel_baseline.sh"

echo ""
echo "=== All experiments complete: $(date -Iseconds) ==="
echo "Outputs in: ${RUN_DIR}"
