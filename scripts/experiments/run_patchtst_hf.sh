#!/usr/bin/env bash
# PatchTST HuggingFace classification-head comparison.
#
# Note: --head patchtst_hf is constrained to --input_mode deep_only by main.py,
# so this script only runs deep_only variants. The MLP-head counterparts come
# from run_patchtst.sh and can be compared directly.
#
# Checkpoint guard: if the MAE raw checkpoint does not exist yet, this script
# triggers pretrain_patchtst.sh to create it.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${DIR}/_common.sh"
export BATCH_SIZE="${PATCHTST_BATCH_SIZE:-128}"

ENCODER="patchtst"
EP="${EP:-40}"
export EXPERIMENTS_BASE="${EXPERIMENTS_BASE}/patchtst_hf"
mkdir -p "${EXPERIMENTS_BASE}"

# --- Checkpoint guard ---
RAW_PRETRAIN_DIR=$(patchtst_mae_raw_dir)
if [ ! -f "${RAW_PRETRAIN_DIR}/DONE" ]; then
    echo ">>> ${ENCODER}: raw MAE checkpoint missing — running pretrain_patchtst.sh"
    bash "${DIR}/pretrain_patchtst.sh"
else
    echo ">>> ${ENCODER}: raw MAE checkpoint found at ${RAW_PRETRAIN_DIR}, skipping pretrain"
fi

RAW_CKPT="${RAW_PRETRAIN_DIR}/pretrain_ep${EP}.pt"

unset FREEZE_ENCODER || true

# From scratch with HF head
RUN_SUFFIX=fromscratch_hf run_experiment "$ENCODER" "deep_only" --head patchtst_hf

# From raw-data MAE checkpoint with HF head
if [ -f "$RAW_CKPT" ]; then
    RUN_SUFFIX="frompretrain_raw_ep${EP}_hf" run_experiment "$ENCODER" "deep_only" \
        --head patchtst_hf --patchtst_checkpoint "$RAW_CKPT"
else
    echo ">>> SKIP frompretrain_raw_ep${EP}_hf: checkpoint not found at ${RAW_CKPT}"
fi
