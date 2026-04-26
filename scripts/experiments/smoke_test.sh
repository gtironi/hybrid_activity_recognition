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

# PatchTST with HF classification head (deep_only)
OUT_HF="${REPO_ROOT}/experiments/patchtst_deep_only_hfhead_${DATASET_ID}_ep${EPOCHS}_bs${BATCH_SIZE}_lr${LR}_s${SEED}"
if [ -f "${OUT_HF}/DONE" ]; then
	echo ">>> patchtst_deep_only_hfhead: already complete, skipping"
else
	mkdir -p "${OUT_HF}"
	PRETRAIN_DIR=$(make_run_dir "patchtst" "pretrain")
	PATCHTST_HF_ARGS=(
		--mode supervised
		--model patchtst
		--input_mode deep_only
		--head patchtst_hf
		--labeled_parquet_train "$TRAIN_PARQUET"
		--labeled_parquet_test "$TEST_PARQUET"
		--output_dir "$OUT_HF"
		--epochs "$EPOCHS"
		--batch_size "$BATCH_SIZE"
		--lr "$LR"
		--seed "$SEED"
		--device "$DEVICE"
	)
	if [ -f "${PRETRAIN_DIR}/best.pt" ]; then
		PATCHTST_HF_ARGS+=(--patchtst_checkpoint "${PRETRAIN_DIR}/best.pt")
	fi
	python -m hybrid_activity_recognition.main \
		"${PATCHTST_HF_ARGS[@]}" \
		2>&1 | tee -a "${OUT_HF}/train.log"
	touch "${OUT_HF}/DONE"
	echo ">>> patchtst_deep_only_hfhead done at $(date)"
fi

# TSFEL baseline
bash "${DIR}/run_tsfel_baseline.sh"

# TSFEL + MLP head baseline
bash "${DIR}/run_tsfel_mlp.sh"

echo ""
echo "=== Smoke test complete: $(date) ==="
echo "Check experiments/ for DONE markers."
