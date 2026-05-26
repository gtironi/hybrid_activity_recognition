#!/usr/bin/env bash
# Smoke test: runs every supervised experiment for 1 epoch to verify the
# training pipeline connects end-to-end.
#
# Out of scope:
#   - Data processing / raw-CSV windowing: tested separately.
#   - Pretraining from scratch: this script REUSES whatever pretrain artifacts
#     already exist in experiments/pretrain/ (real, full-length pretrains).
#     If they're missing, the per-encoder checkpoint guards would try to
#     pretrain and fail because raw windowing is disabled here on purpose.
#
# Smoke artifacts go to experiments/smoke/runs/<name>/ so they never mix with
# real run outputs.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${DIR}/_common.sh"

# Quick supervised settings (only supervised — keep pretrain defaults so the
# checkpoint guards find the real pretrain artifacts in experiments/pretrain/).
export EPOCHS=1
export FINETUNE_EPOCHS=1
export BATCH_SIZE=16
export VAL_FRACTION=0.9   # 10% train, 90% val — keeps each step fast.

# Smoke run dir (isolated from real runs); shared PRETRAIN_BASE so existing
# pretrain artifacts are reused.
RUN_NAME="${RUN_NAME:-smoke_$(date +%Y-%m-%d_%H-%M-%S)}"
RUN_DIR="${REPO_ROOT}/experiments/smoke/runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
export EXPERIMENTS_BASE="${RUN_DIR}"

# Sanity check: existing pretrain artifacts must be in place. We do NOT trigger
# the windowing or pretrain pipelines from this script — those are tested separately.
TS2VEC_CNN_DIR=$(ts2vec_pretrain_raw_dir cnn_lstm)
TS2VEC_ROB_DIR=$(ts2vec_pretrain_raw_dir robust)
PATCHTST_DIR=$(patchtst_mae_raw_dir)
missing=()
[ -f "${TS2VEC_CNN_DIR}/ts2vec_ep100.pt" ] || missing+=("${TS2VEC_CNN_DIR}/ts2vec_ep100.pt")
[ -f "${TS2VEC_ROB_DIR}/ts2vec_ep100.pt" ] || missing+=("${TS2VEC_ROB_DIR}/ts2vec_ep100.pt")
[ -f "${PATCHTST_DIR}/pretrain_ep40.pt" ]  || missing+=("${PATCHTST_DIR}/pretrain_ep40.pt")
if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: smoke test requires existing pretrain artifacts. Missing:" >&2
    printf '  - %s\n' "${missing[@]}" >&2
    echo "Run the real pretraining first (pretrain_encoder.sh, pretrain_patchtst.sh)." >&2
    exit 1
fi

cat > "${RUN_DIR}/manifest.json" <<EOF
{
  "run_name": "${RUN_NAME}",
  "kind": "smoke_test",
  "start_time": "$(date -Iseconds)",
  "epochs": ${EPOCHS},
  "finetune_epochs": ${FINETUNE_EPOCHS},
  "batch_size": ${BATCH_SIZE},
  "val_fraction": ${VAL_FRACTION},
  "pretrain_base": "${PRETRAIN_BASE}"
}
EOF

echo "=== Smoke test (1 epoch, batch_size=16) ==="
echo "Run dir:  ${RUN_DIR}"
echo "Pretrain: ${PRETRAIN_BASE} (reused, not regenerated)"
echo "Start:    $(date)"
echo ""

# Per-encoder full pipelines (pretrain step is a no-op since checkpoints exist)
bash "${DIR}/run_cnn_lstm.sh"
bash "${DIR}/run_robust.sh"
bash "${DIR}/run_patchtst.sh"

# PatchTST ablations
bash "${DIR}/run_patchtst_frozen.sh"
bash "${DIR}/run_patchtst_hf.sh"

# Pretrain checkpoint ablations (one folder per encoder)
ENCODER=cnn_lstm bash "${DIR}/run_pretrain_ablation.sh"
ENCODER=robust   bash "${DIR}/run_pretrain_ablation.sh"
ENCODER=patchtst bash "${DIR}/run_pretrain_ablation.sh"

# TSFEL baselines
bash "${DIR}/run_tsfel_baseline.sh"
bash "${DIR}/run_tsfel_mlp.sh"

echo ""
echo "=== Smoke test complete: $(date) ==="
echo "Outputs in: ${RUN_DIR}"
