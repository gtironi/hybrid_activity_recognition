#!/usr/bin/env bash
# Run rf_hypersearch on EVERY windowed train/test parquet pair found under the
# search roots — paper features (11-class), 6-class (3s/5s), and every kfold fold.
#
# Discovery: any "*windowed*train*.parquet" whose sibling "...test..." exists.
# Each run is isolated; a failure logs a warning and the loop continues.
#
# Usage:
#   VAL_FRACTION=0.2 bash scripts/experiments/run_rf_hypersearch_all.sh
# Override:
#   ROOTS="dataset/processed/AcTBeCalf dataset/processed/kfold" \
#   SEED=2026 OUT_BASE=experiments/rf_hypersearch_all/myrun \
#   bash scripts/experiments/run_rf_hypersearch_all.sh

set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DIR}/../.." && pwd)"
cd "$REPO_ROOT"

VAL_FRACTION="${VAL_FRACTION:-0.2}"
SEED="${SEED:-2026}"
SELECT_METRIC="${SELECT_METRIC:-balanced_accuracy}"
ROOTS="${ROOTS:-dataset/processed}"
TS="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_BASE="${OUT_BASE:-experiments/rf_hypersearch_all/run_${TS}}"
mkdir -p "$OUT_BASE"

echo ">>> rf_hypersearch ALL  | val_fraction=${VAL_FRACTION} seed=${SEED} select=${SELECT_METRIC}"
echo ">>> roots: ${ROOTS}"
echo ">>> out:   ${OUT_BASE}"

# Collect candidate train parquets. A bad/stale parquet just fails its own run
# (logged, non-fatal) — the loop continues.
mapfile -t TRAINS < <(
  for r in $ROOTS; do
    find "$r" -type f -name '*windowed*train*.parquet' 2>/dev/null
  done | sort -u
)

n_ok=0; n_skip=0; n_fail=0
for TR in "${TRAINS[@]}"; do
    TE="${TR/train.parquet/test.parquet}"
    if [ ! -f "$TE" ]; then
        echo "--- skip (no test sibling): $TR"; n_skip=$((n_skip+1)); continue
    fi
    # label = path relative to repo, slashes -> __, drop extension
    REL="${TR#"$REPO_ROOT"/}"; REL="${REL#./}"
    LABEL="$(echo "${REL%.parquet}" | sed 's#/#__#g')"
    OUT="${OUT_BASE}/${LABEL}"
    if [ -f "${OUT}/DONE" ]; then
        echo "--- already done: ${LABEL}"; n_ok=$((n_ok+1)); continue
    fi
    mkdir -p "$OUT"
    echo ""
    echo ">>> [$((n_ok+n_fail+1))] ${LABEL}"
    if PYTHONPATH=src python -m random_forest_baseline.rf_hypersearch \
        --train "$TR" --test "$TE" \
        --val_fraction "$VAL_FRACTION" --seed "$SEED" \
        --select_metric "$SELECT_METRIC" \
        --output_dir "$OUT" 2>&1 | tee "${OUT}/run.log"; then
        touch "${OUT}/DONE"; n_ok=$((n_ok+1))
    else
        echo "!!! FAILED: ${LABEL} (see ${OUT}/run.log)"; n_fail=$((n_fail+1))
    fi
done

echo ""
echo ">>> done. ok=${n_ok} skipped=${n_skip} failed=${n_fail}"

# Aggregate every comparison.json into one master table.
python "${REPO_ROOT}/scripts/aggregate_rf_hypersearch.py" --base "$OUT_BASE" || true
echo ">>> summary: ${OUT_BASE}/SUMMARY.md"
