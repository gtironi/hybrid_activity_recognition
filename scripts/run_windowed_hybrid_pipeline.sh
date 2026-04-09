#!/usr/bin/env bash
# Encadeia: prepare (discover no train) → prepare (apply no test) → hybrid_activity_recognition.main
# Assume train.parquet / test.parquet já existem (ex.: scripts/dataset_processing.py).
#
# Repasse de argumentos: dois separadores -- isolados na ordem:
#   [args prepare train] -- [args prepare test] -- [args main]
#
# Exemplo:
#   ./scripts/run_windowed_hybrid_pipeline.sh \
#     --batch-size 3000 \
#     -- \
#     --batch-size 5000 \
#     -- \
#     --mode supervised --model robust_hybrid --epochs 10
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_IN_TRAIN="${REPO_ROOT}/dataset/processed/AcTBeCalf/train.parquet"
DEFAULT_IN_TEST="${REPO_ROOT}/dataset/processed/AcTBeCalf/test.parquet"
DEFAULT_OUT_TRAIN="${REPO_ROOT}/dataset/processed/AcTBeCalf/windowed_train.parquet"
DEFAULT_OUT_TEST="${REPO_ROOT}/dataset/processed/AcTBeCalf/windowed_test.parquet"
DEFAULT_MANIFEST="${REPO_ROOT}/dataset/processed/AcTBeCalf/tsfel_feature_manifest.json"

PREPARE_TRAIN_ARGS=()
PREPARE_TEST_ARGS=()
MAIN_ARGS=()
phase=0
for arg in "$@"; do
  if [[ "$arg" == "--" ]]; then
    phase=$((phase + 1))
    continue
  fi
  case $phase in
    0) PREPARE_TRAIN_ARGS+=("$arg") ;;
    1) PREPARE_TEST_ARGS+=("$arg") ;;
    *) MAIN_ARGS+=("$arg") ;;
  esac
done

python "${REPO_ROOT}/scripts/prepare_windowed_parquet.py" \
  --input "${DEFAULT_IN_TRAIN}" \
  --output "${DEFAULT_OUT_TRAIN}" \
  --feature-manifest-out "${DEFAULT_MANIFEST}" \
  "${PREPARE_TRAIN_ARGS[@]}"

python "${REPO_ROOT}/scripts/prepare_windowed_parquet.py" \
  --input "${DEFAULT_IN_TEST}" \
  --output "${DEFAULT_OUT_TEST}" \
  --feature-manifest-in "${DEFAULT_MANIFEST}" \
  "${PREPARE_TEST_ARGS[@]}"

export PYTHONPATH="${REPO_ROOT}/src"
python -m hybrid_activity_recognition.main \
  --labeled_parquet_train "${DEFAULT_OUT_TRAIN}" \
  --labeled_parquet_test "${DEFAULT_OUT_TEST}" \
  "${MAIN_ARGS[@]}"
