#!/usr/bin/env bash
# Robust CNN+LSTM encoder: deep_only and hybrid experiments.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
export BATCH_SIZE="${BATCH_SIZE_LARGE:-512}"

for MODE in deep_only hybrid; do
    run_experiment "robust" "$MODE"
done
