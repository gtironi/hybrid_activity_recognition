#!/usr/bin/env bash
# CNN+LSTM encoder: deep_only and hybrid experiments.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

for MODE in deep_only hybrid; do
    run_experiment "cnn_lstm" "$MODE"
done
