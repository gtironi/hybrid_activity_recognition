#!/usr/bin/env bash
# TSFEL-only deep baseline: TSFEL branch + MLP head.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
export BATCH_SIZE="${BATCH_SIZE_LARGE:-512}"

run_experiment "tsfel_mlp" "tsfel_only"
