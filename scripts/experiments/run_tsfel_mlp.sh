#!/usr/bin/env bash
# TSFEL-only deep baseline: TSFEL branch + MLP head.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

run_experiment "tsfel_mlp" "tsfel_only"
