#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

SUITE="sem_pretreino"
SUITE_LOG="${LOG_ROOT}/${SUITE}.log"
: > "${SUITE_LOG}"

SEEDS="${SEEDS:-42 123 456}"

for seed in ${SEEDS}; do
    export SEED="${seed}"
    TAG="nopre_d128_l3_h4_p8_s8"
    echo "[$(ts)] ${SUITE}: seed=${SEED}" | tee -a "${SUITE_LOG}"

    run_supervised_patchtst_variant \
        deep_only \
        "${TAG}" \
        NONE \
        --patchtst_d_model 128 \
        --patchtst_num_layers 3 \
        --patchtst_num_heads 4 \
        --patchtst_patch_length 8 \
        --patchtst_patch_stride 8 \
        --patchtst_dropout 0.1 \
        2>&1 | tee -a "${SUITE_LOG}"
done

echo "[$(ts)] ${SUITE}: done" | tee -a "${SUITE_LOG}"
