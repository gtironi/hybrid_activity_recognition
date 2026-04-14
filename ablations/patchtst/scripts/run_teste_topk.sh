#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

SUITE="topk"
SUITE_LOG="${LOG_ROOT}/${SUITE}.log"
: > "${SUITE_LOG}"

K_VALUES="${K_VALUES:-5,7,10,13,15,18}"
VARIANT_DIR="${TMP_ROOT}/topk_variants"

python "${ABL_ROOT}/preprocess/generate_topk_variants.py" \
    --train "${TRAIN_PARQUET}" \
    --test "${TEST_PARQUET}" \
    --out_dir "${VARIANT_DIR}" \
    --k_values "${K_VALUES}" \
    2>&1 | tee -a "${SUITE_LOG}"

for D in "${VARIANT_DIR}"/k_*; do
    KNAME="$(basename "${D}")"
    VTRAIN="${D}/train.parquet"
    VTEST="${D}/test.parquet"

    export TRAIN_PARQUET="${VTRAIN}"
    export TEST_PARQUET="${VTEST}"
    export PRETRAIN_PARQUET="${VTRAIN}"

    TAG="${SUITE}_${KNAME}_d128_l3_h4_p8_s8"
    echo "[$(ts)] ${SUITE}: ${KNAME}" | tee -a "${SUITE_LOG}"

    CKPT_PATH=$(run_pretrain_patchtst_variant \
        "${TAG}" \
        --patchtst_d_model 128 \
        --patchtst_num_layers 3 \
        --patchtst_num_heads 4 \
        --patchtst_patch_length 8 \
        --patchtst_patch_stride 8 \
        --patchtst_dropout 0.1 \
        2>&1 | tee -a "${SUITE_LOG}" | tail -n 1)

    run_supervised_patchtst_variant \
        hybrid \
        "${TAG}" \
        "${CKPT_PATH}" \
        --patchtst_d_model 128 \
        --patchtst_num_layers 3 \
        --patchtst_num_heads 4 \
        --patchtst_patch_length 8 \
        --patchtst_patch_stride 8 \
        --patchtst_dropout 0.1 \
        2>&1 | tee -a "${SUITE_LOG}"
done

echo "[$(ts)] ${SUITE}: done" | tee -a "${SUITE_LOG}"
