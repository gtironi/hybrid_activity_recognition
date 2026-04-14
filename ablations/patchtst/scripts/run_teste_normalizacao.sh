#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

SUITE="normalizacao"
SUITE_LOG="${LOG_ROOT}/${SUITE}.log"
: > "${SUITE_LOG}"

VARIANT_DIR="${TMP_ROOT}/normalizacao_variants"
NORMALIZATION_VARIANTS="${NORMALIZATION_VARIANTS:-global_pipeline,per_window_zscore}"

python "${ABL_ROOT}/preprocess/generate_normalization_variants.py" \
    --train "${TRAIN_PARQUET}" \
    --test "${TEST_PARQUET}" \
    --out_dir "${VARIANT_DIR}" \
    2>&1 | tee -a "${SUITE_LOG}"

IFS=',' read -r -a VARIANTS <<< "${NORMALIZATION_VARIANTS}"
for VARIANT in "${VARIANTS[@]}"; do
    VARIANT="$(echo "${VARIANT}" | xargs)"
    VTRAIN="${VARIANT_DIR}/${VARIANT}/train.parquet"
    VTEST="${VARIANT_DIR}/${VARIANT}/test.parquet"

    if [ ! -f "${VTRAIN}" ] || [ ! -f "${VTEST}" ]; then
        echo "[$(ts)] ${SUITE}: skipping unknown variant '${VARIANT}'" | tee -a "${SUITE_LOG}"
        continue
    fi

    export TRAIN_PARQUET="${VTRAIN}"
    export TEST_PARQUET="${VTEST}"
    export PRETRAIN_PARQUET="${VTRAIN}"

    TAG="${SUITE}_${VARIANT}_d128_l3_h4_p8_s8"
    echo "[$(ts)] ${SUITE}: ${VARIANT}" | tee -a "${SUITE_LOG}"

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
