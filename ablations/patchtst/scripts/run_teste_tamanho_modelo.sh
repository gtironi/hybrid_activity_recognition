#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

SUITE="tamanho_modelo"
SUITE_LOG="${LOG_ROOT}/${SUITE}.log"
: > "${SUITE_LOG}"

# Orçamento rapido: 12 configuracoes arquiteturais.
CONFIGS=(
  "64 2 2 4 4"
  "64 2 4 4 4"
  "64 3 2 8 8"
  "64 3 4 8 8"
  "128 2 2 4 4"
  "128 2 4 8 8"
  "128 3 2 8 8"
  "128 3 4 4 4"
  "256 2 2 8 8"
  "256 2 4 4 4"
  "256 3 2 4 4"
  "256 3 4 8 8"
)

MAX_CONFIGS="${MAX_CONFIGS:-0}"
processed=0

for cfg in "${CONFIGS[@]}"; do
    if [ "${MAX_CONFIGS}" -gt 0 ] && [ "${processed}" -ge "${MAX_CONFIGS}" ]; then
        echo "[$(ts)] ${SUITE}: reached MAX_CONFIGS=${MAX_CONFIGS}, stopping early" | tee -a "${SUITE_LOG}"
        break
    fi

    read -r D_MODEL N_LAYERS N_HEADS PATCH_LEN PATCH_STRIDE <<< "${cfg}"
    TAG="d${D_MODEL}_l${N_LAYERS}_h${N_HEADS}_p${PATCH_LEN}_s${PATCH_STRIDE}"

    echo "[$(ts)] ${SUITE}: ${TAG}" | tee -a "${SUITE_LOG}"

    CKPT_PATH=$(run_pretrain_patchtst_variant \
        "${SUITE}_${TAG}" \
        --patchtst_d_model "${D_MODEL}" \
        --patchtst_num_layers "${N_LAYERS}" \
        --patchtst_num_heads "${N_HEADS}" \
        --patchtst_patch_length "${PATCH_LEN}" \
        --patchtst_patch_stride "${PATCH_STRIDE}" \
        --patchtst_dropout 0.1 \
        2>&1 | tee -a "${SUITE_LOG}" | tail -n 1)

    run_supervised_patchtst_variant \
        hybrid \
        "${SUITE}_${TAG}" \
        "${CKPT_PATH}" \
        --patchtst_d_model "${D_MODEL}" \
        --patchtst_num_layers "${N_LAYERS}" \
        --patchtst_num_heads "${N_HEADS}" \
        --patchtst_patch_length "${PATCH_LEN}" \
        --patchtst_patch_stride "${PATCH_STRIDE}" \
        --patchtst_dropout 0.1 \
        2>&1 | tee -a "${SUITE_LOG}"

      processed=$((processed + 1))
done

echo "[$(ts)] ${SUITE}: done" | tee -a "${SUITE_LOG}"
