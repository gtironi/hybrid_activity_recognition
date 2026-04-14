#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

SMOKE_LOG="${LOG_ROOT}/run_smoke_patchtst_ablations.log"
: > "${SMOKE_LOG}"

echo "[$(ts)] starting PatchTST smoke run" | tee -a "${SMOKE_LOG}"

# Keep smoke run short and deterministic.
export SEEDS="${SEEDS:-42}"
export EPOCHS="${EPOCHS:-1}"
export PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export MAX_CONFIGS="${MAX_CONFIGS:-1}"
export K_VALUES="${K_VALUES:-5}"
export NORMALIZATION_VARIANTS="${NORMALIZATION_VARIANTS:-global_pipeline}"

bash "${ABL_ROOT}/scripts/run_teste_sem_pretreino.sh" 2>&1 | tee -a "${SMOKE_LOG}"
bash "${ABL_ROOT}/scripts/run_teste_tamanho_modelo.sh" 2>&1 | tee -a "${SMOKE_LOG}"
bash "${ABL_ROOT}/scripts/run_teste_mixing.sh" 2>&1 | tee -a "${SMOKE_LOG}"
bash "${ABL_ROOT}/scripts/run_teste_normalizacao.sh" 2>&1 | tee -a "${SMOKE_LOG}"
bash "${ABL_ROOT}/scripts/run_teste_topk.sh" 2>&1 | tee -a "${SMOKE_LOG}"

python "${ABL_ROOT}/results/summarize_results.py" 2>&1 | tee -a "${SMOKE_LOG}"

echo "[$(ts)] PatchTST smoke run done" | tee -a "${SMOKE_LOG}"
