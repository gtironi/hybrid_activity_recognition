#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

MASTER_LOG="${LOG_ROOT}/run_all_patchtst_ablations.log"
: > "${MASTER_LOG}"

echo "[$(ts)] starting all PatchTST ablations" | tee -a "${MASTER_LOG}"

bash "${ABL_ROOT}/scripts/run_teste_sem_pretreino.sh" 2>&1 | tee -a "${MASTER_LOG}"
bash "${ABL_ROOT}/scripts/run_teste_tamanho_modelo.sh" 2>&1 | tee -a "${MASTER_LOG}"
bash "${ABL_ROOT}/scripts/run_teste_mixing.sh" 2>&1 | tee -a "${MASTER_LOG}"
bash "${ABL_ROOT}/scripts/run_teste_normalizacao.sh" 2>&1 | tee -a "${MASTER_LOG}"
bash "${ABL_ROOT}/scripts/run_teste_topk.sh" 2>&1 | tee -a "${MASTER_LOG}"

python "${ABL_ROOT}/results/summarize_results.py" 2>&1 | tee -a "${MASTER_LOG}"

echo "[$(ts)] all PatchTST ablations done" | tee -a "${MASTER_LOG}"
