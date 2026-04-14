#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

SUITE="mixing"
SUITE_LOG="${LOG_ROOT}/${SUITE}.log"
: > "${SUITE_LOG}"

# Bloco imediato (suportado pela CLI atual):
# compara deep_only vs hybrid com a mesma arquitetura/pretreino.
D_MODEL="${MIX_D_MODEL:-128}"
N_LAYERS="${MIX_NUM_LAYERS:-3}"
N_HEADS="${MIX_NUM_HEADS:-4}"
PATCH_LEN="${MIX_PATCH_LENGTH:-8}"
PATCH_STRIDE="${MIX_PATCH_STRIDE:-8}"

TAG="proxy_d${D_MODEL}_l${N_LAYERS}_h${N_HEADS}_p${PATCH_LEN}_s${PATCH_STRIDE}"

echo "[$(ts)] ${SUITE}: running proxy mixing test ${TAG}" | tee -a "${SUITE_LOG}"
CKPT_PATH=$(run_pretrain_patchtst_variant \
    "${SUITE}_${TAG}" \
    --patchtst_d_model "${D_MODEL}" \
    --patchtst_num_layers "${N_LAYERS}" \
    --patchtst_num_heads "${N_HEADS}" \
    --patchtst_patch_length "${PATCH_LEN}" \
    --patchtst_patch_stride "${PATCH_STRIDE}" \
    --patchtst_dropout 0.1 \
    2>&1 | tee -a "${SUITE_LOG}" | tail -n 1)

for MODE in deep_only hybrid; do
    run_supervised_patchtst_variant \
        "${MODE}" \
        "${SUITE}_${TAG}" \
        "${CKPT_PATH}" \
        --patchtst_d_model "${D_MODEL}" \
        --patchtst_num_layers "${N_LAYERS}" \
        --patchtst_num_heads "${N_HEADS}" \
        --patchtst_patch_length "${PATCH_LEN}" \
        --patchtst_patch_stride "${PATCH_STRIDE}" \
        --patchtst_dropout 0.1 \
        2>&1 | tee -a "${SUITE_LOG}"
done

cat > "${RESULTS_ROOT}/mixing_pending_notes.md" <<'EOF'
# Mixing: pendencias tecnicas

Implementado agora:
- Comparacao proxy deep_only vs hybrid com arquitetura/pretreino controlados.

Pendente para fase futura (exige adaptacao fora da CLI atual):
- channel_attention explicito no encoder PatchTST.
- pooling/classification head alternativos para preservar estrutura entre canais.
EOF

echo "[$(ts)] ${SUITE}: done" | tee -a "${SUITE_LOG}"
