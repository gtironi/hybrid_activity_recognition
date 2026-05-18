#!/usr/bin/env bash
# PatchTST MAE pretrain usando o RAW (Time_Adj_Raw_Data.csv) não rotulado.
#
# 1) Janela o CSV raw num parquet (se ainda não existe).
# 2) Roda o pretrain MAE em cima desse parquet.
#
# Output: experiments/patchtst_pretrain_raw_${DATASET_ID}_ep${PRETRAIN_EPOCHS}_bs${BATCH_SIZE}_lr${PRETRAIN_LR}_s${SEED}/best.pt
set -euo pipefail
source "$(dirname "$0")/_common.sh"
export BATCH_SIZE="${PATCHTST_BATCH_SIZE:-128}"

RAW_CSV="${RAW_CSV:-${REPO_ROOT}/dataset/Time_Adj_Raw_Data.csv}"
RAW_PARQUET="${RAW_PARQUET:-${REPO_ROOT}/dataset/processed/pretrain_raw_windowed.parquet}"
WINDOW_LEN="${WINDOW_LEN:-75}"
STRIDE="${STRIDE:-37}"

# Marca o run com sufixo "raw" para não colidir com o pretrain antigo.
export RUN_SUFFIX="raw"
OUT=$(make_run_dir "patchtst" "pretrain")

# 1) Gera o parquet janelado se ainda não existe.
if [ ! -f "$RAW_PARQUET" ]; then
    echo ">>> Janelando $RAW_CSV → $RAW_PARQUET"
    python "${REPO_ROOT}/scripts/window_raw_for_pretrain.py" \
        --input  "$RAW_CSV" \
        --output "$RAW_PARQUET" \
        --window_len "$WINDOW_LEN" \
        --stride "$STRIDE"
else
    echo ">>> $RAW_PARQUET já existe, pulando janelamento"
fi

# 2) Pretrain.
if [ -f "${OUT}/DONE" ]; then
    echo ">>> Pretrain raw já completo em ${OUT}, pulando"
    exit 0
fi
mkdir -p "${OUT}"

echo ">>> Pretrain PatchTST (MAE) sobre RAW em $(date)"
RESUME_ARGS=()
if [ -f "${OUT}/checkpoint.pt" ]; then
    echo ">>> Retomando de ${OUT}/checkpoint.pt"
    RESUME_ARGS+=(--checkpoint "${OUT}/checkpoint.pt")
fi

python -m hybrid_activity_recognition.main \
    --mode pretrain \
    --pretrain_parquet "$RAW_PARQUET" \
    --output_dir "$OUT" \
    --pretrain_epochs "$PRETRAIN_EPOCHS" \
    --pretrain_lr "$PRETRAIN_LR" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --device "$DEVICE" \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "${OUT}/train.log"

touch "${OUT}/DONE"
echo ">>> Pretrain raw done em $(date) → ${OUT}/best.pt"
