#!/usr/bin/env bash
# Converte todos os datasets padronizados e treina PatchTST (HF) para cada preset.
# Uso (na raiz do repo ou de qualquer lugar):
#   bash hugging/patchtst/run_all.sh
# Variáveis opcionais:
#   EPOCHS=5 DEVICE=cuda DOG_WINDOW=128 DOG_STRIDE=64
#   PRETRAIN=1 PRETRAIN_EPOCHS=10 — SSL mascarado + classificação em cada preset (train_classification_debug)
#   SKIP_CONVERT=1   — só treina (CSV já gerados)
#   SKIP_TRAIN=1       — só converte
#   ACTBECALF_MAX_ROWS — ex.: 500000; vazio = arquivo inteiro (pode demorar / usar muita RAM)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"
DOG_WINDOW="${DOG_WINDOW:-128}"
DOG_STRIDE="${DOG_STRIDE:-64}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"

py() {
  python -m "$@"
}

if [[ -z "${SKIP_CONVERT:-}" ]]; then
  echo "=== Convert: HAR UCI ==="
  py hugging.patchtst.convert.convert_har_uci

  echo "=== Convert: dog (w10, w50, w100, raw) ==="
  for stem in dog_w10 dog_w50 dog_w100 dog_raw; do
    py hugging.patchtst.convert.convert_dog \
      --input "${REPO_ROOT}/data/dog/${stem}.csv" \
      --window "${DOG_WINDOW}" \
      --stride "${DOG_STRIDE}"
  done

  echo "=== Convert: AcTBeCalf ==="
  if [[ -n "${ACTBECALF_MAX_ROWS:-}" ]]; then
    py hugging.patchtst.convert.convert_actbecalf --max_rows "${ACTBECALF_MAX_ROWS}"
  else
    py hugging.patchtst.convert.convert_actbecalf
  fi

  echo "=== Convert: ETTm1 (proxy hour-of-day) ==="
  py hugging.patchtst.convert.convert_ettm1
else
  echo "SKIP_CONVERT=1 — pulando conversores"
fi

if [[ -z "${SKIP_TRAIN:-}" ]]; then
  echo "=== Train: todos os presets ==="
  TRAIN_EXTRA=()
  if [[ -n "${PRETRAIN:-}" ]]; then
    TRAIN_EXTRA+=(--pretrain --pretrain_epochs "${PRETRAIN_EPOCHS}")
    echo "(PRETRAIN=1: ${PRETRAIN_EPOCHS} épocas SSL + ${EPOCHS} épocas supervisionadas por preset)"
  fi
  for preset in har ettm1 actbecalf dog_w10 dog_w50 dog_w100 dog_raw; do
    echo "--- preset: ${preset} ---"
    py hugging.patchtst.train_classification_debug \
      --preset "${preset}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --device "${DEVICE}" \
      "${TRAIN_EXTRA[@]}"
  done
  echo "Concluído. Runs em: ${REPO_ROOT}/hugging/patchtst/runs/"
else
  echo "SKIP_TRAIN=1 — pulando treino"
fi
