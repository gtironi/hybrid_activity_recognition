# Reconhecimento de atividades em bezerros (AcTBeCalf)

Classificação de comportamentos a partir de acelerometria de alta frequência, com modelo **híbrido** (sinal em janela + features **TSFEL**) em PyTorch. Base de dados: [**AcTBeCalf**](https://zenodo.org/records/13259482). Opcionalmente, treino **semi-supervisionado (FixMatch)** com dados não rotulados.

**Participantes:** João Gabriel Machado e Gustavo Tironi (FGV EMAp) · disciplina Deep Learning (Mestrado).

---

## Instalação

```bash
pip install -r requirements.txt
```

Treino e avaliação assumem `PYTHONPATH` apontando para `src` (o script shell abaixo já exporta isso).

---

## Pipeline recomendado (estado atual do repositório)

O fluxo evita misturar treino e teste no mesmo Parquet quando a divisão já é por sujeito (**sem vazamento** de normalização: média/desvio do sinal e `StandardScaler` das features TSFEL são ajustados **só** nas janelas de treino).

| Etapa | O que faz |
|--------|-----------|
| 1. Raw → Parquet longo | `scripts/dataset_processing.py` gera `train.parquet` e `test.parquet` (ex.: `dataset/processed/AcTBeCalf/`). |
| 2. Parquet longo → janelado + TSFEL | `scripts/prepare_windowed_parquet.py`: no **treino** descobre as top-K features e grava o manifest JSON; no **teste** só **aplica** o mesmo conjunto de colunas. |
| 3. Treino / avaliação | `python -m hybrid_activity_recognition.main` com `--labeled_parquet_train` e `--labeled_parquet_test` (Parquets **já janelados**). |

### 1. Separar CSV por sujeito → Parquet bruto

```bash
python scripts/dataset_processing.py \
  --csv dataset/AcTBeCalf.csv \
  --out-dir dataset/processed
```

Saída típica: `dataset/processed/AcTBeCalf/train.parquet` e `test.parquet`.

### 2. Janelamento + TSFEL (dois runs)

**Treino (discover + manifest):**

```bash
python scripts/prepare_windowed_parquet.py \
  --input dataset/processed/AcTBeCalf/train.parquet \
  --output dataset/processed/AcTBeCalf/windowed_train.parquet \
  --feature-manifest-out dataset/processed/AcTBeCalf/tsfel_feature_manifest.json
```

**Teste (apply — mesmas features que o treino):**

```bash
python scripts/prepare_windowed_parquet.py \
  --input dataset/processed/AcTBeCalf/test.parquet \
  --output dataset/processed/AcTBeCalf/windowed_test.parquet \
  --feature-manifest-in dataset/processed/AcTBeCalf/tsfel_feature_manifest.json
```

`--help` lista janela, pureza, `fs`, batch TSFEL, nomes de colunas (`--group-by`, `--label-column`, `--acc-x`/`y`/`z`, etc.).

### 3. Treino com teste fixo

```bash
export PYTHONPATH=src
python -m hybrid_activity_recognition.main \
  --mode supervised \
  --model robust_hybrid \
  --labeled_parquet_train dataset/processed/AcTBeCalf/windowed_train.parquet \
  --labeled_parquet_test dataset/processed/AcTBeCalf/windowed_test.parquet \
  --epochs 50
```

- Validação: `--val_fraction` (fração estratificada **dentro do treino**, default `0.1`) ou `--labeled_parquet_val` com um Parquet janelado à parte.
- Modos: `supervised`, `finetune`, `fixmatch` (este exige `--unlabeled_parquet`), `test`.
- Modelos: `robust_hybrid`, `hybrid_cnn_lstm`.

### Orquestração em um comando

Dois separadores `--` repartem argumentos extra para: (1) prepare no train, (2) prepare no test, (3) `main`.

```bash
./scripts/run_windowed_hybrid_pipeline.sh \
  --batch-size 3000 \
  -- \
  --batch-size 5000 \
  -- \
  --mode supervised --model robust_hybrid --epochs 10
```

Caminhos padrão do shell: `dataset/processed/AcTBeCalf/` (entrada `train.parquet` / `test.parquet`, saída `windowed_*.parquet` e `tsfel_feature_manifest.json`).

---

## Modo legado: um único Parquet janelado

Se ainda tiver um único ficheiro com todas as janelas, o split interno 80/10/10 continua disponível:

```bash
PYTHONPATH=src python -m hybrid_activity_recognition.main \
  --mode supervised --model robust_hybrid \
  --labeled_parquet caminho/para/WindowedCalf.parquet --epochs 50
```

Não combine `--labeled_parquet` com `--labeled_parquet_train` / `--labeled_parquet_test`.

---

## Estrutura do repositório

| Caminho | Descrição |
|---------|-----------|
| `src/hybrid_activity_recognition/` | Pacote instalável por `PYTHONPATH=src`: `data`, `models`, `training`, `main`. |
| `scripts/dataset_processing.py` | CSV → `train.parquet` / `test.parquet` por coluna de sujeito. |
| `scripts/prepare_windowed_parquet.py` | Parquet longo → janelas + TSFEL + manifest. |
| `scripts/run_windowed_hybrid_pipeline.sh` | Encadeia prepare ×2 + `main` com defaults AcTBeCalf. |
| `notebooks/` | EDA e análises atuais (ex.: exploração por bezerro). |
| `old/` | Notebooks e scripts de iteração anterior (janelamento monolítico, RF, etc.); referência histórica, não o caminho preferido do pipeline. |

---

## Checkpoints e saídas

Por defeito, artefactos em `experiments/runs/` (ex.: `supervised_best.pt`, `finetuned_best.pt`, `fixmatch_best.pt`). Use `--output_dir` e `--checkpoint` conforme o modo (`--help` no `main`).
