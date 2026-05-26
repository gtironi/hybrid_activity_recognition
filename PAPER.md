# Paper-Feature Experiments — Dissanayake et al. (2025)

**Reproduce the Hand-Crafted, Catch22, and miniROCKET feature pipelines from [Dissanayake et al. (2025)](https://www.sciencedirect.com/science/article/pii/S0168169925009330) inside the hybrid framework, so each feature set can be compared head-to-head against TSFEL and against the deep encoders alone.**

The paper extracts three feature sets from 8 accelerometer-derived time-series per window and feeds them to classical ML models (XGBoost, RandomForest, RidgeClassifierCV). Here we plug the **same feature representations** into the hybrid pipeline (`deep_only`, `hybrid`, `tsfel_only` modes) and into an MLP / RF baseline.

The original paper's code is bundled in [`calfBehaviourEval/`](calfBehaviourEval/) and the ROCKET reference implementation in [`rocket/`](rocket/).

---

## What's new

Only three files are added; **the hybrid framework itself is unchanged** — feature columns are auto-detected by the existing dataloader and consumed by `MLPTsfelBranch` exactly as TSFEL features would be.

| File | Purpose |
|------|---------|
| [`scripts/prepare_paper_features_parquet.py`](scripts/prepare_paper_features_parquet.py) | Raw row-parquet → windowed parquet with raw signals **plus** paper features as columns |
| [`scripts/experiments/run_paper_features.sh`](scripts/experiments/run_paper_features.sh) | One-shot runner: builds datasets, pretrains, runs the full DL + baseline grid |
| [`requirements.txt`](requirements.txt) | New deps: `aeon`, `pycatch22`, `antropy`, `joblib` |

---

## Installation

The new feature extractors need three additional packages. From the activated venv:

```bash
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

This installs (in addition to the existing deps):

- `aeon>=0.11.0` — `MiniRocket` (multivariate)
- `pycatch22>=0.4.2` — `catch22_all(catch24=True)`
- `antropy>=0.1.6` — spectral entropy used by the paper's HC features
- `joblib>=1.3.0` — persists the fitted MiniRocket model between train and test

---

## Quick Start

End-to-end:

```bash
# 0) Environment (only needed once)
source venv/bin/activate
export PYTHONPATH=src

# 1) Raw CSV → train/test (skip if already done — see README.md step 1)
python scripts/dataset_processing.py \
  --csv dataset/AcTBeCalf.csv \
  --out-dir dataset/processed

# 2) Full paper-feature experimental grid (builds datasets, pretrains, runs everything)
bash scripts/experiments/run_paper_features.sh
```

The runner is **fully idempotent**: re-running it skips any step whose output (`DONE` marker, parquet, or `joblib`) already exists.

**For SSH / detached execution:**

```bash
mkdir -p logs
LOG="logs/run_paper_features_$(date +%Y%m%d_%H%M%S).log"
nohup bash -c "source venv/bin/activate && bash scripts/experiments/run_paper_features.sh" \
  > "$LOG" 2>&1 < /dev/null &
disown
echo "PID: $! | log: $LOG"
```

Or with `screen` / `tmux` as in [README.md](README.md#3-run-full-experimental-grid).

---

## Pipeline overview

```
dataset/processed/AcTBeCalf/{train,test}.parquet       (row-level, from dataset_processing.py)
                ↓
   scripts/prepare_paper_features_parquet.py
   ├─ 5 derived signals (Amag, ODBA, VeDBA, pitch, roll) from raw X/Y/Z
   ├─ 125-sample windows (5 s @ 25 Hz), 50 % overlap, ≥ 90 % label purity
   └─ HC / Catch22 / miniROCKET features over the 8 series
                ↓
dataset/processed/AcTBeCalf/paper_features/
   ├─ windowed_hc_{train,test}.parquet       (raw acc + 88 feature cols)
   ├─ windowed_catch22_{train,test}.parquet  (raw acc + 192 feature cols)
   ├─ windowed_rocket_{train,test}.parquet   (raw acc + 9996 feature cols)
   └─ rocket_model.joblib                    (fitted MiniRocket, reused for test)
                ↓
   scripts/experiments/run_paper_features.sh
   ├─ TS2Vec pretrain (cnn_lstm, robust) — once, on 125-sample raw windows
   ├─ PatchTST MAE pretrain              — once, on 125-sample raw windows
   └─ For each feature set:
        ├─ CNN-LSTM  × {deep_only, hybrid} × {fromscratch, frompretrain} + finetune
        ├─ Robust    × same
        ├─ PatchTST  × same
        ├─ tsfel_mlp tsfel_only + finetune
        └─ RandomForest baseline
                ↓
experiments/paper_features/{hc,catch22,rocket}/<run_dir>/
```

---

## Feature extraction — what `prepare_paper_features_parquet.py` does

Matches [`calfBehaviourEval/Feature datasets derivation.ipynb`](calfBehaviourEval/Feature%20datasets%20derivation.ipynb) and [`Preprocessing and Windowing.ipynb`](calfBehaviourEval/Preprocessing%20and%20Windowing.ipynb).

**1. Derive 5 signals per sample** (no Butterworth filter — exactly as the paper's notebook):

| Series | Formula | Source |
|--------|---------|--------|
| `Amag` | `√(X²+Y²+Z²)` | `calculate_magnitude` |
| `ODBA` | `\|X\|+\|Y\|+\|Z\|` | `calculate_ODBA` |
| `VeDBA` | `√(X²+Y²+Z²)` | `calculate_VeDBA` |
| `pitch` | `180·atan2(Z, √(Y²+X²))/π` | `calculate_pitch` |
| `roll` | `180·atan2(Y, √(X²+Z²))/π` | `calculate_roll` |

**2. Window all 8 series** `[accX, accY, accZ, Amag, ODBA, VeDBA, pitch, roll]` with stride = window_size × (1 - overlap) per `(calfId, segId)` group; keep only windows where the dominant label covers ≥ purity_threshold of samples.

**3. Compute features** per window over the 8 series:

| Feature set | Per series | Total | Library | Column prefix |
|-------------|------------|-------|---------|---------------|
| **HC** | 11 stats (mean, median, min, max, std, q1, q3, spectral_entropy, motion_variation, kurtosis, skewness) | **88** | `calfBehaviourEval.libraries.hand_crafted_features.return_HC_features` | `hc_<series>_<stat>` |
| **Catch22** | 24 (22 canonical + mean + std) | **192** | `pycatch22.catch22_all(..., catch24=True)` | `c22_<series>_<name>` |
| **ROCKET** | — | **9996** | `aeon.transformations.collection.convolution_based.MiniRocket(n_kernels=10000, random_state=42)` on the 8-channel multivariate window | `rocket_<i>` |

Windows where any series returns NaN / non-numeric features are dropped (HC + Catch22 only).

**ROCKET train ⇄ test consistency**: `--rocket-manifest-out` (train) writes the fitted `MiniRocket` via `joblib.dump`; `--rocket-manifest-in` (test) reloads it with `joblib.load`. Train and test features are guaranteed identical for the same input window.

### CLI

```bash
python scripts/prepare_paper_features_parquet.py \
  --features {hc,catch22,rocket} \
  --input  dataset/processed/AcTBeCalf/train.parquet \
  --output dataset/processed/AcTBeCalf/paper_features/windowed_hc_train.parquet \
  [--rocket-manifest-out PATH]   # rocket train
  [--rocket-manifest-in  PATH]   # rocket test
  [--window-size 125] [--overlap 0.5] [--purity-threshold 0.9] [--fs 25] \
  [--group-by calfId segId] [--time-column dateTime] [--label-column behaviour] \
  [--acc-x accX] [--acc-y accY] [--acc-z accZ]
```

Output parquet schema:

```
dateTime | calf_id | acc_x (list) | acc_y (list) | acc_z (list) | label | [feature cols...]
```

`acc_x/y/z` feed the signal encoders; feature columns feed `MLPTsfelBranch`, the MLP baseline, and the RandomForest baseline.

---

## What `run_paper_features.sh` does, step by step

### Stage 0 — Setup

- Sets `WINDOW_LEN=125`, `WINDOW_STRIDE=63` (5 s @ 25 Hz with 50 % overlap).
- Sets `DATASET_ID=AcTBeCalf_paper_w125` so pretrain checkpoints don't collide with the existing 3 s-window pipeline.
- Verifies the source row-parquets exist (`dataset/processed/AcTBeCalf/{train,test}.parquet`).

### Stage 1 — Build the 3 feature parquets

Calls `prepare_paper_features_parquet.py` six times (train + test for each feature set). Skips outputs that already exist.

Disk footprint: HC ≈ tens of MB · Catch22 ≈ ~hundred MB · **ROCKET ≈ several GB** (10 000 columns × N windows).

### Stage 2 — Pretrain encoders (shared across all 3 feature sets)

Because raw accelerometer signals are identical across feature sets, pretraining runs **once** on a 125-sample-windowed slice of the unlabeled raw CSV (`dataset/Time_Adj_Raw_Data.csv`):

| Encoder | Method | Epochs | Final checkpoint used |
|--------|--------|--------|----------------------|
| `cnn_lstm` | TS2Vec contrastive | `PRETRAIN_EPOCHS` (100) | `ts2vec_ep100.pt` |
| `robust` | TS2Vec contrastive | `PRETRAIN_EPOCHS` (100) | `ts2vec_ep100.pt` |
| `patchtst` | MAE | `PATCHTST_PRETRAIN_EPOCHS` (40) | `pretrain_ep40.pt` |

Output:

```
experiments/pretrain/
├── ts2vec_pretrain_cnn_lstm_AcTBeCalf_paper_w125_ep100_s2026/
├── ts2vec_pretrain_robust_AcTBeCalf_paper_w125_ep100_s2026/
└── patchtst_pretrain_raw_AcTBeCalf_paper_w125_ep40_bs1536_lr1e-3_s2026/
```

> Always the **last pretrain epoch** — no frozen-encoder runs, no multi-checkpoint sweep.

### Stage 3 — Supervised + baselines, per feature set

For each `FEAT ∈ {hc, catch22, rocket}` the script runs 14 supervised configs (each with Stage-1 + Stage-2 finetune) + 1 RF baseline:

| Model | Mode | Init | Finetune |
|-------|------|------|----------|
| cnn_lstm | deep_only | fromscratch | ✓ |
| cnn_lstm | deep_only | frompretrain_raw_ep100 | ✓ |
| cnn_lstm | hybrid | fromscratch | ✓ |
| cnn_lstm | hybrid | frompretrain_raw_ep100 | ✓ |
| robust | deep_only | fromscratch | ✓ |
| robust | deep_only | frompretrain_raw_ep100 | ✓ |
| robust | hybrid | fromscratch | ✓ |
| robust | hybrid | frompretrain_raw_ep100 | ✓ |
| patchtst | deep_only | fromscratch | ✓ |
| patchtst | deep_only | frompretrain_raw_ep40 | ✓ |
| patchtst | hybrid | fromscratch | ✓ |
| patchtst | hybrid | frompretrain_raw_ep40 | ✓ |
| tsfel_mlp | tsfel_only | — | ✓ |
| RandomForest | tsfel_only | — | — |

PatchTST always reads raw signals through its patch tokenizer; in `hybrid` mode the paper-feature columns are projected through `MLPTsfelBranch` and concatenated with the PatchTST embedding before the head.

**Stage-1 supervised**: balanced cross-entropy, AdamW, early-stopping on val balanced accuracy (patience 25, max 200 epochs).  
**Stage-2 finetune**: combines train+val, plain CE, lr = 1e-4, 20 epochs, no early stopping.

Per-feature-set output:

```
experiments/paper_features/<feat>/
├── cnn_lstm_deep_only_fromscratch_AcTBeCalf_paper_<feat>_s2026/
│   ├── train.log, finetune.log
│   ├── best.pt
│   ├── test_metrics.json, test_confusion_matrix.png
│   ├── DONE, DONE_finetune
├── ... (13 more DL configs)
├── tsfel_mlp_tsfel_only_AcTBeCalf_paper_<feat>_s2026/
└── rf_baseline_AcTBeCalf_paper_<feat>_s2026/
```

---

## Configuration

Environment variables (override before launching) — all inherit defaults from [`scripts/experiments/_common.sh`](scripts/experiments/_common.sh):

| Variable | Default | Description |
|----------|---------|-------------|
| `WINDOW_LEN` | 125 | Window size in samples (5 s @ 25 Hz) |
| `WINDOW_STRIDE` | 63 | Stride (50 % overlap) |
| `SEED` | 2026 | Global seed |
| `DEVICE` | cuda | `cuda` / `cpu` |
| `EPOCHS` | 200 | Max Stage-1 supervised epochs (ES patience = 25) |
| `FINETUNE_EPOCHS` | 20 | Stage-2 finetune epochs |
| `PRETRAIN_EPOCHS` | 100 | TS2Vec pretrain epochs |
| `PATCHTST_PRETRAIN_EPOCHS` | 40 | PatchTST MAE pretrain epochs |
| `BATCH_SIZE_LARGE` | 512 | CNN-LSTM / Robust / MLP batch size |
| `PATCHTST_BATCH_SIZE` | 128 | PatchTST supervised batch size |
| `PATCHTST_PRETRAIN_BATCH_SIZE` | 1536 | PatchTST MAE batch size |
| `LR` | 1e-3 | Stage-1 lr (Stage-2 always 1e-4) |
| `PRETRAIN_LR` | 1e-3 | Pretrain lr |

Example — bigger seed, longer Stage-1:

```bash
SEED=2027 EPOCHS=300 bash scripts/experiments/run_paper_features.sh
```

---

## Manual building blocks

You can also run pieces independently — useful for inspection or for replacing one feature set without recomputing the others.

**Build a single HC parquet:**

```bash
python scripts/prepare_paper_features_parquet.py \
  --features hc \
  --input  dataset/processed/AcTBeCalf/train.parquet \
  --output dataset/processed/AcTBeCalf/paper_features/windowed_hc_train.parquet \
  --window-size 125 --overlap 0.5 --purity-threshold 0.9
```

**ROCKET — fit on train, then transform test:**

```bash
python scripts/prepare_paper_features_parquet.py --features rocket \
  --input  dataset/processed/AcTBeCalf/train.parquet \
  --output dataset/processed/AcTBeCalf/paper_features/windowed_rocket_train.parquet \
  --rocket-manifest-out dataset/processed/AcTBeCalf/paper_features/rocket_model.joblib

python scripts/prepare_paper_features_parquet.py --features rocket \
  --input  dataset/processed/AcTBeCalf/test.parquet \
  --output dataset/processed/AcTBeCalf/paper_features/windowed_rocket_test.parquet \
  --rocket-manifest-in  dataset/processed/AcTBeCalf/paper_features/rocket_model.joblib
```

**Train a single supervised run on the HC parquet:**

```bash
TRAIN_PARQUET=dataset/processed/AcTBeCalf/paper_features/windowed_hc_train.parquet \
TEST_PARQUET=dataset/processed/AcTBeCalf/paper_features/windowed_hc_test.parquet  \
DATASET_ID=AcTBeCalf_paper_hc \
EXPERIMENTS_BASE=experiments/paper_features/hc \
bash -c '
  source scripts/experiments/_common.sh
  RUN_SUFFIX=fromscratch run_experiment robust hybrid
  RUN_SUFFIX=fromscratch run_finetune   robust hybrid
'
```

**RandomForest baseline on any feature parquet:**

```bash
PYTHONPATH=src python -m random_forest_baseline.tsfel_baseline \
  --train dataset/processed/AcTBeCalf/paper_features/windowed_hc_train.parquet \
  --test  dataset/processed/AcTBeCalf/paper_features/windowed_hc_test.parquet \
  --output_dir experiments/paper_features/hc/rf_only \
  --n_estimators 200
```

---

## Resume behavior

Same conventions as the main `run_all.sh`:

- `DONE` file in a run dir → Stage 1 skipped.
- `DONE_finetune` file → Stage 2 skipped.
- `checkpoint.pt` in a run dir → Stage 1 resumes from that epoch.
- ROCKET parquets and `rocket_model.joblib` are skipped if they already exist.

To force a single experiment to re-run: `rm -rf <its_run_dir>`. To force a feature parquet to be rebuilt: delete it (and also `rocket_model.joblib` if you're rebuilding ROCKET — train must run before test).

---

## What to expect — comparing against the paper

The paper's Table 6 / Fig. 4 give RidgeClassifierCV balanced accuracy at the best preprocessing config:

| Feature set | Paper RCV BA | Paper's best window config |
|-------------|--------------|---------------------------|
| Hand-Crafted | 0.66 | 5 s, 0 % overlap |
| Catch22 | 0.74 | 5 s, 50 % overlap |
| ROCKET | 0.81 | 5 s, 25 % overlap |

We fix **all 3 feature sets at 5 s / 50 % overlap** (slightly off the paper's per-feature optima for HC and ROCKET) and **replace RCV with RandomForest + MLP + the hybrid DL pipeline**. The most informative comparisons are:

1. **`*_deep_only_fromscratch`** vs **`*_hybrid_fromscratch`** — does the feature set actually help the deep encoder?
2. **`*_hybrid_fromscratch`** vs **`*_hybrid_frompretrain_raw_*`** — does self-supervised pretrain still help when features are present?
3. **`rf_baseline` / `tsfel_mlp_tsfel_only`** — how good is the feature set on its own (no raw signal)?
4. **HC vs Catch22 vs ROCKET** within the same model+mode — which feature representation generalises best?

---

## Troubleshooting

**`FileNotFoundError: dataset/processed/AcTBeCalf/train.parquet`**
→ Run `scripts/dataset_processing.py` first (see [README.md](README.md#1-prepare-data) step 1).

**`ImportError: cannot import name 'MiniRocketMultivariate' from 'aeon...'`**
→ aeon ≥ 1.0 collapsed the multivariate variant into the unified `MiniRocket` class. Our script uses `MiniRocket` directly — make sure `aeon>=0.11.0` is installed (`pip install -r requirements.txt`).

**`Manifest ROCKET não encontrado: ...rocket_model.joblib`**
→ The ROCKET test run is launched before the train run finished. Delete the train parquet, run the train extraction first, then the test extraction (the master script does this automatically).

**`ERROR: missing .../ts2vec_ep100.pt`** (or `pretrain_ep40.pt`)
→ Pretraining failed or was interrupted. Check `experiments/pretrain/.../pretrain.log`. Delete the pretrain dir and re-run if needed.

**ROCKET parquet is huge / disk full**
→ Expected (9 996 columns × N windows). Either drop ROCKET (`for FEAT in hc catch22`) or regenerate from `rocket_model.joblib` only when needed.

**`tsfel RuntimeWarning: catastrophic cancellation`** during HC / Catch22
→ Harmless: near-constant windows trip scipy moment calculations. Already silenced via `warnings.filterwarnings`.

---

## References

- **Dissanayake, O. et al. (2025).** "Hand-Crafted, Catch22 and ROCKET features for calf-behaviour classification from accelerometer data." *Computers and Electronics in Agriculture* 238, 110799. (PDF: [`Rocket.pdf`](Rocket.pdf))
- **Lubba, C. H. et al. (2019).** "catch22: CAnonical Time-series CHaracteristics." *Data Mining and Knowledge Discovery* 33, 1821–1852. — `pycatch22`
- **Dempster, A. et al. (2021).** "MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification." *KDD 2021*. — `aeon` `MiniRocket`

---

## Source-of-truth references in this repo

- Paper code (8 series, HC, Catch22, ROCKET): [`calfBehaviourEval/Feature datasets derivation.ipynb`](calfBehaviourEval/Feature%20datasets%20derivation.ipynb)
- Paper preprocessing (no Butterworth, derived signals from raw X/Y/Z): [`calfBehaviourEval/Preprocessing and Windowing.ipynb`](calfBehaviourEval/Preprocessing%20and%20Windowing.ipynb)
- HC stat definitions: [`calfBehaviourEval/libraries/hand_crafted_features.py`](calfBehaviourEval/libraries/hand_crafted_features.py)
- Derived signal formulae: [`calfBehaviourEval/libraries/functions.py`](calfBehaviourEval/libraries/functions.py)
- Hybrid framework dataloader (auto-detects feature columns): [`src/hybrid_activity_recognition/data/dataloader.py`](src/hybrid_activity_recognition/data/dataloader.py)
- Existing TSFEL pipeline (for comparison): [`scripts/prepare_windowed_parquet.py`](scripts/prepare_windowed_parquet.py)
