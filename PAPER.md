# Paper-Feature Experiments

Replicates the feature-extraction methodology of **Dissanayake et al. (2025)** — Hand-Crafted, Catch22 and miniROCKET — and plugs each feature set into the hybrid framework so it can be compared head-to-head against the existing TSFEL pipeline.

The original paper's code lives in [`calfBehaviourEval/`](calfBehaviourEval/) and the ROCKET reference is in [`rocket/`](rocket/). Everything new is in:

| File | Purpose |
|------|---------|
| [`scripts/prepare_paper_features_parquet.py`](scripts/prepare_paper_features_parquet.py) | Raw row-parquet → windowed parquet with raw signals **plus** paper features as columns |
| [`scripts/experiments/run_paper_features.sh`](scripts/experiments/run_paper_features.sh) | One-shot runner: builds datasets, pretrains, runs the full DL + baseline grid |
| [`requirements.txt`](requirements.txt) | New deps: `aeon`, `pycatch22`, `antropy`, `joblib` |

The hybrid framework itself was **not modified**: feature columns are auto-detected by the existing dataloader and consumed by `MLPTsfelBranch` exactly as TSFEL features would be.

---

## Quick start

```bash
source venv/bin/activate
bash scripts/experiments/run_paper_features.sh
```

The script is fully idempotent. Re-running it skips any step whose output (`DONE` marker or parquet) already exists.

---

## What the script does, step by step

### 0 — Setup

- Forces window length to **125 samples = 5 s @ 25 Hz** with **50 % overlap** (stride 63).
- Sets `DATASET_ID=AcTBeCalf_paper_w125` so pretrain checkpoint paths don't collide with the existing 3 s-window pipeline.
- Verifies the source row-parquets exist:
  - `dataset/processed/AcTBeCalf/train.parquet`
  - `dataset/processed/AcTBeCalf/test.parquet`

### 1 — Build three feature parquets

For each of `{hc, catch22, rocket}` × `{train, test}`, calls [`prepare_paper_features_parquet.py`](scripts/prepare_paper_features_parquet.py), which:

1. Reads the raw row-parquet.
2. Adds 5 derived columns per sample (no Butterworth filter, exactly as in [`calfBehaviourEval/Preprocessing and Windowing.ipynb`](calfBehaviourEval/Preprocessing%20and%20Windowing.ipynb)):
   - `Amag = √(X²+Y²+Z²)`
   - `ODBA = |X|+|Y|+|Z|`
   - `VeDBA = √(X²+Y²+Z²)`
   - `pitch = 180·atan2(Z, √(Y²+X²))/π`
   - `roll = 180·atan2(Y, √(X²+Z²))/π`
3. Slides a 125-sample window with stride 63 over each `(calfId, segId)` group, keeping windows whose dominant label covers ≥ 90 % of samples.
4. Computes features per window over the 8 series `[accX, accY, accZ, Amag, ODBA, VeDBA, pitch, roll]`:

   | Feature set | How | Cols per series | Total cols | Column prefix |
   |-------------|-----|----------------|-----------|---------------|
   | **HC** | `return_HC_features` from `calfBehaviourEval/libraries/hand_crafted_features.py` | 11 | **88** | `hc_<series>_<stat>` |
   | **Catch22** | `pycatch22.catch22_all(..., catch24=True)` (22 + mean + std) | 24 | **192** | `c22_<series>_<name>` |
   | **ROCKET** | `aeon.MiniRocket(n_kernels=10000, random_state=42)` on the 8-channel window | — | **9996** | `rocket_<i>` |

   For ROCKET, the train run fits the model and writes it to `rocket_model.joblib`; the test run reloads that exact model.

Output parquet schema (used by both the DL pipeline and the RF baseline):
```
dateTime | calf_id | acc_x (list) | acc_y (list) | acc_z (list) | label | [feature cols...]
```

Output directory:
```
dataset/processed/AcTBeCalf/paper_features/
├── rocket_model.joblib
├── windowed_hc_train.parquet         (88 feat cols)
├── windowed_hc_test.parquet
├── windowed_catch22_train.parquet    (192 feat cols)
├── windowed_catch22_test.parquet
├── windowed_rocket_train.parquet     (9996 feat cols)
└── windowed_rocket_test.parquet
```

### 2 — Pretrain encoders (shared across all 3 feature sets)

Because the raw accelerometer signals are identical across feature sets, pretraining runs **once** on a 125-sample-windowed slice of the unlabeled raw CSV (`dataset/Time_Adj_Raw_Data.csv`):

| Encoder | Mode | Checkpoint used downstream |
|--------|------|---------------------------|
| `cnn_lstm` | TS2Vec, 100 epochs | `ts2vec_ep100.pt` |
| `robust` | TS2Vec, 100 epochs | `ts2vec_ep100.pt` |
| `patchtst` | MAE, 40 epochs | `pretrain_ep40.pt` |

Output:
```
experiments/pretrain/
├── ts2vec_pretrain_cnn_lstm_AcTBeCalf_paper_w125_ep100_s2026/
├── ts2vec_pretrain_robust_AcTBeCalf_paper_w125_ep100_s2026/
└── patchtst_pretrain_raw_AcTBeCalf_paper_w125_ep40_bs1536_lr1e-3_s2026/
```

> **No frozen-encoder runs and no multi-checkpoint sweep** — we always use the last pretrain epoch.

### 3 — Supervised experiments + baselines, per feature set

For each `FEAT ∈ {hc, catch22, rocket}`, the script runs **14 supervised configs + 1 RF baseline = 27 output dirs** (Stage 1 + Stage 2 are two runs each):

| Model | Mode | Init | Stage 2 (finetune) |
|-------|------|------|--------------------|
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

PatchTST always reads raw signals via its patch tokenizer; in `hybrid` mode the paper-feature columns are projected through `MLPTsfelBranch` and concatenated with the PatchTST embedding before the head.

Stage-1 supervised: balanced cross-entropy, AdamW, early-stopping on validation balanced accuracy (patience 25 epochs, max 200).  
Stage-2 finetune: combines train+val, plain CE, lr=1e-4, 20 epochs, no early stopping.

Output per feature set:
```
experiments/paper_features/<feat>/
├── cnn_lstm_deep_only_fromscratch_AcTBeCalf_paper_<feat>_s2026/
│   ├── train.log
│   ├── finetune.log
│   ├── best.pt
│   ├── test_metrics.json
│   ├── test_confusion_matrix.png
│   ├── DONE
│   └── DONE_finetune
├── cnn_lstm_deep_only_frompretrain_raw_ep100_AcTBeCalf_paper_<feat>_s2026/
├── … (12 more DL configs)
├── tsfel_mlp_tsfel_only_AcTBeCalf_paper_<feat>_s2026/
└── rf_baseline_AcTBeCalf_paper_<feat>_s2026/
    ├── train.log
    ├── test_metrics.json
    └── test_confusion_matrix.png
```

---

## Resource & time estimates

| Phase | Approx. cost |
|------|-------------|
| Build HC / Catch22 parquets | minutes (CPU-bound) |
| Build ROCKET parquet | ~10–30 min, mostly RAM-bound; `windowed_rocket_train.parquet` is **~10 000 columns × N windows**, on the order of a few GB |
| TS2Vec pretrain (×2 encoders) | a few hours each on GPU |
| PatchTST MAE pretrain | a few hours on GPU |
| Stage-1 supervised + finetune (×42 DL runs) | the bulk of the wall time |
| RF baselines (×3) | minutes |

The ROCKET parquet is by far the largest artifact on disk. If you need to free space later, `dataset/processed/AcTBeCalf/paper_features/windowed_rocket_*.parquet` can be regenerated from `rocket_model.joblib`.

---

## What to expect — comparing against the paper

The paper's Table 6 / Fig. 4 give RidgeClassifierCV balanced accuracy at the best preprocessing config:

| Feature set | Paper RCV BA (best window config) |
|-------------|------------------------------------|
| Hand-Crafted | 0.66 (5 s, 0 % overlap) |
| Catch22 | 0.74 (5 s, 50 % overlap) |
| ROCKET | 0.81 (5 s, 25 % overlap) |

Our experiments use a **fixed 5 s / 50 % overlap** config for all three (slightly off the paper's per-feature optima), and we **replace RCV with RandomForest + MLP + the hybrid DL pipeline**. The hybrid pipeline allows the deep encoder to do most of the work and the feature columns to act as an auxiliary signal — so the most informative comparisons are:

1. **`*_deep_only_fromscratch`** vs **`*_hybrid_fromscratch`** — does the feature set actually help the deep model?
2. **`*_hybrid_fromscratch`** vs **`*_hybrid_frompretrain_raw_*`** — does self-supervised pretrain still help when features are present?
3. **`rf_baseline` / `tsfel_mlp_tsfel_only`** — how good is the feature set on its own (no raw signal)?

The dataset labels are the canonical AcTBeCalf set produced by [`scripts/dataset_processing.py`](scripts/dataset_processing.py) (see [`split_report.json`](dataset/processed/AcTBeCalf/split_report.json) for the exact label distribution and subject split).

---

## Manual building blocks

You can also run pieces individually.

**Build a single feature parquet:**

```bash
python scripts/prepare_paper_features_parquet.py \
  --features hc \
  --input  dataset/processed/AcTBeCalf/train.parquet \
  --output dataset/processed/AcTBeCalf/paper_features/windowed_hc_train.parquet \
  --window-size 125 --overlap 0.5 --purity-threshold 0.9
```

ROCKET requires fitting on train first, then transforming test with the same model:

```bash
python scripts/prepare_paper_features_parquet.py --features rocket \
  --input  .../train.parquet --output .../windowed_rocket_train.parquet \
  --rocket-manifest-out .../rocket_model.joblib

python scripts/prepare_paper_features_parquet.py --features rocket \
  --input  .../test.parquet --output .../windowed_rocket_test.parquet \
  --rocket-manifest-in  .../rocket_model.joblib
```

**Tweak the global hyperparameters by env var** before launching the runner:

```bash
SEED=2027 EPOCHS=300 PRETRAIN_EPOCHS=100 PATCHTST_PRETRAIN_EPOCHS=40 \
bash scripts/experiments/run_paper_features.sh
```

---

## Source-of-truth references

- Paper code (8 series, HC, Catch22, ROCKET): [`calfBehaviourEval/Feature datasets derivation.ipynb`](calfBehaviourEval/Feature%20datasets%20derivation.ipynb)
- Paper preprocessing (no Butterworth, derived signals from raw X/Y/Z): [`calfBehaviourEval/Preprocessing and Windowing.ipynb`](calfBehaviourEval/Preprocessing%20and%20Windowing.ipynb)
- HC stat definitions: [`calfBehaviourEval/libraries/hand_crafted_features.py`](calfBehaviourEval/libraries/hand_crafted_features.py)
- Derived signal formulae: [`calfBehaviourEval/libraries/functions.py`](calfBehaviourEval/libraries/functions.py)
- Hybrid framework dataloader (auto-detects feature columns): [`src/hybrid_activity_recognition/data/dataloader.py`](src/hybrid_activity_recognition/data/dataloader.py)
- Existing TSFEL pipeline (for comparison): [`scripts/prepare_windowed_parquet.py`](scripts/prepare_windowed_parquet.py)
