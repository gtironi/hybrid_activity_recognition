# Hybrid Activity Recognition with Deep Learning and Handcrafted Features

**Modular PyTorch framework for time-series activity classification combining deep learned representations with statistical features (TSFEL).**

This repository implements a hybrid architecture that fuses signal embeddings from deep encoders (CNN+LSTM, Transformer) with handcrafted time-series features, evaluated on the [AcTBeCalf](https://zenodo.org/records/13259482) cattle behavior dataset.

---

## Architecture Overview

The framework follows a four-component modular design:

```
x_signal (B, C, T)  →  [SignalEncoder]  ────┐
                                            ├─→ [Fusion] → [Head] → logits
x_features (B, K)   →  [TsfelBranch]   ────┘
```

**Three operational modes:**
- **`deep_only`**: Encoder → Head (TSFEL branch disabled)
- **`hybrid`**: Encoder + TSFEL → Fusion → Head (default)
- **`tsfel_only`**: TSFEL branch → Head (signal ignored)

**Three encoder families:**
- `cnn_lstm`: 2 Conv1D blocks + 2-layer BiLSTM
- `robust`: 3 Conv1D blocks + 1-layer BiLSTM (deeper CNN, Kaiming init)
- `patchtst`: Transformer with patch-based tokenization (HuggingFace implementation)

**Experimental grid:** 3 encoders × 2 modes = 6 deep learning experiments + 2 TSFEL-only baselines (Random Forest and TSFEL+MLP).

---

## Installation

```bash
git clone <repository-url>
cd hybrid_activity_recognition
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

**Requirements:**
- Python ≥3.10
- PyTorch ≥2.0 (see [pytorch.org](https://pytorch.org) for CUDA-specific builds)
- `transformers>=4.36.0` (PatchTST support)
- `tsfel`, `scikit-learn`, `pandas`, `pyarrow`

---

## Quick Start

### Default Flow (Recommended)

Run this sequence end-to-end:

```bash
# 0) Environment
source venv/bin/activate
export PYTHONPATH=src

# 1) Raw CSV -> train/test (80/20 stratified by behaviour)
python scripts/dataset_processing.py \
  --csv dataset/AcTBeCalf.csv \
  --out-dir dataset/processed \
  --split-by behavior \
  --behavior-column behaviour \
  --test-fraction 0.2 \
  --random-state 42

# 2) Window train (discover TSFEL top-K + save manifest)
python scripts/prepare_windowed_parquet.py \
  --input dataset/processed/AcTBeCalf/train.parquet \
  --output dataset/processed/AcTBeCalf/windowed_train.parquet \
  --feature-manifest-out dataset/processed/AcTBeCalf/tsfel_feature_manifest.json

# 3) Window test (apply same TSFEL columns from manifest)
python scripts/prepare_windowed_parquet.py \
  --input dataset/processed/AcTBeCalf/test.parquet \
  --output dataset/processed/AcTBeCalf/windowed_test.parquet \
  --feature-manifest-in dataset/processed/AcTBeCalf/tsfel_feature_manifest.json

# 4) Run full experimental grid
bash scripts/experiments/run_all.sh
```

### 1. Prepare Data

**Step 1a: Split raw CSV by behavior (stratified train/test split)**

```bash
python scripts/dataset_processing.py \
  --csv dataset/AcTBeCalf.csv \
  --out-dir dataset/processed
```

Output: `dataset/processed/AcTBeCalf/{train,test}.parquet` (long-form time series).
Default split: 80/20 stratified by `behaviour`. Use `--split-by subject` if you need the old cow-level split.

**Step 1b: Window + extract TSFEL features**

Training set (discovers top-K features, saves manifest):

```bash
python scripts/prepare_windowed_parquet.py \
  --input dataset/processed/AcTBeCalf/train.parquet \
  --output dataset/processed/AcTBeCalf/windowed_train.parquet \
  --feature-manifest-out dataset/processed/AcTBeCalf/tsfel_feature_manifest.json
```

Test set (applies same features from manifest):

```bash
python scripts/prepare_windowed_parquet.py \
  --input dataset/processed/AcTBeCalf/test.parquet \
  --output dataset/processed/AcTBeCalf/windowed_test.parquet \
  --feature-manifest-in dataset/processed/AcTBeCalf/tsfel_feature_manifest.json
```

### 2. Train a Model

**Example: Robust CNN+LSTM hybrid**

```bash
export PYTHONPATH=src
python -m hybrid_activity_recognition.main \
  --mode supervised \
  --model robust \
  --input_mode hybrid \
  --labeled_parquet_train dataset/processed/AcTBeCalf/windowed_train.parquet \
  --labeled_parquet_test dataset/processed/AcTBeCalf/windowed_test.parquet \
  --output_dir experiments/my_run \
  --epochs 50 \
  --batch_size 64 \
  --device cuda
```

**Example: PatchTST with pretraining**

```bash
# 1. Pretrain (masked auto-encoding)
PYTHONPATH=src python -m hybrid_activity_recognition.main \
  --mode pretrain \
  --pretrain_parquet dataset/processed/AcTBeCalf/windowed_train.parquet \
  --output_dir experiments/patchtst_pretrain \
  --pretrain_epochs 100 \
  --batch_size 64 \
  --device cuda

# 2. Supervised fine-tuning (hybrid mode)
PYTHONPATH=src python -m hybrid_activity_recognition.main \
  --mode supervised \
  --model patchtst \
  --input_mode hybrid \
  --patchtst_checkpoint experiments/patchtst_pretrain/best.pt \
  --labeled_parquet_train dataset/processed/AcTBeCalf/windowed_train.parquet \
  --labeled_parquet_test dataset/processed/AcTBeCalf/windowed_test.parquet \
  --output_dir experiments/patchtst_hybrid \
  --epochs 50 \
  --device cuda
```

### 3. Run Full Experimental Grid

**Smoke test (2 epochs, validates everything works):**

```bash
bash scripts/experiments/smoke_test.sh
```

**Full grid (50 epochs, all 8 experiments):**

```bash
bash scripts/experiments/run_all.sh
```

**For remote execution via SSH:**

```bash
# Option 1: nohup
nohup bash scripts/experiments/run_all.sh > logs/run_all.log 2>&1 &
disown

# Option 2: screen (recommended, allows reconnection)
screen -dmS experiments bash scripts/experiments/run_all.sh
# Reconnect: screen -r experiments

# Option 3: tmux
tmux new -d -s experiments 'bash scripts/experiments/run_all.sh'
# Reconnect: tmux attach -t experiments
```

---

## CLI Reference

### Main Entry Point

```bash
PYTHONPATH=src python -m hybrid_activity_recognition.main --help
```

**Key arguments:**

| Argument | Values | Description |
|----------|--------|-------------|
| `--mode` | `supervised`, `pretrain`, `finetune`, `test` | Training mode |
| `--model` | `cnn_lstm`, `robust`, `patchtst`, `tsfel_mlp` | Encoder family |
| `--input_mode` | `deep_only`, `hybrid`, `tsfel_only` | Architecture mode (default: `hybrid`) |
| `--labeled_parquet_train` | path | Windowed training parquet |
| `--labeled_parquet_test` | path | Windowed test parquet |
| `--checkpoint` | path | Resume from this checkpoint (optional) |
| `--patchtst_checkpoint` | path | Pretrained PatchTST encoder (required for `patchtst` model) |
| `--pretrain_parquet` | path | Windowed parquet for MAE pretraining (mode=`pretrain`) |
| `--output_dir` | path | Output directory for checkpoints and logs |
| `--epochs` | int | Training epochs (default: 50) |
| `--batch_size` | int | Batch size (default: 64) |
| `--device` | `cuda`, `cpu` | Device (default: `cuda`) |
| `--seed` | int | Random seed (default: 42) |

**PatchTST-specific:**
- `--patchtst_d_model`, `--patchtst_num_layers`, `--patchtst_num_heads`
- `--patchtst_patch_length`, `--patchtst_patch_stride`, `--patchtst_dropout`
- `--pretrain_epochs`, `--pretrain_lr`, `--pretrain_mask_ratio`

---

## Experimental Workflow

### Data Preprocessing

**Input:** Raw CSV with columns `dateTime`, `calfId`, `segId`, `accX`, `accY`, `accZ`, `behaviour`.

**Pipeline:**

1. **Train/test split** (default: stratified by behavior):
   ```bash
   python scripts/dataset_processing.py \
     --csv dataset/AcTBeCalf.csv \
     --out-dir dataset/processed
   ```

   To force split by subject:
   ```bash
   python scripts/dataset_processing.py \
     --csv dataset/AcTBeCalf.csv \
     --out-dir dataset/processed \
     --split-by subject \
     --subject-column calfId \
     --test-subjects 1329 1343 1353 1357 1372
   ```

2. **Windowing + TSFEL** (discover on train, apply on test):
   ```bash
   # Train: discover top-75 features
   python scripts/prepare_windowed_parquet.py \
     --input dataset/processed/AcTBeCalf/train.parquet \
     --output dataset/processed/AcTBeCalf/windowed_train.parquet \
     --feature-manifest-out dataset/processed/AcTBeCalf/tsfel_feature_manifest.json \
     --window-size 75 --overlap 0.5 --purity-threshold 0.9 --fs 25

   # Test: apply same features
   python scripts/prepare_windowed_parquet.py \
     --input dataset/processed/AcTBeCalf/test.parquet \
     --output dataset/processed/AcTBeCalf/windowed_test.parquet \
     --feature-manifest-in dataset/processed/AcTBeCalf/tsfel_feature_manifest.json
   ```

**Normalization:** Signal z-score and TSFEL StandardScaler are fitted **only** on training windows (no test leakage).

### Training Modes

**Supervised (from scratch):**

```bash
PYTHONPATH=src python -m hybrid_activity_recognition.main \
  --mode supervised \
  --model robust \
  --input_mode hybrid \
  --labeled_parquet_train dataset/processed/AcTBeCalf/windowed_train.parquet \
  --labeled_parquet_test dataset/processed/AcTBeCalf/windowed_test.parquet \
  --output_dir experiments/robust_hybrid_run1 \
  --epochs 50 --lr 1e-3 --device cuda
```

**Pretrain + Fine-tune (PatchTST):**

```bash
# 1. MAE pretraining (unlabeled data)
PYTHONPATH=src python -m hybrid_activity_recognition.main \
  --mode pretrain \
  --pretrain_parquet dataset/processed/AcTBeCalf/windowed_train.parquet \
  --output_dir experiments/patchtst_pretrain \
  --pretrain_epochs 100 --pretrain_lr 1e-3 --device cuda

# 2. Supervised fine-tuning
PYTHONPATH=src python -m hybrid_activity_recognition.main \
  --mode supervised \
  --model patchtst \
  --input_mode hybrid \
  --patchtst_checkpoint experiments/patchtst_pretrain/best.pt \
  --labeled_parquet_train dataset/processed/AcTBeCalf/windowed_train.parquet \
  --labeled_parquet_test dataset/processed/AcTBeCalf/windowed_test.parquet \
  --output_dir experiments/patchtst_hybrid \
  --epochs 50 --device cuda
```

**Resume interrupted training:**

```bash
PYTHONPATH=src python -m hybrid_activity_recognition.main \
  --mode supervised \
  --model robust \
  --input_mode hybrid \
  --checkpoint experiments/robust_hybrid_run1/checkpoint.pt \
  --labeled_parquet_train dataset/processed/AcTBeCalf/windowed_train.parquet \
  --labeled_parquet_test dataset/processed/AcTBeCalf/windowed_test.parquet \
  --output_dir experiments/robust_hybrid_run1 \
  --epochs 50 --device cuda
```

**Test only (evaluate a saved checkpoint):**

```bash
PYTHONPATH=src python -m hybrid_activity_recognition.main \
  --mode test \
  --model robust \
  --input_mode hybrid \
  --checkpoint experiments/robust_hybrid_run1/best.pt \
  --labeled_parquet_train dataset/processed/AcTBeCalf/windowed_train.parquet \
  --labeled_parquet_test dataset/processed/AcTBeCalf/windowed_test.parquet \
  --output_dir experiments/robust_hybrid_run1
```

---

## Automated Experiments

### Smoke Test (Quick Validation)

Runs all 7 experiments for **2 epochs** with small batch size to verify everything connects:

```bash
bash scripts/experiments/smoke_test.sh
```

**Expected output:** 7 directories in `experiments/` with `DONE` markers and `best.pt` checkpoints.

### Full Experimental Grid

Runs all experiments with default hyperparameters (150 epochs, batch 256):

```bash
bash scripts/experiments/run_all.sh
```

**Execution order:**
1. CNN+LSTM: `deep_only` + `hybrid`
2. Robust CNN+LSTM: `deep_only` + `hybrid`
3. PatchTST: pretrain → `deep_only` + `hybrid`
4. TSFEL-only baseline (Random Forest)

**Individual model scripts:**
- `bash scripts/experiments/run_cnn_lstm.sh`
- `bash scripts/experiments/run_robust.sh`
- `bash scripts/experiments/run_patchtst.sh`
- `bash scripts/experiments/run_tsfel_baseline.sh`

**Remote execution (SSH):**

```bash
# Using screen (recommended)
screen -dmS experiments bash scripts/experiments/run_all.sh
# Reconnect: screen -r experiments

# Using nohup
nohup bash scripts/experiments/run_all.sh > logs/run_all.log 2>&1 &
disown
```

**Resume behavior:**
- If `checkpoint.pt` exists in the run directory, training automatically resumes from the last epoch.
- If `DONE` marker exists, the experiment is skipped.
- To restart from scratch: delete the run directory or remove the `DONE` file.

---

## Checkpointing and Resume

Each training run creates a directory named with key hyperparameters:

```
experiments/{model}_{mode}_{dataset}_ep{epochs}_bs{batch}_lr{lr}_s{seed}/
```

**Files saved:**
- `checkpoint.pt`: Full state (model, optimizer, scheduler, counters) for resume
- `best.pt`: Best model by validation accuracy (model weights only)
- `train.log`: Complete training log (dual output: console + file)
- `DONE`: Marker indicating successful completion

**Resume is explicit:** Pass `--checkpoint path/to/checkpoint.pt` to resume. Without it, training starts from scratch.

---

## Testing

**Unit tests (dimension sanity checks):**

```bash
PYTHONPATH=src pytest tests/ -v
```

**Coverage:**
- `test_encoders.py`: Forward pass shape validation for all encoders
- `test_fusion.py`: Fusion, TSFEL branch, and head dimension checks
- `test_hybrid_model.py`: End-to-end forward + gradient flow for both modes

**Runtime:** <5 seconds, no GPU required, no real data needed.

---

## Baseline: TSFEL-only (Random Forest)

Standalone sklearn baseline for comparison (no deep learning):

```bash
PYTHONPATH=src python -m random_forest_baseline.tsfel_baseline \
  --train dataset/processed/AcTBeCalf/windowed_train.parquet \
  --test dataset/processed/AcTBeCalf/windowed_test.parquet \
  --output_dir experiments/tsfel_baseline \
  --k 50 \
  --n_estimators 200
```

**Method:** SelectKBest (f_classif) + RandomForestClassifier with balanced class weights.

---

## Customization

### Adding a New Encoder

1. Implement `SignalEncoder` in `src/hybrid_activity_recognition/models/modular/encoders.py`:
   ```python
   class MyEncoder(SignalEncoder):
       @property
       def output_dim(self) -> int:
           return self._dim

       def forward(self, x_signal: Tensor) -> Tensor:
           # x_signal: (B, 3, T)
           # return: (B, output_dim)
           ...
   ```

2. Register in `models/modular/__init__.py`:
   ```python
   _ENCODER_REGISTRY["my_encoder"] = MyEncoder
   ```

3. Use via CLI:
   ```bash
   --model my_encoder --input_mode hybrid
   ```

### Adapting to a New Dataset

1. Ensure raw CSV has columns: `datetime`, `subject`, `acc_x`, `acc_y`, `acc_z`, `label` (or use `--column-map` in `dataset_processing.py`).
2. Run the same 3-step pipeline (split → window → train).
3. Update `DATASET_ID` in `scripts/experiments/_common.sh` for proper run naming.

---

## Architecture Details

### Signal Encoders

| Encoder | Architecture | Output Dim | Notes |
|---------|--------------|------------|-------|
| `CNNLSTMEncoder` | 2 Conv1D (64→128) + BiLSTM (2 layers) | `2 × hidden_lstm` (default: 128) | Last timestep aggregation |
| `RobustCNNLSTMEncoder` | 3 Conv1D (64→128→256) + BiLSTM (1 layer) | `2 × hidden_lstm` (default: 256) | h_n concatenation, Kaiming init |
| `PatchTSTEncoder` | Patch tokenization + Transformer | `d_model` (default: 128) | HuggingFace wrapper, mean pooling |

### TSFEL Branch

- **`MLPTsfelBranch`**: Linear → BatchNorm → ReLU → Dropout
- Projects variable-length TSFEL features to fixed `hidden_dim` (default: encoder's `output_dim`)

### Fusion

- **`ConcatFusion`**: Simple concatenation (`output_dim = enc_dim + tsfel_dim`)
- Future work: Gated fusion, cross-attention

### Classification Head

- **`MLPHead`**: Linear → ReLU → Dropout → Linear (default: 256 hidden)
- **`LinearHead`**: Single linear layer (minimal parameters)

---

## Output and Metrics

**Per-epoch logging:**
- Training loss, training accuracy
- Validation loss, validation accuracy
- Learning rate (after scheduler step)

**Final test metrics:**
- Accuracy
- Macro F1
- Weighted F1

**Saved to:** `{output_dir}/train.log` and console.

**Checkpoints:**
- `best.pt`: Best validation accuracy (for final evaluation)
- `checkpoint.pt`: Latest epoch (for resume)

---

## Reproducibility

- **Seed control:** `--seed` sets `random`, `numpy`, and `torch` seeds.
- **Deterministic ops:** `CUBLAS_WORKSPACE_CONFIG=:4096:8` set by `utils/repro.py`.
- **Data splits:** Default is stratified 80/20 by behavior. Subject-based split is optional via `--split-by subject`.
- **Normalization:** Statistics computed only on training set, applied to val/test.
- **TSFEL features:** Manifest ensures test uses the same features as training.

---

## References

- **PatchTST:** Nie, Y. et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." *ICLR 2023*.
- **TSFEL:** Barandas, M. et al. (2020). "TSFEL: Time Series Feature Extraction Library." *SoftwareX*, 11, 100456.

---

## Troubleshooting

**`FileNotFoundError: windowed_train.parquet`**
→ Run `prepare_windowed_parquet.py` on `train.parquet` first (see Quick Start step 1b).

**`Manifest not found`**
→ Run the training set windowing with `--feature-manifest-out` before processing the test set.

**`unrecognized arguments` with checkpoint paths**
→ Ensure paths with spaces are quoted: `--checkpoint "path/with spaces/checkpoint.pt"`.

**CUDA driver warnings**
→ The code falls back to CPU automatically. To suppress warnings: `export DEVICE=cpu`.

**TSFEL `RuntimeWarning: catastrophic cancellation`**
→ Expected for near-constant signal windows; does not affect results. Suppress with `PYTHONWARNINGS=ignore::RuntimeWarning`.

---

## Contact

[Gustavo Tironi](https://github.com/gtironi) · [João Gabriel Machado](https://github.com/jgabrielsg)
