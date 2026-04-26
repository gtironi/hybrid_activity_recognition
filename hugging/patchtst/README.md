# Hugging Face PatchTST (classification debug)

Small pipeline: convert several datasets to a **shared CSV layout**, then train `transformers.PatchTSTForClassification` with a plain PyTorch loop (no `Trainer`), save metrics and optional **t-SNE** via `extras.py`.

## Layout

- **Converters**: `convert/convert_*.py` → write `standardized/<name>/` with `meta.json`, `train.csv`, `val.csv`, `test.csv`.
- **Train**: `train_classification_debug.py` → writes under `runs/<timestamp>_<preset>_seed<seed>/` (`config.json`, `metrics.json`, `confusion_matrix.png`, `model.pt`).
- **Extras**: `extras.py tsne` for 2-D embedding of pooled encoder features after a classification checkpoint.

Generated data and runs are gitignored (`standardized/`, `runs/`).

## Canonical CSV format

- `meta.json`: `context_length` (T), `num_channels` (C), `label2id`, `id2label`, `feature_order: "time_major"`.
- Each split CSV: column `label` (int), then `x_000` … `x_{T*C-1}` flattening each window `(T, C)` in **time-major** order (matches `past_values` `(B, T, C)`).

## Convert datasets

From repository root (`hybrid_activity_recognition/`):

```bash
# UCI HAR — uses **Inertial Signals** (128×9). Do not use `X_train.txt` (561 engineered features).
python -m hugging.patchtst.convert.convert_har_uci \
  --har_dir "data/HAR UCI"

# Dog IMU (pick one CSV). Sliding windows; split by DogID (no subject leakage).
python -m hugging.patchtst.convert.convert_dog \
  --input data/dog/dog_w10.csv \
  --window 128 --stride 64

# AcTBeCalf (large). Optional smoke test:
python -m hugging.patchtst.convert.convert_actbecalf \
  --max_rows 500000

# ETTm1 — no behaviour labels; default proxy task is **hour-of-day** (24 classes) at window end.
python -m hugging.patchtst.convert.convert_ettm1 \
  --window 96 --stride 48
```

Each script prints the suggested training command.

## Train

```bash
python -m hugging.patchtst.train_classification_debug --preset har --epochs 5 --device cuda
python -m hugging.patchtst.train_classification_debug --preset dog_w10 --epochs 5
python -m hugging.patchtst.train_classification_debug --preset actbecalf --epochs 5
python -m hugging.patchtst.train_classification_debug --preset ettm1 --epochs 5
```

Use `--data_dir /path/to/standardized/foo` instead of `--preset` for custom outputs.

**API notes** (Transformers 5.5.x): `PatchTSTConfig` uses `patch_stride` (not `stride`). The forward pass passes labels as `target_values=`.

## t-SNE (classification checkpoint)

After a run produces `model.pt`:

```bash
python -m hugging.patchtst.extras tsne \
  --model_pt hugging/patchtst/runs/<run_id>/model.pt \
  --preset har \
  --split test \
  --pooling mean
```

`encoder_only` / **pretrain → probe** is not implemented here; see `extras.pretrain_then_linear_probe_stub` and `load_tfc_pt_windows_stub`.

## Presets

| `--preset`   | Standardized directory        |
|-------------|-------------------------------|
| `dog_w10`   | `standardized/dog_w10`        |
| `dog_w50`   | `standardized/dog_w50`        |
| `dog_w100`  | `standardized/dog_w100`       |
| `dog_raw`   | `standardized/dog_raw`        |
| `actbecalf` | `standardized/actbecalf`      |
| `har`       | `standardized/har_uci`        |
| `ettm1`     | `standardized/ettm1_hour`     |

Run the matching `convert_*` script once before using a preset.
