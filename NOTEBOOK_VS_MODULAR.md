# Notebook vs Modular Implementation — Gap Analysis

## TL;DR

The notebook beats the modular impl for four reasons, roughly in order of impact:

1. `MLPTsfelBranch` is an identity function — the 3-layer MLP inside `TsfelOnlyModel` never runs
2. Two-stage training: balanced loss → plain loss fine-tune
3. Early-stopping patience is 10 (modular) vs 25 (notebook)
4. No Kaiming init on TSFEL/head layers in modular code

---

## 1. Model Architecture — Critical

### Notebook `TsfelOnlyModel`
```
Input (B, 75)
→ Linear(75, 256) → BatchNorm1d(256) → ReLU → Dropout(0.4)
→ Linear(256, 128) → BatchNorm1d(128) → ReLU → Dropout(0.4)
→ Linear(128, 64)  → BatchNorm1d(64)  → ReLU → Dropout(0.4)
→ Linear(64, num_classes)
```
Kaiming init on every Linear + BN.

### Modular `tsfel_only` mode
`MLPTsfelBranch` (in `tsfel_branches.py`) is an **identity**: `forward` returns `x_features` unchanged.
`MLPHead` then does:
```
Input (B, 75)
→ Linear(75, 256) → ReLU → Dropout(0.4)
→ Linear(256, num_classes)
```

**Gaps:**
| | Notebook | Modular |
|---|---|---|
| Hidden layers | 3 (256→128→64) | 1 (256) |
| BatchNorm | after every linear | none |
| Dropout | 0.4 after each block | 0.4 once |
| Output dim before classifier | 64 | 256 |
| Kaiming init | yes, all layers | no (only RobustCNNLSTMEncoder) |

**Fix:** Replace `MLPTsfelBranch` identity with real 3-layer MLP + BN. Add Kaiming init to heads.

---

## 2. Two-Stage Training — Critical

The notebook runs two sequential passes on the **same train/val loaders**.

### Stage 1 — `train_robust_model_advanced`
| Param | Notebook | Modular `train_supervised` |
|---|---|---|
| Loss | `CrossEntropyLoss(weight=class_weights)` | same ✓ |
| LR | `0.001` | `1e-3` ✓ |
| Weight decay | `5e-4` | `5e-4` ✓ |
| Scheduler | `ReduceLROnPlateau(patience=5, factor=0.3)` | same ✓ |
| Early stopping | patience = **25** | patience = **10** ✗ |
| Max epochs | **500** | 1000 default ✗ (too many — but ES dominates) |
| Grad clip | 1.0 | 1.0 ✓ |

### Stage 2 — `train_finetune_stage2`
| Param | Notebook | Modular `finetune` |
|---|---|---|
| Loss | `CrossEntropyLoss()` — **no class weights** | same ✓ |
| LR | `0.0001` | `1e-4` ✓ |
| Weight decay | `1e-4` | `1e-4` ✓ |
| Scheduler | `ReduceLROnPlateau(patience=3, factor=0.5)` | same ✓ |
| Early stopping | **none** | none ✓ |
| Max epochs | **50** | 1000 default ✗ — must pass `--epochs 50` |
| Grad clip | 1.0 | 1.0 ✓ |

**Why two stages work:** Stage 1 forces the model to learn rare classes (balanced loss lifts their gradient contribution). Stage 2 re-calibrates decision boundaries to match real class frequencies — improves accuracy on majority classes without forgetting minorities.

**Current modular gap:** LR / WD / scheduler are identical. The only real training-hyperparams gaps are:
- ES patience 10 vs 25 (Stage 1 kills training too early)
- No CLI guard preventing 1000 finetune epochs

---

## 3. Early Stopping Patience

| | Notebook Stage 1 |Modular `train_supervised` |
|---|---|---|
| Patience | **25** | **10** |

With patience=10, the model stops before LR scheduling cycles have time to recover. This alone can explain premature convergence.

**Fix:** change default in `trainer.py:51` from `10` → `25`.

---

## 4. Weight Initialization

Notebook calls `self.apply(self._init_weights)` on every model:
- `nn.Linear`: Kaiming normal
- `nn.Conv1d`: Kaiming normal
- `nn.BatchNorm1d`: weight=1, bias=0

Modular: only `RobustCNNLSTMEncoder._init_weights` does this. `MLPHead` and `MLPTsfelBranch` use PyTorch defaults.

**Fix:** Add `_init_weights` to `MLPHead` and the new real `MLPTsfelBranch`.

---

## 5. NaN Guards

Notebook checks every batch:
```python
if torch.isnan(x_sig).any() or torch.isnan(x_feat).any():
    return model
if torch.isnan(loss):
    return model
```
Modular: none. Not critical for correctness but helps diagnose exploding gradients early.

---

## 6. FixMatch Semi-Supervised (Stage 3)

Notebook has a third optional stage:
- Uses `windowed_train.parquet` as unlabeled pool
- Trains `TsfelOnlyModel` with pseudo-labels, threshold=0.90, lambda_u=1.0
- Loads from Stage 2 checkpoint, 200 epochs

Modular: not implemented. Optional — skip unless replicating exact notebook numbers.

---

## 7. Device Detection

| | Notebook | Modular `main.py` |
|---|---|---|
| MPS (Mac GPU) | `mps if available else cpu` | not detected — falls back to CPU |

**Fix:**
```python
if args.device == "cuda" and torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

---

## Summary: What to Change

Priority order:

| # | Change | File | Impact |
|---|---|---|---|
| 1 | Replace `MLPTsfelBranch` identity with 3-layer MLP + BN (256→128→64, Dropout 0.4) | `tsfel_branches.py` | **High** |
| 2 | Add Kaiming init to `MLPHead` and new `MLPTsfelBranch` | `heads.py`, `tsfel_branches.py` | **High** |
| 3 | Increase default `early_stopping_patience` from 10 → 25 | `trainer.py:51` | **Medium** |
| 4 | Cap finetune epochs default at 50 or document the required CLI flag | `main.py` / docs | Medium |
| 5 | Fix MPS device detection | `main.py` | Low (local only) |
| 6 | Add NaN guards | `trainer.py` | Low |
| 7 | Implement FixMatch | new file | Optional |

## LR / WD / Scheduler — Already Identical ✓

No changes needed here:
- Stage 1: LR=1e-3, WD=5e-4, ReduceLROnPlateau(patience=5, factor=0.3)
- Stage 2: LR=1e-4, WD=1e-4, ReduceLROnPlateau(patience=3, factor=0.5)
