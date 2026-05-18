"""Roda os 3 modelos em 2 regimes (Sup + FixMatch) e salva resultados em results_notebook/.

Uso:
    python scripts/notebook_runner/run.py
    python scripts/notebook_runner/run.py --stage1 100 --finetune 30 --fixmatch 40
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from data import prepare_dataloaders, prepare_unlabeled_dataloader
from models import TsfelOnlyModel, RobustHybridModel, HybridCNNLSTM
from train import train_stage1, train_stage2, train_fixmatch, evaluate

TRAIN_PARQUET = ROOT / "dataset/processed/AcTBeCalf/windowed_train.parquet"
TEST_PARQUET = ROOT / "dataset/processed/AcTBeCalf/windowed_test.parquet"
UNLAB_PARQUET = ROOT / "dataset/processed/AcTBeCalf/Windowed_Time_Adj.parquet"

RESULTS_DIR = ROOT / "results_notebook"
CKPT_DIR = RESULTS_DIR / "checkpoints"

MODELS = {
    "tsfel_only": TsfelOnlyModel,
    "robust": RobustHybridModel,
    "hybrid_cnn_lstm": HybridCNNLSTM,
}


def save_eval(name, regime, eval_dict, class_names):
    out_dir = RESULTS_DIR / f"{name}__{regime}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": name,
        "regime": regime,
        "accuracy": float(eval_dict["accuracy"]),
        "f1_macro": float(eval_dict["f1_macro"]),
        "f1_weighted": float(eval_dict["f1_weighted"]),
        "loss": float(eval_dict["loss"]),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(eval_dict["classification_report"])
    np.save(out_dir / "y_true.npy", eval_dict["y_true"])
    np.save(out_dir / "y_pred.npy", eval_dict["y_pred"])
    cm = eval_dict["confusion_matrix"]
    np.save(out_dir / "confusion_matrix.npy", cm)
    pd.DataFrame(cm, index=list(class_names), columns=list(class_names)) \
        .to_csv(out_dir / "confusion_matrix.csv")

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    cm_norm = np.nan_to_num(cm_norm)
    pd.DataFrame(cm_norm, index=list(class_names), columns=list(class_names)) \
        .to_csv(out_dir / "confusion_matrix_normalized.csv")

    title = (f"{name} | {regime} (Acc: {metrics['accuracy']:.2%}, "
             f"F1-macro: {metrics['f1_macro']:.3f})")
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=list(class_names), yticklabels=list(class_names),
                cbar_kws={"label": "Proporcao"})
    plt.title(title, fontsize=14)
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_normalized.png", dpi=120)
    plt.close()

    print(f"   Resultados salvos em: {out_dir}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1", type=int, default=100)
    p.add_argument("--finetune", type=int, default=30)
    p.add_argument("--fixmatch", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                   help=f"Quais modelos rodar: {list(MODELS.keys())}")
    p.add_argument("--skip-fixmatch", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    dl_train, dl_val, dl_test, class_names, n_feats = prepare_dataloaders(
        str(TRAIN_PARQUET), str(TEST_PARQUET), batch_size=args.batch_size,
    )
    num_classes = len(class_names)

    dl_unlab = None
    if not args.skip_fixmatch:
        if not UNLAB_PARQUET.exists():
            print(f"AVISO: unlabeled nao encontrado ({UNLAB_PARQUET}). Pulando FixMatch.")
            args.skip_fixmatch = True
        else:
            dl_unlab = prepare_unlabeled_dataloader(str(UNLAB_PARQUET), batch_size=args.batch_size)

    all_rows = []
    for name in args.models:
        if name not in MODELS:
            print(f"Modelo desconhecido: {name}. Pulando.")
            continue
        model_cls = MODELS[name]
        print("\n" + "#" * 70)
        print(f"# MODELO: {name} ({model_cls.__name__})")
        print("#" * 70)

        stage1_ckpt = CKPT_DIR / f"{name}_stage1.pth"
        finetune_ckpt = CKPT_DIR / f"{name}_finetune.pth"
        fixmatch_ckpt = CKPT_DIR / f"{name}_fixmatch.pth"

        for p_ in [stage1_ckpt, finetune_ckpt, fixmatch_ckpt]:
            if p_.exists():
                p_.unlink()

        train_stage1(model_cls, dl_train, dl_val, num_classes, n_feats, device,
                     epochs=args.stage1, checkpoint_path=str(stage1_ckpt))
        train_stage2(model_cls, dl_train, dl_val, num_classes, n_feats, device,
                     epochs=args.finetune,
                     checkpoint_path=str(stage1_ckpt), finetune_path=str(finetune_ckpt))

        res_sup = evaluate(model_cls, dl_test, num_classes, n_feats, device,
                           checkpoint_path=str(finetune_ckpt), class_names=class_names)
        if res_sup is not None:
            all_rows.append(save_eval(name, "supervised", res_sup, class_names))

        if not args.skip_fixmatch and dl_unlab is not None:
            train_fixmatch(model_cls, dl_train, dl_unlab, dl_val, num_classes, n_feats, device,
                           epochs=args.fixmatch,
                           checkpoint_path=str(finetune_ckpt), save_path=str(fixmatch_ckpt),
                           threshold=0.72, lambda_u=1.0)
            res_fm = evaluate(model_cls, dl_test, num_classes, n_feats, device,
                              checkpoint_path=str(fixmatch_ckpt), class_names=class_names)
            if res_fm is not None:
                all_rows.append(save_eval(name, "fixmatch", res_fm, class_names))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if all_rows:
        df = pd.DataFrame(all_rows)
        df["accuracy"] = (df["accuracy"] * 100).round(2)
        df["f1_macro"] = df["f1_macro"].round(4)
        df["f1_weighted"] = df["f1_weighted"].round(4)
        df.to_csv(RESULTS_DIR / "summary.csv", index=False)
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(df.to_string(index=False))
        print(f"\nSalvo em: {RESULTS_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
