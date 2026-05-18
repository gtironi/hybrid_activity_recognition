"""Regenera confusion_matrix_normalized.{csv,png} a partir dos .npy ja salvos.

Util pra resultados ja existentes em results_notebook/{modelo}__{regime}/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results_notebook"


def replot(exp_dir: Path):
    cm_path = exp_dir / "confusion_matrix.csv"
    if not cm_path.exists():
        print(f"skip (sem confusion_matrix.csv): {exp_dir.name}")
        return
    cm_df = pd.read_csv(cm_path, index_col=0)
    class_names = list(cm_df.columns)
    cm = cm_df.values.astype(float)
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
    cm_norm = np.nan_to_num(cm_norm)

    pd.DataFrame(cm_norm, index=class_names, columns=class_names) \
        .to_csv(exp_dir / "confusion_matrix_normalized.csv")

    metrics = {}
    metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    acc = metrics.get("accuracy", float("nan"))
    f1m = metrics.get("f1_macro", float("nan"))
    name = metrics.get("model", exp_dir.name)
    regime = metrics.get("regime", "")

    title = f"{name} | {regime} (Acc: {acc:.2%}, F1-macro: {f1m:.3f})"
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={"label": "Proporcao"})
    plt.title(title, fontsize=14)
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(exp_dir / "confusion_matrix_normalized.png", dpi=120)
    plt.close()
    print(f"OK: {exp_dir.name}")


def main():
    if not RESULTS_DIR.exists():
        print(f"Nao existe: {RESULTS_DIR}")
        return
    for sub in sorted(RESULTS_DIR.iterdir()):
        if sub.is_dir() and sub.name != "checkpoints":
            replot(sub)


if __name__ == "__main__":
    main()
