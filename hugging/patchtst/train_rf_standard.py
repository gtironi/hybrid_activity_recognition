#!/usr/bin/env python3
"""Random Forest on standardized splits: optional CV on train+val; fit on train+val; test eval."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from hugging.patchtst.io_standard import load_csv_tensors, load_meta, package_root

PRESET_DIRS = {
    "dog_w10": "standardized/dog_w10",
    "dog_w50": "standardized/dog_w50",
    "dog_w100": "standardized/dog_w100",
    "dog_raw": "standardized/dog_raw",
    "actbecalf": "standardized/actbecalf",
    "har": "standardized/har_uci",
    "ettm1": "standardized/ettm1_hour",
}


def _make_clf(args: argparse.Namespace) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        class_weight="balanced_subsample",
    )


def _resolve_data_dir(preset: str | None, data_dir: Path | None) -> Path:
    if data_dir is not None:
        return Path(data_dir).resolve()
    if not preset:
        raise ValueError("Provide --data_dir or --preset")
    rel = PRESET_DIRS.get(preset)
    if not rel:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESET_DIRS)}")
    return (package_root() / rel).resolve()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", type=str, default="dog_raw")
    p.add_argument("--data_dir", type=Path, default=None, help="Override preset directory")
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Stratified K-fold CV on train+val (0 = skip cross-validation)",
    )
    p.add_argument(
        "--confusion_matrix_png",
        type=Path,
        default=Path("confusion_matrix.png"),
        help="Path for confusion matrix figure (counts; % of row)",
    )
    args = p.parse_args()

    data_dir = _resolve_data_dir(args.preset if args.data_dir is None else None, args.data_dir)
    meta = load_meta(data_dir)

    x_tr, y_tr = load_csv_tensors(data_dir / "train.csv", meta)
    x_va, y_va = load_csv_tensors(data_dir / "val.csv", meta)
    x_te, y_te = load_csv_tensors(data_dir / "test.csv", meta)

    x_fit = np.vstack([x_tr, x_va])
    print("x_fit.shape", x_fit.shape)
    y_fit = np.concatenate([y_tr, y_va])

    print(f"data_dir={data_dir}")
    print(f"train+val rows={len(y_fit)}, test rows={len(y_te)}")

    if args.cv >= 2:
        skf = StratifiedKFold(
            n_splits=args.cv, shuffle=True, random_state=args.random_state
        )
        scoring = {
            "accuracy": "accuracy",
            "f1_weighted": "f1_weighted",
            "f1_macro": "f1_macro",
        }
        cv_res = cross_validate(
            _make_clf(args),
            x_fit,
            y_fit,
            cv=skf,
            scoring=scoring,
            n_jobs=1,
        )
        for name in scoring:
            vals = cv_res[f"test_{name}"]
            print(
                f"CV {args.cv}-fold {name}: "
                f"{vals.mean():.4f} ± {vals.std():.4f}"
            )

    clf = _make_clf(args)
    clf.fit(x_fit, y_fit)

    y_pred = clf.predict(x_te)
    acc = accuracy_score(y_te, y_pred)
    f1w = f1_score(y_te, y_pred, average="weighted")
    f1m = f1_score(y_te, y_pred, average="macro")

    print(
        f"Held-out test: accuracy={acc:.4f}  "
        f"f1_weighted={f1w:.4f}  f1_macro={f1m:.4f}"
    )
    n_cls = int(meta["num_classes"])
    labels = list(range(n_cls))
    id2 = meta.get("id2label", {})
    names = [id2[str(i)] for i in labels]
    cm = confusion_matrix(y_te, y_pred, labels=labels)

    out_png = Path(args.confusion_matrix_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        cm_np = cm.astype(float)
        row_sum = cm_np.sum(axis=1, keepdims=True)
        row_sum = np.maximum(row_sum, 1.0)
        cm_row_pct = cm_np / row_sum * 100.0

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(cm_np, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set_title("Confusion matrix (counts; % of row = true class)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)

        txt_color_thresh = cm_np.max() / 2.0 if cm_np.size and cm_np.max() > 0 else 0
        for i in range(n_cls):
            for j in range(n_cls):
                cnt = int(cm_np[i, j])
                pct = cm_row_pct[i, j]
                ax.text(
                    j,
                    i,
                    f"{cnt}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if cm_np[i, j] > txt_color_thresh else "black",
                )

        plt.tight_layout()
        plt.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved confusion matrix figure to {out_png.resolve()}")
    except ImportError:
        print(
            "matplotlib is not installed; skipped confusion matrix PNG "
            "(pip install matplotlib)."
        )

    print(
        classification_report(
            y_te, y_pred, labels=labels, target_names=names, zero_division=0
        )
    )


if __name__ == "__main__":
    main()
