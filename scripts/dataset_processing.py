#!/usr/bin/env python3
"""
CSV → train/test Parquet + split_report.json.

- --split-by behavior: sujeitos disjuntos; teste escolhido com genSplit (ver scripts/genSplit.py).
- --split-by subject: sujeitos de teste fixos (--test-subjects).
- Após o split, remove comportamentos com menos de --min-train-samples-per-behavior amostras
  no treino (e qualquer comportamento que só exista no teste) de **treino e teste** para alinhar
  labels com o que o modelo pode aprender.
- Depois, remove do teste linhas cujo --behavior-column não aparece no treino; o relatório JSON
  inclui quantas linhas e quais comportamentos foram removidos.

Saída: {out_dir}/{csv_stem}/train.parquet, test.parquet, split_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import genSplit

DEFAULT_TEST_SUBJECTS = (1329, 1343, 1353, 1357, 1372)
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_BEHAVIOR_COLUMN = "behaviour"
DEFAULT_MIN_TRAIN_SAMPLES_PER_BEHAVIOR = 500


def _coerce_subject_values(series: pd.Series, raw: list[str]) -> set:
    if pd.api.types.is_integer_dtype(series):
        return {int(x) for x in raw}
    if pd.api.types.is_float_dtype(series):
        return {float(x) for x in raw}
    return set(raw)


def _require_columns(df: pd.DataFrame, *names: str) -> None:
    missing = [c for c in names if c not in df.columns]
    if missing:
        raise SystemExit(f"Colunas em falta: {missing}. Ex.: {list(df.columns)[:30]}...")


def _subject_behavior_wide(df: pd.DataFrame, subject_col: str, behavior_col: str) -> pd.DataFrame:
    """Matriz larga esperada por genSplit: subject_id + uma coluna por classe (contagens)."""
    g = df.groupby([subject_col, behavior_col], sort=False).size().rename("n").reset_index()
    wide = g.pivot(index=subject_col, columns=behavior_col, values="n").fillna(0).astype(np.int64)
    return wide.reset_index().rename(columns={subject_col: "subject_id"})


def _label_distribution(series: pd.Series) -> dict:
    vc = series.value_counts(dropna=False)
    total = int(vc.sum())
    counts = {str(k): int(v) for k, v in vc.items()}
    prop = {k: (counts[k] / total if total else 0.0) for k in counts}
    return {"counts": counts, "proportions": prop}


def filter_test_behaviors_to_train(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    behavior_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Mantém no teste apenas linhas cujo comportamento (como string) existe no treino.
    Retorna metadados sobre linhas removidas para o split_report.json.
    """
    _require_columns(train, behavior_column)
    _require_columns(test, behavior_column)
    known = set(train[behavior_column].astype(str).unique())
    beh_test = test[behavior_column].astype(str)
    in_train = beh_test.isin(known)
    removed = test.loc[~in_train]
    n_removed = int(len(removed))
    removed_counts: dict[str, int] = {}
    if n_removed:
        vc = removed[behavior_column].astype(str).value_counts()
        removed_counts = {str(k): int(v) for k, v in vc.items()}
    behaviors_test_not_train = sorted(set(beh_test.unique()) - known, key=str)
    test_out = test.loc[in_train].reset_index(drop=True)
    meta = {
        "behavior_column": behavior_column,
        "rows_removed": n_removed,
        "removed_row_counts_by_behavior": removed_counts,
        "behaviors_present_in_test_but_not_in_train": behaviors_test_not_train,
    }
    if n_removed:
        print(
            f"[dataset_processing] Teste: removidas {n_removed} linhas com "
            f"{behavior_column!r} fora do treino ({len(known)} comportamentos no treino)."
        )
    if test_out.empty:
        raise SystemExit(
            "Conjunto de teste ficou vazio após alinhar comportamentos ao treino. "
            "Verifique --behavior-column e a distribuição por sujeito."
        )
    return train, test_out, meta


def filter_behaviors_below_min_train_count(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    behavior_column: str,
    min_train_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Remove from both train and test every behavior that has fewer than ``min_train_samples``
    rows in train, and any behavior that appears only in test (train count 0).
    """
    _require_columns(train, behavior_column)
    _require_columns(test, behavior_column)
    tr = train[behavior_column].astype(str)
    te = test[behavior_column].astype(str)
    counts = tr.value_counts()
    train_classes = set(tr.unique())
    low_train = set(counts[counts < min_train_samples].index.astype(str))
    only_in_test = set(te.unique()) - train_classes
    to_drop = low_train | only_in_test
    if not to_drop:
        return train, test, {
            "behavior_column": behavior_column,
            "min_train_samples": min_train_samples,
            "behaviors_removed": [],
            "train_rows_removed": 0,
            "test_rows_removed": 0,
        }

    removed_train = int(tr.isin(to_drop).sum())
    removed_test = int(te.isin(to_drop).sum())
    train_out = train.loc[~tr.isin(to_drop)].reset_index(drop=True)
    test_out = test.loc[~te.isin(to_drop)].reset_index(drop=True)
    meta = {
        "behavior_column": behavior_column,
        "min_train_samples": min_train_samples,
        "behaviors_removed": sorted(to_drop, key=str),
        "train_rows_removed": removed_train,
        "test_rows_removed": removed_test,
        "train_sample_count_per_removed_behavior": {
            b: int(tr.eq(b).sum()) for b in sorted(to_drop, key=str)
        },
    }
    print(
        f"[dataset_processing] Comportamentos excluídos (<{min_train_samples} amostras no treino "
        f"ou só no teste): {meta['behaviors_removed']!r} "
        f"(−{removed_train} treino, −{removed_test} teste)"
    )
    if train_out.empty:
        raise SystemExit(
            "Conjunto de treino ficou vazio após --min-train-samples-per-behavior. "
            "Reduza o limiar ou verifique os dados."
        )
    if test_out.empty:
        raise SystemExit(
            "Conjunto de teste ficou vazio após filtrar comportamentos raros no treino. "
            "Reduza --min-train-samples-per-behavior ou verifique o split."
        )
    return train_out, test_out, meta


def split_subject_list(
    df: pd.DataFrame, *, subject_column: str, test_subjects: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _require_columns(df, subject_column)
    test_ids = _coerce_subject_values(df[subject_column], test_subjects)
    mask = df[subject_column].isin(test_ids)
    train, test = df.loc[~mask].reset_index(drop=True), df.loc[mask].reset_index(drop=True)
    meta = {
        "name": "subject_fixed_list",
        "subject_column": subject_column,
        "test_subject_ids": sorted(test_ids, key=str),
    }
    return train, test, meta


def split_behavior_gen_split(
    df: pd.DataFrame,
    *,
    subject_column: str,
    behavior_column: str,
    test_fraction: float,
    max_combinations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _require_columns(df, subject_column, behavior_column)
    if not 0 < test_fraction < 1:
        raise SystemExit("--test-fraction deve estar entre 0 e 1.")

    train_pct = round(100.0 * (1.0 - test_fraction))
    test_pct = round(100.0 * test_fraction)
    if train_pct + test_pct != 100:
        raise SystemExit(
            f"Escolha --test-fraction tal que (1-f)×100 e f×100 sejam inteiros que somem 100; "
            f"obtido train={train_pct}% test={test_pct}%."
        )

    wide = _subject_behavior_wide(df, subject_column, behavior_column)
    subjects = sorted(wide["subject_id"].unique().tolist())
    n_sub = len(subjects)
    if n_sub < 2:
        raise SystemExit("São necessários pelo menos 2 sujeitos.")

    n_tr, n_val, n_te = genSplit.calc_split_subject_amounts(
        n_sub, {"train": float(train_pct), "validation": 0.0, "test": float(test_pct)}
    )
    if n_val != 0 or n_te < 1 or n_tr < 1:
        raise SystemExit(f"Contagem inválida de sujeitos: train={n_tr} val={n_val} test={n_te} (n={n_sub}).")

    n_comb = math.comb(n_sub, n_te)
    if n_comb > max_combinations:
        raise SystemExit(
            f"C({n_sub},{n_te}) = {n_comb:,} combinações (> --max-combinations={max_combinations})."
        )

    ratio = test_fraction / (1.0 - test_fraction)
    test_tuple = genSplit.find_optimal_calf_combinations_for_split(
        tuple(subjects), n_te, wide, ratio, cv=1
    )
    if test_tuple is None:
        raise SystemExit(
            "genSplit não encontrou combinação válida (todas as classes devem permanecer no treino). "
            "Ajuste --test-fraction ou verifique classes raras."
        )

    test_ids = set(test_tuple)
    mask = df[subject_column].isin(test_ids)
    train, test = df.loc[~mask].reset_index(drop=True), df.loc[mask].reset_index(drop=True)

    meta = {
        "split": "behavior_subject_genSplit",
        "subject_column": subject_column,
        "behavior_column": behavior_column,
        "test_fraction_subjects": test_pct / 100.0,
        "split_ratio_target": ratio,
        "n_subjects_total": n_sub,
        "n_subjects_train": n_sub - len(test_ids),
        "n_subjects_test": len(test_ids),
        "test_subject_ids": sorted(test_ids, key=str),
    }
    return train, test, meta


def build_split_report(
    df_full: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    subject_column: str,
    behavior_column: str,
    method: dict,
    test_label_alignment: dict | None = None,
) -> dict:
    _require_columns(train, subject_column)
    _require_columns(test, subject_column)

    def _subject_ids(frame: pd.DataFrame) -> list:
        return sorted(frame[subject_column].dropna().unique().tolist(), key=str)

    report: dict = {
        "schema_version": 2,
        "samples": {"total": len(df_full), "train": len(train), "test": len(test)},
        "subjects": {
            "column": subject_column,
            "train_ids": _subject_ids(train),
            "test_ids": _subject_ids(test),
            "n_subjects_train": int(train[subject_column].nunique()),
            "n_subjects_test": int(test[subject_column].nunique()),
        },
        "behavior_distribution": None,
        "method": method,
        "test_label_alignment": test_label_alignment,
    }

    if behavior_column in train.columns and behavior_column in test.columns:
        report["behavior_distribution"] = {
            "column": behavior_column,
            "train": _label_distribution(train[behavior_column]),
            "test": _label_distribution(test[behavior_column]),
        }
    elif behavior_column and behavior_column not in df_full.columns:
        warnings.warn(
            f"Coluna {behavior_column!r} ausente; behavior_distribution fica null.",
            RuntimeWarning,
            stacklevel=2,
        )

    return report


def main() -> None:
    p = argparse.ArgumentParser(description="CSV train/test → Parquet + split_report.json")
    p.add_argument("--csv", type=Path, default=Path("dataset/AcTBeCalf.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("dataset/processed"))
    p.add_argument("--subject-column", default="calfId")
    p.add_argument("--test-subjects", type=str, nargs="*", default=[str(x) for x in DEFAULT_TEST_SUBJECTS])
    p.add_argument(
        "--split-by",
        choices=("behavior", "subject"),
        default="behavior",
        help="behavior = genSplit por sujeito; subject = --test-subjects fixos",
    )
    p.add_argument("--max-combinations", type=int, default=2_000_000)
    p.add_argument("--behavior-column", default=DEFAULT_BEHAVIOR_COLUMN)
    p.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    p.add_argument(
        "--min-train-samples-per-behavior",
        type=int,
        default=DEFAULT_MIN_TRAIN_SAMPLES_PER_BEHAVIOR,
        help="Remove behaviors with fewer than this many train rows from both train and test (default: 500).",
    )
    args = p.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"Arquivo não encontrado: {args.csv}")

    out = args.out_dir / args.csv.stem
    out.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lendo {args.csv} ...")
    df = pd.read_csv(args.csv)

    if args.split_by == "subject":
        train, test, method = split_subject_list(
            df, subject_column=args.subject_column, test_subjects=args.test_subjects
        )
        tid = method["test_subject_ids"]
        summary = f"subject (lista fixa) | col={args.subject_column!r} | test={tid!r}"
    else:
        train, test, method = split_behavior_gen_split(
            df,
            subject_column=args.subject_column,
            behavior_column=args.behavior_column,
            test_fraction=args.test_fraction,
            max_combinations=args.max_combinations,
        )
        summary = (
            f"behavior + genSplit | subject={args.subject_column!r} "
            f"| behavior={args.behavior_column!r} | test_fraction={args.test_fraction:.2f}"
        )

    train, test, insufficient_train_meta = filter_behaviors_below_min_train_count(
        train,
        test,
        behavior_column=args.behavior_column,
        min_train_samples=args.min_train_samples_per_behavior,
    )

    train, test, test_label_alignment = filter_test_behaviors_to_train(
        train, test, behavior_column=args.behavior_column
    )

    n = len(df)
    print(
        f"{summary}\n"
        f"Total: {n:,} | Treino: {len(train):,} ({100 * len(train) / n:.2f}%) | "
        f"Teste: {len(test):,} ({100 * len(test) / n:.2f}%)"
    )

    path_train, path_test = out / "train.parquet", out / "test.parquet"
    train.to_parquet(path_train, engine="pyarrow", compression="snappy", index=False)
    test.to_parquet(path_test, engine="pyarrow", compression="snappy", index=False)

    report = build_split_report(
        df,
        train,
        test,
        subject_column=args.subject_column,
        behavior_column=args.behavior_column,
        method=method,
        test_label_alignment=test_label_alignment,
    )
    report["insufficient_train_behavior_filter"] = insufficient_train_meta
    path_report = out / "split_report.json"
    path_report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Relatório: {path_report.resolve()}\nPasta: {out.resolve()}\nTreino: {path_train.resolve()}\nTeste: {path_test.resolve()}")


if __name__ == "__main__":
    main()
