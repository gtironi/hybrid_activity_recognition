#!/usr/bin/env python3
"""
Separa um CSV em treino / teste por comportamento ou por sujeito.

Modo padrão: split estratificado por comportamento (20% para teste), preservando
a proporção das classes.

Modo alternativo: split por sujeito, usando os valores em --test-subjects.

Saída: Parquet em {out_dir}/{nome_do_csv_sem_extensão}/train.parquet e test.parquet (pyarrow).
    Ex.: dataset/processed/AcTBeCalf/train.parquet

Uso:
    python scripts/dataset_processing.py
    python scripts/dataset_processing.py --csv dataset/AcTBeCalf.csv --out-dir dataset/processed
    python scripts/dataset_processing.py --split-by subject --subject-column calfId --test-subjects 1329 1343
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_TEST_SUBJECTS = (1329, 1343, 1353, 1357, 1372)
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_BEHAVIOR_COLUMN = "behaviour"


def _test_values_for_column(series: pd.Series, raw: list[str]) -> set:
    """Alinha tipos dos argumentos ao dtype da coluna."""
    if pd.api.types.is_integer_dtype(series):
        return {int(x) for x in raw}
    if pd.api.types.is_float_dtype(series):
        return {float(x) for x in raw}
    return set(raw)


def _split_by_subject(
    df: pd.DataFrame,
    *,
    subject_column: str,
    test_subjects: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if subject_column not in df.columns:
        raise SystemExit(f"Coluna inexistente: {subject_column!r}. Colunas: {list(df.columns)}")

    test_set = _test_values_for_column(df[subject_column], test_subjects)
    mask_test = df[subject_column].isin(test_set)
    df_test = df.loc[mask_test].reset_index(drop=True)
    df_train = df.loc[~mask_test].reset_index(drop=True)
    return df_train, df_test


def _split_by_behavior(
    df: pd.DataFrame,
    *,
    behavior_column: str,
    test_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if behavior_column not in df.columns:
        raise SystemExit(f"Coluna inexistente: {behavior_column!r}. Colunas: {list(df.columns)}")
    if not 0 < test_fraction < 1:
        raise SystemExit("--test-fraction deve estar entre 0 e 1.")

    counts = df[behavior_column].value_counts(dropna=False)
    use_stratify = counts.min() >= 2
    if not use_stratify:
        warnings.warn(
            f"Split sem estratificação: a coluna {behavior_column!r} tem classes com menos de 2 amostras.",
            RuntimeWarning,
            stacklevel=2,
        )

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_fraction,
        stratify=df[behavior_column] if use_stratify else None,
        random_state=random_state,
        shuffle=True,
    )
    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test = df.loc[test_idx].reset_index(drop=True)
    return df_train, df_test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split CSV train/test por comportamento ou sujeito → Parquet"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("dataset/AcTBeCalf.csv"),
        help="CSV fonte",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dataset/processed"),
        help="Pasta base; dentro dela é criada uma subpasta com o nome do CSV (stem)",
    )
    parser.add_argument(
        "--subject-column",
        default="calfId",
        help="Nome da coluna usada no split por sujeito",
    )
    parser.add_argument(
        "--test-subjects",
        type=str,
        nargs="*",
        default=[str(x) for x in DEFAULT_TEST_SUBJECTS],
        help="Valores dessa coluna que vão para o teste quando --split-by subject",
    )
    parser.add_argument(
        "--split-by",
        choices=("behavior", "subject"),
        default="behavior",
        help="Modo de split: behavior = estratificado por comportamento; subject = por sujeito",
    )
    parser.add_argument(
        "--behavior-column",
        default=DEFAULT_BEHAVIOR_COLUMN,
        help="Nome da coluna usada para estratificar o split por comportamento",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Fração reservada para teste quando --split-by behavior",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed usada no split por comportamento",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"Arquivo não encontrado: {args.csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = args.out_dir / args.csv.stem
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lendo {args.csv} ...")
    df = pd.read_csv(args.csv)

    if args.split_by == "subject":
        col = args.subject_column
        df_train, df_test = _split_by_subject(
            df,
            subject_column=col,
            test_subjects=args.test_subjects,
        )
        split_summary = (
            f"Coluna: {col} | test_subjects="
            f"{sorted(_test_values_for_column(df[col], args.test_subjects), key=str)!r}"
        )
    else:
        col = args.behavior_column
        df_train, df_test = _split_by_behavior(
            df,
            behavior_column=col,
            test_fraction=args.test_fraction,
            random_state=args.random_state,
        )
        split_summary = (
            f"Coluna: {col} | test_fraction={args.test_fraction:.2f} | "
            f"random_state={args.random_state}"
        )

    n = len(df)
    n_train, n_test = len(df_train), len(df_test)
    print(
        f"{split_summary}\n"
        f"Total: {n:,} | Treino: {n_train:,} ({100 * n_train / n:.2f}%) | "
        f"Teste: {n_test:,} ({100 * n_test / n:.2f}%)"
    )

    path_train = dataset_dir / "train.parquet"
    path_test = dataset_dir / "test.parquet"

    df_train.to_parquet(path_train, engine="pyarrow", compression="snappy", index=False)
    df_test.to_parquet(path_test, engine="pyarrow", compression="snappy", index=False)

    print(f"Pasta:   {dataset_dir.resolve()}")
    print(f"Treino:  {path_train.resolve()}")
    print(f"Teste:   {path_test.resolve()}")


if __name__ == "__main__":
    main()
