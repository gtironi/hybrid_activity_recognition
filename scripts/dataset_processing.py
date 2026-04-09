#!/usr/bin/env python3
"""
Separa um CSV em treino / teste por coluna de sujeito (ex.: calfId, user_id).

Linhas cuja coluna de sujeito está em --test-subjects → teste; o restante → treino.

Saída: Parquet em {out_dir}/{nome_do_csv_sem_extensão}/train.parquet e test.parquet (pyarrow).
  Ex.: dataset/processed/AcTBeCalf/train.parquet

Uso:
  python scripts/dataset_processing.py
  python scripts/dataset_processing.py --csv dataset/AcTBeCalf.csv --out-dir dataset/processed
  python scripts/dataset_processing.py --subject-column participant_id --test-subjects A1 B2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_TEST_SUBJECTS = (1329, 1343, 1353, 1357, 1372)


def _test_values_for_column(series: pd.Series, raw: list[str]) -> set:
    """Alinha tipos dos argumentos ao dtype da coluna."""
    if pd.api.types.is_integer_dtype(series):
        return {int(x) for x in raw}
    if pd.api.types.is_float_dtype(series):
        return {float(x) for x in raw}
    return set(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split CSV train/test por coluna de sujeito → Parquet"
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
        help="Nome da coluna usada para dividir treino/teste (IDs de sujeito)",
    )
    parser.add_argument(
        "--test-subjects",
        type=str,
        nargs="*",
        default=[str(x) for x in DEFAULT_TEST_SUBJECTS],
        help="Valores dessa coluna que vão para o teste (padrão: os 5 bezerros AcTBeCalf)",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"Arquivo não encontrado: {args.csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = args.out_dir / args.csv.stem
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lendo {args.csv} ...")
    df = pd.read_csv(args.csv)

    col = args.subject_column
    if col not in df.columns:
        raise SystemExit(f"Coluna inexistente: {col!r}. Colunas: {list(df.columns)}")

    test_set = _test_values_for_column(df[col], args.test_subjects)

    mask_test = df[col].isin(test_set)
    df_test = df.loc[mask_test].reset_index(drop=True)
    df_train = df.loc[~mask_test].reset_index(drop=True)

    n = len(df)
    n_train, n_test = len(df_train), len(df_test)
    print(
        f"Coluna: {col} | test_subjects={sorted(test_set, key=str)!r}\n"
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
