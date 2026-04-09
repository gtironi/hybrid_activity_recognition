#!/usr/bin/env python3
"""
Análise exploratória do AcTBeCalf por calfId: contagem de linhas,
distribuição percentual por comportamento e nº de comportamentos únicos.

Usa pandas (read_csv em memória).

Uso:
  python scripts/explore_actbecalf_by_calf.py
  python scripts/explore_actbecalf_by_calf.py --csv dataset/AcTBeCalf.csv --out-dir output/eda_calf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="EDA AcTBeCalf por calfId")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("dataset/AcTBeCalf.csv"),
        help="Caminho para AcTBeCalf.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Se definido, grava CSVs detalhados nesta pasta",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"Arquivo não encontrado: {args.csv}")

    df = pd.read_csv(args.csv, dtype={"calfId": "int64", "segId": "int64"})

    per_calf = (
        df.groupby("calfId", sort=True)
        .agg(n_rows=("behaviour", "size"), n_behaviours_unique=("behaviour", "nunique"))
        .reset_index()
    )

    global_unique = df["behaviour"].nunique()
    behaviour_list = pd.DataFrame(sorted(df["behaviour"].dropna().unique()), columns=["behaviour"])

    counts = (
        df.groupby(["calfId", "behaviour"], sort=True).size().reset_index(name="n")
    )
    totals = per_calf[["calfId", "n_rows"]].rename(columns={"n_rows": "total_rows"})
    dist = counts.merge(totals, on="calfId", how="left")
    dist["pct_within_calf"] = dist["n"] / dist["total_rows"] * 100.0

    print("=" * 60)
    print("AcTBeCalf — resumo por calfId")
    print("=" * 60)
    print(f"Arquivo: {args.csv.resolve()}")
    print(f"Comportamentos únicos no dataset (global): {global_unique}")
    print()
    print("--- Por bezerro: [comportamento (%)...] + linhas + comportamentos distintos ---")
    for _, row in per_calf.sort_values("calfId").iterrows():
        cid = int(row["calfId"])
        n_lines = int(row["n_rows"])
        n_beh = int(row["n_behaviours_unique"])
        sub = dist[dist["calfId"] == cid].sort_values("pct_within_calf", ascending=False)
        parts = [
            f"{r['behaviour']} ({r['pct_within_calf']:.2f}%)" for _, r in sub.iterrows()
        ]
        print(f"calf {cid}: [{', '.join(parts)}]")
        print(f"  → {n_lines:,} linhas, {n_beh} comportamentos distintos\n")

    print("--- Lista global de comportamentos ---")
    print(behaviour_list.to_string(index=False))

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        per_calf.to_csv(args.out_dir / "per_calf_summary.csv", index=False)
        dist.to_csv(args.out_dir / "per_calf_behaviour_distribution.csv", index=False)
        behaviour_list.to_csv(args.out_dir / "behaviours_unique_list.csv", index=False)
        print()
        print(f"CSVs salvos em: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
