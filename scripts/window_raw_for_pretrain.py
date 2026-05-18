"""Janelar CSV raw (dateTime, accX, accY, accZ) em parquet pronto para PatchTST pretrain.

Saída: parquet com colunas ``acc_x, acc_y, acc_z`` onde cada célula é uma lista
de ``window_len`` floats. Sem label, sem TSFEL — apenas o sinal para SSL.

Uso:
    python scripts/window_raw_for_pretrain.py \\
        --input dataset/Time_Adj_Raw_Data.csv \\
        --output dataset/processed/pretrain_raw_windowed.parquet \\
        --window_len 75 --stride 37
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def window_raw(df: pd.DataFrame, window_len: int, stride: int) -> pd.DataFrame:
    data = df[["accX", "accY", "accZ"]].to_numpy(dtype=np.float32)
    N = len(data)
    starts = range(0, N - window_len + 1, stride)
    xs, ys, zs = [], [], []
    for s in starts:
        e = s + window_len
        xs.append(data[s:e, 0].tolist())
        ys.append(data[s:e, 1].tolist())
        zs.append(data[s:e, 2].tolist())
    return pd.DataFrame({"acc_x": xs, "acc_y": ys, "acc_z": zs})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="CSV raw com dateTime,accX,accY,accZ")
    p.add_argument("--output", required=True, help="Parquet janelado de saída")
    p.add_argument("--window_len", type=int, default=75)
    p.add_argument("--stride", type=int, default=37)
    p.add_argument("--chunksize", type=int, default=2_000_000,
                   help="Linhas por chunk ao ler o CSV (CSV grande)")
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Lendo {args.input} em chunks de {args.chunksize}...")
    chunks = []
    total_rows = 0
    leftover = pd.DataFrame(columns=["dateTime", "accX", "accY", "accZ"])
    for i, chunk in enumerate(pd.read_csv(args.input, chunksize=args.chunksize)):
        total_rows += len(chunk)
        # Concatena sobra do chunk anterior para não perder janelas na fronteira.
        chunk = pd.concat([leftover, chunk], ignore_index=True)
        w = window_raw(chunk, args.window_len, args.stride)
        chunks.append(w)
        # Guarda as últimas (window_len - 1) linhas para o próximo chunk.
        leftover = chunk.iloc[-(args.window_len - 1):].reset_index(drop=True)
        print(f"  chunk {i}: linhas acumuladas={total_rows} | janelas no chunk={len(w)}")

    df_out = pd.concat(chunks, ignore_index=True)
    df_out.to_parquet(out, index=False)
    print(f"OK: {len(df_out)} janelas → {out}")


if __name__ == "__main__":
    main()
