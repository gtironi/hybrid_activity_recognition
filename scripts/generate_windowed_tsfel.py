"""
Gera o dataset janelado com features TSFEL (paralelo por chunk).

Le:  dataset/processed/AcTBeCalf/Time_Adj_Raw_Data.parquet
Escreve: dataset/processed/AcTBeCalf/Windowed_Time_Adj.parquet

Estrategia: 8 chunks em paralelo, cada um TSFEL single-thread.
            Evita o overhead de IPC do TSFEL ao distribuir 135k DataFrames pequenos.

Uso (background, sobrevive a logout):
    cd /home/emap/tironi/hybrid_activity_recognition
    mkdir -p logs
    nohup ./venv/bin/python scripts/generate_windowed_tsfel.py \
        > logs/generate_windowed_tsfel.log 2>&1 &
    echo "PID: $!"
    disown
"""
import os

# IMPORTANTE: limitar threads de BLAS/OpenMP a 1 ANTES de importar numpy/scipy/tsfel.
# Sem isso, cada worker spawna 4+ threads internos e disputam CPU entre si
# (oversubscription -> cada worker fica em ~20% CPU em vez de ~100%).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import contextlib
import gc
import io
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tsfel
from numpy.lib.stride_tricks import sliding_window_view

INPUT_PATH = "dataset/processed/AcTBeCalf/Time_Adj_Raw_Data.parquet"
OUTPUT_PATH = "dataset/processed/AcTBeCalf/Windowed_Time_Adj.parquet"
TEMP_DIR = "temp_chunks"

WINDOW_SIZE = 75
OVERLAP = 0.50
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP))
CHUNK_SIZE = 1_000_000
N_PARALLEL_CHUNKS = 8
SAMPLING_RATE = 25

SELECTED_FEATURES = [
    "accX_Entropy", "accX_Signal distance", "accY_Entropy",
    "accY_Peak to peak distance", "accZ_Signal distance", "accZ_Entropy",
    "accX_Peak to peak distance", "accX_Interquartile range", "accY_Signal distance",
    "accZ_Sum absolute diff", "accX_Spectral decrease", "accX_Spectral centroid",
    "accX_Spectral spread", "accZ_Spectral decrease", "accY_Sum absolute diff",
    "accY_ECDF Percentile_1", "accY_Histogram mode", "accZ_Peak to peak distance",
    "accY_Max", "accY_Spectral decrease", "accY_ECDF Percentile_0", "accY_Min",
    "accX_Spectral distance", "accX_Wavelet energy_1.56Hz", "accX_Maximum frequency",
    "accY_Area under the curve", "accX_Wavelet variance_1.56Hz",
    "accX_Spectrogram mean coefficient_2.02Hz", "accY_Absolute energy",
    "accY_Wavelet energy_0.69Hz", "accY_Spectral spread", "accY_Spectral centroid",
    "accX_Wavelet energy_1.25Hz", "accX_Positive turning points",
    "accY_Spectral distance", "accZ_Spectral spread", "accX_Root mean square",
    "accY_Spectral variation", "accY_MFCC_0", "accX_Max", "accX_Mean",
    "accX_Area under the curve", "accX_Wavelet variance_1.25Hz",
    "accX_Wavelet energy_1.04Hz", "accZ_ECDF Percentile_1", "accX_Spectral entropy",
    "accX_Absolute energy", "accX_Wavelet variance_1.04Hz", "accZ_Max",
    "accZ_Spectral centroid", "accX_Autocorrelation", "accY_Maximum frequency",
    "accZ_ECDF Percentile_0", "accX_Wavelet energy_0.69Hz", "accZ_Min",
    "accZ_Wavelet variance_1.56Hz", "accZ_Absolute energy", "accY_LPCC_1",
    "accX_Negative turning points", "accX_Histogram mode",
    "accZ_Area under the curve", "accY_Negative turning points",
    "accX_ECDF Percentile_1", "accY_Autocorrelation", "accX_MFCC_1",
    "accZ_Wavelet energy_0.69Hz", "accZ_Wavelet variance_2.08Hz", "accZ_MFCC_1",
    "accX_MFCC_0", "accZ_MFCC_0", "accX_Min", "accY_Spectral entropy",
    "accY_MFCC_1", "accX_Spectral variation", "accY_Power bandwidth",
]


class _NullStream(io.IOBase):
    """Descarta writes (suprime a barra de progresso interna do TSFEL)."""
    def write(self, _): return 0
    def flush(self): pass
    def isatty(self): return False


def get_optimized_tsfel_config():
    cfg_all = tsfel.get_features_by_domain()
    cfg_filtered = {}
    target_bases = set()
    for feat in SELECTED_FEATURES:
        base = feat.split("_", 1)[1] if "_" in feat else feat
        target_bases.add(base)
        if "_" in base:
            target_bases.add(base.split("_")[0])

    for domain, features in cfg_all.items():
        for feat_name, feat_params in features.items():
            if feat_name in target_bases or any(t in feat_name for t in target_bases):
                cfg_filtered.setdefault(domain, {})[feat_name] = feat_params
    return cfg_filtered


def process_chunk_worker(args):
    """
    Roda em um processo worker. Recebe (chunk_id, accX, accY, accZ, times)
    e devolve (chunk_id, save_path, n_windows) ou (chunk_id, None, 0) em erro.
    Cada worker e TSFEL roda single-thread (sem n_jobs).
    """
    chunk_id, accX, accY, accZ, times = args

    cfg = get_optimized_tsfel_config()
    data = np.stack([accX, accY, accZ], axis=1).astype(np.float32)

    try:
        windows_view = sliding_window_view(data, window_shape=(WINDOW_SIZE, 3))
        windows = windows_view[::STRIDE, 0, :, :]
        times_window = times[: len(data) - WINDOW_SIZE + 1][::STRIDE]
    except ValueError:
        return (chunk_id, None, 0)

    if len(windows) == 0:
        return (chunk_id, None, 0)

    df_win = pd.DataFrame({
        "dateTime": times_window,
        "acc_x": list(windows[:, :, 0]),
        "acc_y": list(windows[:, :, 1]),
        "acc_z": list(windows[:, :, 2]),
        "label": -1,
    })

    tsfel_input = [pd.DataFrame(w, columns=["accX", "accY", "accZ"]) for w in windows]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with contextlib.redirect_stdout(_NullStream()), \
                 contextlib.redirect_stderr(_NullStream()):
                X_feat = tsfel.time_series_features_extractor(
                    cfg, tsfel_input, fs=SAMPLING_RATE, n_jobs=1
                )
        except Exception as e:
            return (chunk_id, f"ERRO: {e}", 0)

    X_feat = X_feat.astype(np.float32).fillna(0)
    cols_to_keep = [c for c in X_feat.columns if c in SELECTED_FEATURES]
    X_feat = X_feat[cols_to_keep]
    for col in SELECTED_FEATURES:
        if col not in X_feat.columns:
            X_feat[col] = 0.0
    X_feat = X_feat[SELECTED_FEATURES]

    df_final = pd.concat([df_win, X_feat], axis=1)
    save_path = os.path.join(TEMP_DIR, f"chunk_{chunk_id:06d}.parquet")
    df_final.to_parquet(save_path, index=False)

    n = len(df_win)
    del windows, tsfel_input, X_feat, df_final, df_win, data
    gc.collect()
    return (chunk_id, save_path, n)


def make_task(chunk_id, df_chunk):
    """Converte DataFrame para tupla serializavel (arrays numpy) para mandar ao worker."""
    return (
        chunk_id,
        df_chunk["accX"].values.astype(np.float32),
        df_chunk["accY"].values.astype(np.float32),
        df_chunk["accZ"].values.astype(np.float32),
        df_chunk["dateTime"].values,
    )


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"ERRO: arquivo de entrada nao existe: {INPUT_PATH}", flush=True)
        sys.exit(1)

    os.makedirs(TEMP_DIR, exist_ok=True)

    parquet_file = pq.ParquetFile(INPUT_PATH)
    total_rows = parquet_file.metadata.num_rows
    n_chunks_est = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(
        f"Input: {INPUT_PATH} | linhas: {total_rows:,} | "
        f"chunk_size: {CHUNK_SIZE:,} | parallel: {N_PARALLEL_CHUNKS} workers | "
        f"chunks estimados: {n_chunks_est}",
        flush=True,
    )

    t_start = time.time()
    chunk_files = []
    n_done = 0
    next_pct_mark = 10

    batches = parquet_file.iter_batches(batch_size=CHUNK_SIZE)

    with ProcessPoolExecutor(max_workers=N_PARALLEL_CHUNKS) as exe:
        pending = {}

        def submit_next(chunk_id):
            try:
                batch = next(batches)
            except StopIteration:
                return False
            df_chunk = batch.to_pandas()
            task = make_task(chunk_id, df_chunk)
            del df_chunk
            fut = exe.submit(process_chunk_worker, task)
            pending[fut] = chunk_id
            return True

        # Submete os primeiros N_PARALLEL_CHUNKS
        next_id = 0
        for _ in range(N_PARALLEL_CHUNKS):
            if submit_next(next_id):
                next_id += 1

        # A medida que cada um termina, submete o proximo
        while pending:
            done_fut = next(as_completed(pending))
            cid = pending.pop(done_fut)
            try:
                chunk_id, path, n_windows = done_fut.result()
            except Exception as e:
                print(f"[Chunk {cid}] FALHOU: {e}", flush=True)
                if submit_next(next_id):
                    next_id += 1
                continue

            n_done += 1
            elapsed = time.time() - t_start
            eta = (elapsed / n_done) * (n_chunks_est - n_done)

            if path is None or (isinstance(path, str) and path.startswith("ERRO")):
                print(
                    f"[Chunk {chunk_id+1}/{n_chunks_est}] vazio/erro: {path} | "
                    f"({n_done}/{n_chunks_est} feitos)",
                    flush=True,
                )
            else:
                chunk_files.append(path)
                print(
                    f"[Chunk {chunk_id+1}/{n_chunks_est}] {n_windows:,} janelas | "
                    f"feitos: {n_done}/{n_chunks_est} | "
                    f"elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min",
                    flush=True,
                )

            pct = (n_done / n_chunks_est) * 100
            while next_pct_mark <= 100 and pct >= next_pct_mark:
                print(
                    f"=== {next_pct_mark}% completo ({n_done}/{n_chunks_est} chunks, "
                    f"{elapsed/60:.1f}min) ===",
                    flush=True,
                )
                next_pct_mark += 10

            if submit_next(next_id):
                next_id += 1

    if not chunk_files:
        print("Nenhum chunk processado.", flush=True)
        sys.exit(2)

    print(f"\nConcatenando {len(chunk_files)} chunks -> {OUTPUT_PATH}...", flush=True)
    chunk_files.sort()
    tables = [pq.read_table(f) for f in chunk_files]
    full_table = pa.concat_tables(tables)
    pq.write_table(full_table, OUTPUT_PATH, compression="snappy")
    print(
        f"OK. {full_table.num_rows:,} janelas | "
        f"{os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB | "
        f"tempo total: {(time.time()-t_start)/60:.1f}min",
        flush=True,
    )

    for f in chunk_files:
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        os.rmdir(TEMP_DIR)
    except OSError:
        pass


if __name__ == "__main__":
    main()
