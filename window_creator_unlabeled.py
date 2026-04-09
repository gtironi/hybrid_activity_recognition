import pandas as pd
import numpy as np
import os
import tsfel
import warnings
import gc
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

INPUT_PATH = '/kaggle/input/actbecalf/Time_Adj_Raw_Data.parquet'
OUTPUT_PATH = 'Windowed_Time_Adj.parquet'
TEMP_DIR = 'temp_chunks'
WINDOW_SIZE = 75
OVERLAP = 0.50
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP))
CHUNK_SIZE = 500000 

# Lista exata das 75 features
SELECTED_FEATURES = [
    'accY_Entropy', 'accX_Signal distance', 'accY_Sum absolute diff', 'accX_Entropy',
    'accZ_Entropy', 'accY_Peak to peak distance', 'accZ_Sum absolute diff',
    'accY_Signal distance', 'accZ_Signal distance', 'accX_Maximum frequency',
    'accX_Spectral spread', 'accX_Peak to peak distance', 'accY_Spectral centroid',
    'accX_Spectral centroid', 'accX_Interquartile range', 'accZ_Peak to peak distance',
    'accX_Spectral decrease', 'accZ_Spectral centroid', 'accY_Spectral decrease',
    'accY_ECDF Percentile_1', 'accY_Spectral variation', 'accX_Spectrogram mean coefficient',
    'accY_Max', 'accX_Spectral distance', 'accY_Histogram mode', 'accX_MFCC',
    'accY_Spectral spread', 'accZ_Spectral decrease', 'accY_MFCC', 'accX_Wavelet energy',
    'accY_ECDF Percentile', 'accX_Max', 'accZ_MFCC', 'accZ_Spectral skewness',
    'accX_Wavelet variance', 'accY_Min', 'accX_Positive turning points',
    'accY_LPCC', 'accX_Spectral entropy', 'accZ_Spectral spread', 
    'accY_Area under the curve', 'accX_Absolute energy', 'accY_Root mean square',
    'accY_Maximum frequency', 'accY_Spectral distance', 'accX_Negative turning points',
    'accX_Root mean square', 'accZ_LPCC', 'accY_Absolute energy', 'accZ_Spectral entropy',
    'accY_Power bandwidth', 'accX_Mean', 'accZ_Area under the curve', 'accY_Autocorrelation'
]

def get_optimized_tsfel_config():
    """ 
    Cria um config que só calcula as 75 que queremos
    """
    cfg_all = tsfel.get_features_by_domain()
    cfg_filtered = {}
    
    target_bases = set()
    for feat in SELECTED_FEATURES:
        if '_' in feat:
            base = feat.split('_', 1)[1] 
        else:
            base = feat
            
        target_bases.add(base)
        if '_' in base:
            target_bases.add(base.split('_')[0])

    # Filtra o dicionário
    count = 0
    for domain, features in cfg_all.items():
        for feat_name, feat_params in features.items():
            if feat_name in target_bases or any(t in feat_name for t in target_bases):
                if domain not in cfg_filtered:
                    cfg_filtered[domain] = {}
                cfg_filtered[domain][feat_name] = feat_params
                count += 1
                
    print(f"Config TSFEL: {count} funções base selecionadas.")
    return cfg_filtered

CFG_OPTIMIZED = get_optimized_tsfel_config()
os.makedirs(TEMP_DIR, exist_ok=True)

def process_chunk_vectorized(df_chunk, chunk_id):
    # Dados para Numpy (mais rápido)
    data = df_chunk[['accX', 'accY', 'accZ']].values.astype(np.float32)
    times = df_chunk['dateTime'].values
    
    try:
        windows_view = sliding_window_view(data, window_shape=(WINDOW_SIZE, 3))
        windows = windows_view[::STRIDE, 0, :, :]
        times_window = times[:len(data) - WINDOW_SIZE + 1][::STRIDE]
    except ValueError:
        return None # Chunk muito pequeno

    if len(windows) == 0: return None

    # Prepara listas para o DataFrame final
    acc_x_list = list(windows[:, :, 0])
    acc_y_list = list(windows[:, :, 1])
    acc_z_list = list(windows[:, :, 2])

    df_win = pd.DataFrame({
        'dateTime': times_window,
        'acc_x': acc_x_list,
        'acc_y': acc_y_list,
        'acc_z': acc_z_list,
        'label': -1
    })

    # Extração TSFEL Otimizada
    tsfel_input = [pd.DataFrame(w, columns=['accX', 'accY', 'accZ']) for w in windows]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            X_feat = tsfel.time_series_features_extractor(
                CFG_OPTIMIZED, tsfel_input, fs=25, verbose=0, n_jobs=-1
            )
        except Exception as e:
            print(f"⚠️ Erro TSFEL Chunk {chunk_id}: {e}")
            return None

    # Limpeza
    X_feat = X_feat.astype(np.float32).fillna(0)
    cols_to_keep = [c for c in X_feat.columns if c in SELECTED_FEATURES]
    X_feat = X_feat[cols_to_keep]
    
    # Cria colunas faltantes com 0 (caso alguma feature complexa tenha falhado)
    for col in SELECTED_FEATURES:
        if col not in X_feat.columns: X_feat[col] = 0.0
    X_feat = X_feat[SELECTED_FEATURES]

    # Salva Chunk
    df_final = pd.concat([df_win, X_feat], axis=1)
    save_path = os.path.join(TEMP_DIR, f'chunk_{chunk_id}.parquet')
    df_final.to_parquet(save_path, index=False)
    
    del windows, tsfel_input, X_feat, df_final
    gc.collect()
    
    print(f"   -> Chunk {chunk_id} salvo ({len(df_win)} janelas).")
    return save_path

if os.path.exists(INPUT_PATH):
    print(f"Processando {INPUT_PATH} com Otimização Máxima...")
    parquet_file = pq.ParquetFile(INPUT_PATH)
    chunk_files = []
    
    # SUBSAMPLING
    SKIP_FACTOR = 1 
    
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=CHUNK_SIZE)):
        if i % SKIP_FACTOR != 0: continue
            
        print(f"\n📦 Batch {i}...")
        df_chunk = batch.to_pandas()
        saved = process_chunk_vectorized(df_chunk, i)
        if saved: chunk_files.append(saved)
        
        del df_chunk
        gc.collect()

    if chunk_files:
        import pyarrow as pa
        tables = [pq.read_table(f) for f in chunk_files]
        full_table = pa.concat_tables(tables)
        pq.write_table(full_table, OUTPUT_PATH, compression='snappy')
        
        print("Sucesso!")
        for f in chunk_files: os.remove(f)
        os.rmdir(TEMP_DIR)
    else:
        print("Nada processado.")
else:
    print("Arquivo não encontrado.")