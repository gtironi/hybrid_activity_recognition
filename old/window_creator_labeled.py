import pandas as pd
import numpy as np
from collections import Counter
import os
import tsfel
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

INPUT_PATH = 'AcTBeCalf.parquet'
OUTPUT_PATH = 'WindowedCalf.parquet'
WINDOW_SIZE = 75                      # 3s * 25Hz
OVERLAP = 0.50                        # 50%
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP))
PURITY_THRESHOLD = 0.90
NUM_TOP_FEATURES = 75                 # Quantas features do TSFEL queremos salvar
SAMPLE_SIZE_FOR_SELECTION = 3000      # Quantas janelas usar para descobrir quais features são boas

# Taxonomia de 19 Classes
def map_behavior(label):
    label = str(label).lower().strip()
    if any(x in label for x in ['cough', 'fall', 'vocalisation']): return 'rare_event'
    if any(x in label for x in ['cross-suckle', 'tongue', 'abnormal']): return 'abnormal'
    if any(x in label for x in ['scratch', 'rub', 'stretch', 'srs']): return 'srs'
    if 'defecat' in label or 'urinat' in label: return 'elimination'
    if any(x in label for x in ['play', 'jump', 'headbutt', 'mount']): return 'play'
    if 'social' in label or 'nudge' in label: return 'social_interaction'
    if 'ruminat' in label: return 'rumination'
    if 'drink' in label: return 'drinking'
    if 'eat' in label: return 'eating'
    if 'sniff' in label: return 'sniff'
    if 'oral' in label or 'manipulation' in label: return 'oral_manipulation'
    if 'groom' in label: return 'grooming'
    if 'rising' in label: return 'rising'
    if 'lying down' in label or 'lying-down' in label: return 'lying_down_action'
    if 'run' in label: return 'running'
    if 'walk' in label or 'backward' in label: return 'walking'
    if 'ly' in label: return 'lying'
    if 'stand' in label: return 'standing'
    return 'other'

# Criação de Janelas 
def create_windowed_dataframe(input_file):
    print(f"📂 Lendo arquivo bruto: {input_file}...")
    cols = ['dateTime', 'calfId', 'segId', 'accX', 'accY', 'accZ', 'behaviour']
    df = pd.read_parquet(input_file, columns=cols)
    
    df['behaviour'] = df['behaviour'].apply(map_behavior)

    window_list = []
    grouped = df.groupby(['calfId', 'segId'])

    for i, ((calf_id, seg_id), group) in enumerate(grouped):
        raw_data = group[['accX', 'accY', 'accZ']].values
        timestamps = group['dateTime'].values
        labels = group['behaviour'].values
        n_samples = len(raw_data)
        
        if n_samples < WINDOW_SIZE: continue
            
        for start_idx in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
            end_idx = start_idx + WINDOW_SIZE
            window_labels = labels[start_idx : end_idx]
            
            # Checa pureza
            counts = Counter(window_labels)
            most_common_label, count = counts.most_common(1)[0]
            
            if (count / WINDOW_SIZE) >= PURITY_THRESHOLD:
                window_signals = raw_data[start_idx : end_idx]
                
                window_dict = {
                    'dateTime': timestamps[start_idx], # Pega o tempo do INÍCIO da janela
                    'calf_id': calf_id,
                    'acc_x': window_signals[:, 0].tolist(),
                    'acc_y': window_signals[:, 1].tolist(),
                    'acc_z': window_signals[:, 2].tolist(),
                    'label': most_common_label
                }
                window_list.append(window_dict)

    print(f"✅ Total de Janelas Geradas: {len(window_list)}")
    return pd.DataFrame(window_list)

# Função para Selecionar as Top Features 
def discover_top_features(df_windows, top_n=75):
    print("\n🔍 FASE DE DESCOBERTA: Selecionando as melhores features...")
    
    try:
        sample_df, _ = train_test_split(df_windows, train_size=SAMPLE_SIZE_FOR_SELECTION, stratify=df_windows['label'], random_state=42)
    except ValueError:
        sample_df = df_windows.sample(n=min(len(df_windows), SAMPLE_SIZE_FOR_SELECTION), random_state=42)
        
    print(f"   Amostra de {len(sample_df)} janelas criada.")
    
    # Formata para TSFEL
    tsfel_input = []
    for _, row in sample_df.iterrows():
        tsfel_input.append(pd.DataFrame({'accX': row['acc_x'], 'accY': row['acc_y'], 'accZ': row['acc_z']}))
    cfg = tsfel.get_features_by_domain()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("   Extraindo features da amostra (Isso pode demorar)...")
        X = tsfel.time_series_features_extractor(cfg, tsfel_input, fs=25, verbose=0)
    
    X.fillna(0, inplace=True)
    
    # Limpeza Rápida (tirar baixa variância e alta correlação)
    sel_var = VarianceThreshold(threshold=0.02)
    try:
        sel_var.fit(X)
        X = X.loc[:, sel_var.get_support()]
    except: pass
    
    corr_features = tsfel.correlated_features(X, threshold=0.98)
    X.drop(corr_features, axis=1, inplace=True)
    
    # Random Forest para Ranking
    print("   Treinando Random Forest para ranqueamento...")
    le = LabelEncoder()
    y = le.fit_transform(sample_df['label'])
    
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = X.columns[indices].tolist()
    
    print(f"{len(top_features)} Features Selecionadas!")
    print(f"   Top 3: {top_features[:3]}")
    return top_features

if os.path.exists(INPUT_PATH):
    df_main = create_windowed_dataframe(INPUT_PATH)
    top_feature_names = discover_top_features(df_main, top_n=NUM_TOP_FEATURES)

    print(f"\nCalculando features para todas as {len(df_main)} janelas...")
    tsfel_input_full = []
    tsfel_input_full = [
        pd.DataFrame({'accX': r[0], 'accY': r[1], 'accZ': r[2]}) 
        for r in zip(df_main['acc_x'], df_main['acc_y'], df_main['acc_z'])
    ]
    
    cfg = tsfel.get_features_by_domain()
    BATCH_SIZE = 5000
    all_features_dfs = []
    
    total_batches = (len(tsfel_input_full) // BATCH_SIZE) + 1
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(0, len(tsfel_input_full), BATCH_SIZE):
            batch = tsfel_input_full[i : i + BATCH_SIZE]
            print(f"   Batch {(i // BATCH_SIZE) + 1}/{total_batches}...")
            
            # Extrai todas do batch
            X_batch = tsfel.time_series_features_extractor(cfg, batch, fs=25, verbose=0)
            X_batch.fillna(0, inplace=True)
            
            # Filtra apenas as Top Features
            available_cols = [c for c in top_feature_names if c in X_batch.columns]
            X_batch_filtered = X_batch[available_cols]
            
            # Garante que todas as 75 colunas existam
            for col in top_feature_names:
                if col not in X_batch_filtered.columns:
                    X_batch_filtered[col] = 0.0
                    
            X_batch_filtered = X_batch_filtered[top_feature_names]
            all_features_dfs.append(X_batch_filtered)
            
    df_features_final = pd.concat(all_features_dfs, axis=0).reset_index(drop=True)
    
    # Reseta índice do df_main para garantir alinhamento
    df_main.reset_index(drop=True, inplace=True)
    
    # Junta as colunas originais com as novas features
    df_final = pd.concat([df_main, df_features_final], axis=1)
    
    print(f"Salvando em {OUTPUT_PATH}...")
    df_final.to_parquet(OUTPUT_PATH, engine='pyarrow', compression='snappy')

    print(df_final.info())
    print(f"Shape final: {df_final.shape}")
    print("Features incluídas (Exemplo):", df_final.columns[-5:].tolist())
else:
    print("Arquivo de entrada não encontrado.")