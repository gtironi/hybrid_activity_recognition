import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

PARQUET_PATH = 'WindowedCalf.parquet'

def train_best_random_forest(file_path):
    print(f"Carregando dataset: {file_path}...")
    
    # Lê o dataset completo
    df = pd.read_parquet(file_path)
    class_counts = df['label'].value_counts()
    print("\n🔍 Verificando contagem de classes:")
    print(class_counts)
    to_remove = class_counts[class_counts < 2].index
    
    if len(to_remove) > 0:
        print(f"\nREMOVENDO classes com amostras insuficientes (<2): {list(to_remove)}")
        df = df[~df['label'].isin(to_remove)]
        print(f"   Novo tamanho do dataset: {len(df)} janelas")
    else:
        print("\nTodas as classes têm amostras suficientes.")

    # SEPARAÇÃO DE FEATURES
    metadata_cols = ['dateTime', 'calfId', 'calf_id', 'segId', 'acc_x', 'acc_y', 'acc_z', 'label']
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    X = df[feature_cols].copy()
    y_raw = df['label']
    X.fillna(0, inplace=True)
    
    # PREPARAÇÃO DOS LABELS
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_
    print(f"Classes finais ({len(class_names)}): {class_names}")

    # SPLIT DE DADOS
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"⚔️ Treino: {X_train.shape[0]} janelas | Teste: {X_test.shape[0]} janelas")
    
    # CONFIGURAÇÃO DA RANDOM FOREST
    rf = RandomForestClassifier(
        n_estimators=300,              
        criterion='entropy',           
        max_depth=None,                
        min_samples_split=5,           
        min_samples_leaf=2,            
        max_features='sqrt',           
        bootstrap=True,
        class_weight='balanced_subsample', 
        n_jobs=-1,                     
        random_state=24,
        verbose=1
    )
    
    rf.fit(X_train, y_train)
    
    # AVALIAÇÃO 
    y_pred = rf.predict(X_test)
    
    # Métricas Principais
    acc = accuracy_score(y_test, y_pred)
    print("\n" + "="*50)
    print(f"ACURÁCIA FINAL: {acc:.2%}")
    print("="*50)
    
    # Relatório por Classe
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusão Normalizada (Acurácia: {acc:.2%})')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # FEATURE IMPORTANCE
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nTop 10 melhores features:")
    for f in range(min(10, len(feature_cols))):
        print(f"{f+1}. {feature_cols[indices[f]]} ({importances[indices[f]]:.4f})")

train_best_random_forest(PARQUET_PATH)