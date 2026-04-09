import pandas as pd
PARQUET_PATH = 'WindowedCalf.parquet'

def inspect_dataset(file_path):
    # Lê o arquivo
    df = pd.read_parquet(file_path)
    
    print("\n" + "="*60)
    print("RESUMO GERAL DO DATASET")
    print("="*60)
    print(f"• Total de Janelas (Linhas): {len(df)}")
    print(f"• Total de Colunas:          {len(df.columns)}")
    print(f"• Colunas Disponíveis:       {df.columns.tolist()[:6]} ... [e mais {len(df.columns)-6} features]")
    
    # Verificação das Classes
    print("\n" + "-"*60)
    print("DISTRIBUIÇÃO DAS CLASSES")
    print("-" * 60)
    counts = df['label'].value_counts()
    for label, count in counts.items():
        percentage = (count / len(df)) * 100
        print(f"{label:<25} | {count:>6} janelas | {percentage:.2f}%")
    
    # Verificação de Integridade das Janelas (Arrays)
    print("\n" + "-"*60)
    print("INTEGRIDADE DAS JANELAS)")
    print("-" * 60)
    
    # Pega a primeira linha para teste
    sample_row = df.iloc[0]
    len_x = len(sample_row['acc_x'])
    len_y = len(sample_row['acc_y'])
    len_z = len(sample_row['acc_z'])
    
    print(f"• Tamanho do Array acc_x: {len_x} pontos ")
    print(f"• Tamanho do Array acc_y: {len_y} pontos ")
    print(f"• Tamanho do Array acc_z: {len_z} pontos ")
    print(f"• Tipo de dado nos arrays: {type(sample_row['acc_x'])}")
    
    # Verificação das Features TSFEL
    standard_cols = ['dateTime', 'calf_id', 'acc_x', 'acc_y', 'acc_z', 'label']
    tsfel_cols = [c for c in df.columns if c not in standard_cols]
    
    print("\n" + "-"*60)
    print(f"FEATURES TSFEL ({len(tsfel_cols)} detectadas)")
    print("-" * 60)
    if len(tsfel_cols) > 0:
        print(f"• Exemplo das primeiras 5: {tsfel_cols[:5]}")
        nans = df[tsfel_cols].isnull().sum().sum()
        print(f"• Total de NaNs nas features: {nans}")
        feat_sample = tsfel_cols[0]
        print(f"• Exemplo de valores para '{feat_sample}':")
    else:
        ...

    # Visualização de Exemplo
    print("\n" + "-"*60)
    print("VISUALIZAÇÃO DE 3 LINHAS ALEATÓRIAS")
    print("-" * 60)
    
    # Configuração para o Pandas não cortar o texto do array no print
    pd.set_option('display.max_colwidth', 60) 
    pd.set_option('display.max_columns', 10) # Mostra poucas colunas pra caber na tela
    
    sample_df = df.sample(3)
    cols_to_show = ['dateTime', 'label', 'acc_x'] + tsfel_cols[:2]
    
    print(sample_df[cols_to_show])

inspect_dataset(PARQUET_PATH)