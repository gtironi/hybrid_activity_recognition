# Plano: Interface de Dados Padronizada e Pipeline de Pre-processamento

## Prompt original do usuario

> Estava relendo o plano anterior e vi que me equivoquei. Criei aquele plano achando que o ramo
> deep learning estaria recebendo o parquet bruto diretamente, o que era estranho. Agora que sei
> que o janelamento do dataset e so uma decisao de implementacao, e que o modelo recebe exatamente
> xyz apenas, aquela complexidade toda nao faz mais sentido.
>
> Ainda faz sentido:
> - a motivacao de adaptar mais facilmente para novos datasets raw
> - separar melhor a responsabilidade entre dado bruto, janelamento, extracao de TSFEL e treino
> - ter uma base melhor para pre-treino do PatchTST em dados sem rotulo
>
> Mas toda a parte de mudar muito codigo apenas para usar o arquivo raw no lugar do janelado perdeu
> forca. Prefiro seguir com o janelado, melhorando a explicabilidade do pre-processamento que o
> gera. Deixando ele mais generalizavel para outros datasets e mais como parte do pipeline, ainda
> sendo um step anterior bem documentado.
>
> Quero ter uma interface de dados raw que, se eu colocar no pre-processamento, deve funcionar.
> Usando o seguinte padrao de colunas:
>
>   datetime, subject, features com nome padrao (sempre tera acc, gyr e mag, nao necessariamente
>   todos esses), label
>
> Qualquer coisa assim deve funcionar.
> Continue com o parquet janelado para os experimentos supervisionados comparativos.
> Padronize bem o script que gera essas janelas e o TSFEL.

---

## Decisao: manter o parquet janelado como formato de treino

O pipeline de treino supervisionado **nao muda**. Todos os experimentos continuam consumindo
parquets janelados com a mesma estrutura atual:

```
windowed_train.parquet
  colunas: calf_id, acc_x=[...], acc_y=[...], acc_z=[...], label, feat_1, ..., feat_K
```

O que muda e o **pre-processamento que gera esses parquets**: ele passa a aceitar qualquer
dataset bruto que siga uma convencao de colunas padrao, sem precisar de adaptacao manual
para cada nova fonte de dados.

---

## 1. Convencao de colunas para datasets brutos

Todo dataset bruto aceito pelo pipeline deve ter estas colunas obrigatorias:

| Coluna | Obrigatoria | Descricao |
|--------|-------------|-----------|
| `datetime` | Sim | Timestamp da amostra (qualquer formato parsavel pelo pandas) |
| `subject` | Sim | Identificador do sujeito (animal, pessoa, dispositivo) |
| `label` | Sim (treino) / Nao (pretreino) | Rotulo de comportamento/atividade |
| `segment_id` | Nao | Segmento continuo de gravacao; se ausente, e inferido por sujeito + continuidade temporal |
| `acc_x`, `acc_y`, `acc_z` | Condicional | Acelerometro (pelo menos um grupo de sensor deve existir) |
| `gyr_x`, `gyr_y`, `gyr_z` | Nao | Giroscopio (opcional) |
| `mag_x`, `mag_y`, `mag_z` | Nao | Magnetometro (opcional) |

### Regra de nomes dos sensores

Colunas de sensor seguem o padrao `{tipo}_{eixo}`:
- Tipos aceitos: `acc`, `gyr`, `mag`
- Eixos aceitos: `x`, `y`, `z`

O pipeline detecta automaticamente quais canais existem e monta o tensor de sinal `(C, T)` onde
`C` e o numero de eixos presentes. Se so existir acelerometro, `C=3`. Se existir acelerometro
e giroscopio, `C=6`.

### Mapeamento para o dataset atual (AcTBeCalf)

| Coluna atual | Coluna padrao |
|-------------|---------------|
| `dateTime` | `datetime` |
| `calfId` | `subject` |
| `accX` | `acc_x` |
| `accY` | `acc_y` |
| `accZ` | `acc_z` |
| `behaviour` | `label` |
| `segId` | `segment_id` |

O script `dataset_processing.py` recebe um argumento `--column-map` ou le um JSON de
mapeamento para renomear colunas antes de qualquer processamento. Para o AcTBeCalf, esse
mapeamento ja esta embutido como default.

---

## 2. Pipeline de pre-processamento padronizado

```
Dataset bruto (CSV ou Parquet)
    ↓
dataset_processing.py
    --column-map {mapeamento de colunas}
    --subject-col subject
    --test-subjects [lista]
    → train.parquet / test.parquet   (formato padrao, sem janelamento)

    ↓
prepare_windowed_parquet.py
    --window-size 75
    --stride 37
    --purity-threshold 0.9
    --label-map {mapeamento de rotulos}  (ex: 50 rotulos → 19 classes)
    --mode discover|apply
    → windowed_train.parquet / windowed_test.parquet
    → tsfel_manifest.json

    ↓
Treino supervisionado (sem mudancas)
```

### Responsabilidades de cada script

#### `dataset_processing.py` (existente, generalizar)

Responsabilidade unica: split por sujeito sem vazamento.

Mudancas necessarias:
- Aceitar `--column-map '{"dateTime":"datetime","calfId":"subject",...}'` (JSON inline ou path)
- Padronizar saida: sempre `datetime, subject, segment_id, acc_x, acc_y, acc_z, [gyr_*, mag_*], label`
- Remover dependencias hard-coded de nomes de colunas do AcTBeCalf

#### `prepare_windowed_parquet.py` (existente, generalizar)

Responsabilidade unica: janelamento + extracao TSFEL.

Mudancas necessarias:
- Detectar colunas de sinal automaticamente via padrao `{tipo}_{eixo}` em vez de `acc_x` hard-coded
- Aceitar `--sensor-cols` como override explicito se necessario
- Garantir que o manifest JSON salva quais colunas de sinal foram usadas (para reproducibilidade)
- Modo `discover`: extrai e seleciona features TSFEL do treino, salva manifest
- Modo `apply`: usa manifest para aplicar as mesmas features ao teste

### Novo utilitario: `src/.../utils/schema.py`

Funcoes puras para lidar com a convencao de colunas:

```python
def detect_signal_cols(df: pd.DataFrame) -> list[str]:
    """Retorna colunas de sinal no padrao {tipo}_{eixo}, ordenadas por tipo e eixo."""
    tipos = ("acc", "gyr", "mag")
    eixos = ("x", "y", "z")
    return [f"{t}_{e}" for t in tipos for e in eixos if f"{t}_{e}" in df.columns]

def validate_schema(df: pd.DataFrame) -> None:
    """Lanca ValueError se colunas obrigatorias estiverem ausentes."""
    required = {"datetime", "subject"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")
    if not detect_signal_cols(df):
        raise ValueError("Nenhuma coluna de sinal encontrada (esperado acc_x, acc_y, acc_z, ...).")

def apply_column_map(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Renomeia colunas conforme mapeamento antes de qualquer processamento."""
    return df.rename(columns=mapping)
```

---

## 3. Formato de saida: parquet janelado padronizado

O parquet janelado de saida mantem o mesmo formato atual, porem com nomes de colunas canonicos:

| Coluna | Tipo | Descricao |
|--------|------|-----------|
| `subject` | str/int | Identificador do sujeito |
| `segment_id` | int | Segmento de origem |
| `window_start` | int | Indice de inicio da janela no segmento |
| `label` | str | Rotulo mapeado (pos label_map) |
| `acc_x` | list[float] | Sinal X da janela (window_size valores) |
| `acc_y` | list[float] | Sinal Y da janela |
| `acc_z` | list[float] | Sinal Z da janela |
| `gyr_x`, ...  | list[float] | Outros canais se existirem |
| `feat_*` | float | Features TSFEL selecionadas (K colunas) |

O `dataloader.py` usa `detect_signal_cols()` para montar o tensor `(C, T)` automaticamente,
sem assumir que sempre serao exatamente `acc_x, acc_y, acc_z`.

---

## 4. Dado de pretreino para PatchTST

O pretreino do PatchTST usa dado bruto sem rotulo. Ele segue a mesma convencao de colunas:

```
datetime, subject, [segment_id], acc_x, acc_y, acc_z
```

Sem coluna `label`. Se `label` existir, e ignorada.

O script `prepare_windowed_parquet.py` com flag `--no-label` gera um parquet janelado de
sinais apenas (sem features TSFEL, sem rotulo), que e usado pelo `PretrainWindowDataset`.

Isso mantem a separacao de responsabilidades: o janelamento continua sendo feito offline
(pre-processamento), nao dentro do Dataset em tempo de treino.

---

## 5. O que muda vs. o que fica igual

### Fica igual

- Formato dos parquets janelados de treino/teste/val
- `CalfHybridDataset`, `dataloader.py`, `Trainer`, todos os modelos
- Todos os experimentos comparativos do `PLAN_experiments_patchtst.md`
- O script `.sh` de experimentos

### Muda (generalizacao dos scripts de pre-processamento)

| Script | O que muda |
|--------|-----------|
| `dataset_processing.py` | Aceita `--column-map`; valida schema padrao; saida com nomes canonicos |
| `prepare_windowed_parquet.py` | Detecta colunas de sinal automaticamente; salva lista de canais no manifest; aceita `--no-label` para pretreino |
| `dataloader.py` | Usa `detect_signal_cols()` para montar tensor em vez de `acc_x/y/z` hard-coded |

### Novo arquivo

| Arquivo | Conteudo |
|---------|----------|
| `src/.../utils/schema.py` | `detect_signal_cols()`, `validate_schema()`, `apply_column_map()` |

---

## 6. Exemplo: usar um novo dataset

Para usar um dataset diferente do AcTBeCalf (ex: dataset de humanos com acelerometro+giroscopio):

```bash
# 1. Mapear colunas para o padrao
python scripts/dataset_processing.py \
    --input    novo_dataset.csv \
    --column-map '{"time":"datetime","pid":"subject","ax":"acc_x","ay":"acc_y","az":"acc_z","gx":"gyr_x","gy":"gyr_y","gz":"gyr_z","activity":"label"}' \
    --test-subjects P01 P02 \
    --output-dir dataset/processed/novo_dataset/

# 2. Janelar + extrair TSFEL
python scripts/prepare_windowed_parquet.py \
    --input   dataset/processed/novo_dataset/train.parquet \
    --mode    discover \
    --output  dataset/processed/novo_dataset/windowed_train.parquet \
    --manifest dataset/processed/novo_dataset/tsfel_manifest.json

# 3. Treinar — identico ao AcTBeCalf
PYTHONPATH=src python -m hybrid_activity_recognition.main \
    --mode supervised \
    --model robust \
    --input_mode hybrid \
    --labeled_parquet_train dataset/processed/novo_dataset/windowed_train.parquet \
    ...
```

O passo 3 nao muda absolutamente nada em relacao ao AcTBeCalf.

---

## 7. Checklist de implementacao

- [ ] `src/.../utils/schema.py` com `detect_signal_cols()`, `validate_schema()`, `apply_column_map()`
- [ ] `dataset_processing.py`: aceitar `--column-map`, validar schema, saida com nomes canonicos
- [ ] `prepare_windowed_parquet.py`: usar `detect_signal_cols()`, salvar canais no manifest, aceitar `--no-label`
- [ ] `dataloader.py`: usar `detect_signal_cols()` para construir tensor de sinal
- [ ] Testar com AcTBeCalf com mapeamento explicito (resultado deve ser identico ao atual)
- [ ] Testar modo `--no-label` para gerar parquet de pretreino

---

## 8. Referencias

- Barandas, M. et al. (2020). "TSFEL: Time Series Feature Extraction Library." *SoftwareX*.
- Dehghani, A. et al. (2019). "A Quantitative Comparison of Overlapping and Non-Overlapping
  Sliding Windows for Human Activity Recognition." *Sensors*, 19(22), 5026.
- Ordonez, F.J. & Roggen, D. (2016). "Deep Convolutional and LSTM Recurrent Neural Networks
  for Multimodal Wearable Activity Recognition." *Sensors*, 16(1), 115.
