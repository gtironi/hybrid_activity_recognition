# Roadmap de implementação — pré-treino e ablações

Documento operacional derivado de **`PLAN_PRETRAINING_ABLATIONS.md`** (contexto, política **C=1**, baselines, fairness). Este ficheiro é o **plano por fases** com entregáveis concretos e ordem de trabalho.

---

## Objetivo global

1. Entender e documentar cada dataset (séries, canais, ruído).
2. Fixar **formato padrão** + pré-processamento reprodutível.
3. **Smoke runs** com PatchTST (HF), PatchTSMixer (HF) e TFC (original), escolhendo dataset por CLI.
4. **Re-explorar** os mesmos gráficos da Fase 1 em cima dos dados **já transformados** (formato padrão), para comparar com bruto e apanhar erros de pré-processamento.
5. Depois, **pipeline modular** para experiências completas (métricas, runs, t-SNE de embeddings, etc.).

---

## Fase 0 — Ambiente e inventário (curta)

| Tarefa | Notas |
|--------|--------|
| Listar datasets “oficiais” do roadmap | Tabela abaixo; atualizar caminhos se mudarem. |
| `requirements` ou `conda` para Fase 2 | `torch`, `transformers` (versão com PatchTST + PatchTSMixer), `pandas`, `pyarrow`, `matplotlib`, `scikit-learn`, `umap-learn` (opcional, alternativa ao t-SNE). |
| Convenção de canal 0 | Documentar no `dataset_registry.yaml` (ou JSON): qual coluna/tensor index = canal usado na grelha **C=1**. |

### Registry inicial (ajustar paths ao teu disco)

| ID dataset | Uso | Fonte bruta / processada (exemplo no repo) |
|------------|-----|-----------------------------------------------|
| `har_tfc` | IMU, classificação, `.pt` TFC | `data/processed/HAR/` (`train.pt`, …) |
| `actbecalf_windowed` | IMU vaca, parquet | `dataset/processed/AcTBeCalf/windowed_*.parquet` |
| `actbecalf_long` | Exploração longitudinal | `dataset/processed/AcTBeCalf/test.parquet` |
| `dog` | IMU cão | `data/dog/*.csv` |
| `etth` / `ettm` | Sanidade PatchTST/PatchTSMixer (forecast) | `data/etth/ETTm1.csv`, etc. |

---

## Fase 1 — Exploração e visualização

### 1.1 Séries temporais: um subplot por canal

**Objetivo:** ver forma de onda, escala, ruído e possíveis artefactos, **sem** treinar modelo.

**Entregável:** script único, por exemplo `pretreino_abalations/exploracao/plot_timeseries_channels.py`.

**Porque “raw” primeiro, e janelas à parte**

- **Plot raw (recomendado por defeito):** um **segmento contínuo** do sinal por canal (CSV/parquet longo: primeiras `N` linhas, intervalo de tempo, ou um `segId`/`calf_id` inteiro). Isto responde “como é que X/Y/Z se comportam no tempo real?” — contexto, deriva, picos, buracos. É o que faz mais sentido para **explorar dados**.
- **Plot por janela (opcional, `--mode windows`):** útil quando a fonte já é **amostras fixas** (`.pt` com shape `(N,C,T)`, ou parquet windowed com vetores `acc_*`). Mostra **o que o modelo vai ver** numa única amostra; não substitui o raw para entender a série global.

**Comportamento sugerido**

- **CLI:** `--dataset <id>` (registry); modo **`--mode raw`** (default) ou **`--mode windows`**.
  - Raw: `--max_rows` ou `--duration_s` ou `--start/--end` (timestamp); opcional `--group_id` (ex. um `calf_id`) para um trecho contíguo por entidade.
  - Windows: `--max_plots 5` (número de amostras/janelas a desenhar); seed para reprodutibilidade.
- **Subplots:** um painel por **canal físico** (accX, accY, accZ, …); eixo horizontal = tempo ou índice de amostra; título com nome do canal + unidades se existirem.

**Fontes heterogéneas**

- **Parquet longo / CSV:** ler colunas de canal + tempo; plot raw direto (após ordenar por tempo).
- **`.pt`:** em modo raw **não existe** série única — ou plotas **várias janelas empilhadas** como traços curtos, ou concatenas trechos se tiveres metadata; o mais simples é usar **modo windows** aqui (cada `samples[i]` é um subplot multi-canal no tempo relativo 0…T−1).
- **Parquet windowed:** modo windows = uma linha → vetores `acc_x` …; modo raw só se reconstruíres a série a partir do longo ou ignorares este ficheiro para o plot contínuo.

**Saída:** `pretreino_abalations/exploracao/figures/<dataset_id>_channels_<mode>_<timestamp>.png` (+ opcional PDF).

### 1.2 t-SNE (apenas datasets IMU)

**Objetivo:** ver se janelas **com rótulo** formam estrutura em 2D quando usas um vetor de features **simples** (não é ainda o embedding do transformer).

**Entregável:** `pretreino_abalations/exploracao/tsne_imu_windows.py`.

**Feature vector sugerida (rápida e estável):**

- Por janela: concatena **estatísticas por canal** (mean, std, min, max) ou **média temporal do sinal** subamostrada (downsample) para vetor de dimensão fixa.
- Ou: flatten da janela **univariada** (canal 0 só) alinhado com política C=1, com PCA opcional antes do t-SNE se `T` for grande.

**CLI:** `--dataset <id>`, `--max_points 5000`, `--perplexity 30`, `--seed 42`.

**Saída:** scatter cor = classe (ou comportamento); legenda com `label2id`; guardar PNG + CSV com coordenadas e índices.

**Nota:** ETTh pode ficar **fora** deste script IMU ou modo `--task imu_only` que filtra o registry.

**Depois do export (Fase 2):** repetir estes plots sobre o `.pt` transformado — ver **Fase 3**.

---

## Fase 2 — Formato padrão + scripts simples (não modulares)

### 2.1 Fechar especificação do artefacto padrão

Antes de codificar export em massa, documentar num ficheiro curto, por exemplo **`pretreino_abalations/docs/FORMATO_PADRAO.md`**:

- Estrutura do dict em `.pt`: `samples` (`float32`, shape `(N, C_raw, T)` ou direto `(N, 1, T)` se já C=1), `labels` (`int64`), opcional `groups`, `meta`.
- `label2id.json`, `splits/` (ids ou índices), `TSlength_aligned` / `context_length` por dataset.
- Regra **C=1** aplicada na exportação **ou** na entrada dos runners (uma só escolha; recomenda-se aplicar na exportação para HF e documentar para TFC).

### 2.2 Pré-processamento por dataset

**Entregável:** um módulo ou scripts finos por fonte, todos escrevendo o **mesmo contrato**, por exemplo:

- `pretreino_abalations/preprocess/export_har_tfc.py`
- `pretreino_abalations/preprocess/export_actbecalf.py`
- `pretreino_abalations/preprocess/export_dog.py`
- `pretreino_abalations/preprocess/export_etth_slice.py` (só sanidade / forecast)

Cada um: lê dados brutos ou já processados, aplica janelas se necessário, **split** (train/val/test) com política anti-leakage quando houver `calf_id` / sujeito, grava `.pt` + JSON lateral.

### 2.3 Três runners CLI (`--dataset <id>`)

Local sugerido: `pretreino_abalations/smoke/`.

| Script | Responsabilidade |
|--------|------------------|
| `run_patchtst_hf.py` | Carrega dados do registry; treino curto ou 1 epoch **PatchTST** (pretrain MAE ou classificação se houver labels); log + checkpoint opcional. |
| `run_patchtsmixer_hf.py` | Idem para **PatchTSMixer** (`PatchTSMixerForPretraining` ou `ForTimeSeriesClassification` conforme labels). |
| `run_tfc_original.py` | `subprocess` ou `sys.path` para `TFC-pretraining/code/TFC/main.py` com `--pretrain_dataset` / paths apontando para cópia ou symlink dos `.pt` no layout que o TFC espera (`datasets/<Name>/train.pt`). |

**Contrato CLI comum:**

```text
python run_patchtst_hf.py --dataset har_tfc [--epochs 2 --device cuda]
```

**Implementação prática do TFC:** o upstream espera pastas tipo `datasets/SleepEEG`. Para `har_tfc`, ou (a) geres `pretreino_abalations/data_tfc_layout/HAR/{train,val,test}.pt` e apontas o working dir, ou (b) adaptas só o **path** no teu wrapper sem editar o núcleo do TFC.

**Critério de sucesso Fase 2:** para cada dataset IMU no registry, pelo menos um run completa sem excepção e escreve log; ETTh valida HF separadamente.

---

## Fase 3 — Exploração sobre dados transformados (QA / comparação com Fase 1)

**Objetivo:** garantir que o pré-processamento (Fase 2) **não distorceu** nem quebrou o sinal: repetir a **mesma lógica de visualização** da Fase 1, mas lendo **apenas** os artefactos no **formato padrão** (`.pt` exportados, metadados alinhados com `FORMATO_PADRAO.md`).

**Porquê uma fase à parte**

- A Fase 1 olha para **fontes brutas ou semi-processadas** (parquet longo, CSV, `.pt` legado).
- Depois da Fase 2 existem **novos** tensores (normalização, crop temporal, C=1, reordenação de eixos, filtros). Pequenos bugs (escala errada, eixo T/C trocado, labels desalinhados) aparecem aqui — comparar lado a lado com as figuras da Fase 1 poupa semanas de treino estranho.

**Entregáveis**

| Entregável | Notas |
|------------|--------|
| **Reutilizar ou parametrizar** `plot_timeseries_channels.py` | Novo modo ou flag, por exemplo `--source transformed` + `--pt_path` / `--dataset <id>_export` no registry que aponta para `pretreino_abalations/processed/<dataset>/train.pt`. Mesmo contrato de **subplots por canal** e **janelas com cores distintas** (`--max_windows`). |
| **Reutilizar ou parametrizar** `tsne_imu_windows.py` | Idem: ler janelas + labels do `.pt` transformado; comparar scatter com o da Fase 1 (mesmo `seed`, mesmo `max_points`). |
| **Relatório curto opcional** | `pretreino_abalations/exploracao/QA_TRANSFORMED.md` ou pasta `figures/compare_phase1_vs_phase3/` com pares raw vs transformed lado a lado (mesmo índice de janela quando aplicável). |

**Checklist mínima antes de avançar**

- Shapes e `dtype` batem com a spec; `min`/`max`/`nan` por canal nas primeiras `K` janelas.
- Distribuição global não mudou de ordem de grandeza sem explicação (ex.: normalização Z-score documentada).
- Contagem de classes e histograma de labels **idênticos** em espírito ao esperado (salvo mudança documentada de `label2id`).
- Se usaste **C=1** na exportação, plots da Fase 3 mostram **só** esse canal; anotar na legenda.

**Pré-condição:** exports da Fase 2 existem para o dataset (pelo menos `train.pt`).

**Critério de sucesso:** figuras transformadas **interpretáveis** e, quando comparáveis, **consistentes** com a Fase 1; qualquer divergência explicada (ex.: só existe série windowed após export).

---

## Fase 4 — Pipeline modular (experiências)

**Pré-condição:** Fases 2 **e** 3 OK nos datasets alvo (smoke + QA visual).

**Entregável:** pacote tipo `pretreino_abalations/src/pretreino_ablations/` (nome final à tua escolha) com:

- Registry de datasets + loaders únicos.
- Encoders / métodos de pré-treino / cabeças / modos `freeze_encoder` vs `full_finetune`.
- Um comando tipo `pretreino-experiment --config configs/foo.yaml` que cria **`runs/<run_id>/`** com `config.yaml`, métricas, matriz de confusão, t-SNE de embeddings (pré-treino e pós fine-tune), checkpoints.

Detalhe de métricas e estrutura de pastas: ver **`PLAN_PRETRAINING_ABLATIONS.md`** (secções de eval e per-run layout).

---

## Sugestões extra (exploração / visualização / QA)

Estas não substituem a Fase 1 nem a **Fase 3**; complementam decisões de pré-processamento.

1. **Histogramas / KDE** por canal e por classe — detectar shift de distribuição e classes raras.
2. **PSD (Welch)** por canal em poucas janelas por classe — ruído de banda vs movimento periódico.
3. **Espectrograma (STFT)** de 2–3 janelas exemplo — comparar HAR vs vaca vs cão.
4. **NaNs, outliers, saturação** — contagem por coluna; plot de valores max |acc| ao longo do tempo.
5. **Duração e cadência** — histograma de Δt entre linhas (parquet longo); confirmar Hz assumido vs real.
6. **Balanceamento de classes** — bar chart; se muito desequilibrado, planear **macro-F1** e possivelmente **class weights** só na fase modular.
7. **Group leakage check** — para AcTBeCalf: garantir que nenhum `calf_id` aparece em mais de um split; heatmap ou lista de violações.
8. **Correlação entre canais** (X,Y,Z) na mesma janela — mesmo com C=1 no treino, ajuda a explicar quando o sinal é “quase só ruído” num eixo.
9. **Sanity check numérico** — após export `.pt`, script que imprime `shape`, `dtype`, `min/max/mean` de `samples` e lista `unique(labels)`.

---

## Ordem de execução recomendada

1. Fase 0 → Fase 1 (plots + t-SNE IMU, dados brutos / fonte original).  
2. Fechar **FORMATO_PADRAO.md** + primeiro export (ex.: só `har_tfc`).  
3. Smoke PatchTST → Smoke PatchTSMixer → Smoke TFC no **mesmo** export.  
4. Repetir export para `actbecalf_windowed`, `dog`.  
5. **Fase 3:** voltar a correr exploração (plots + t-SNE) nos `.pt` transformados; comparar com Fase 1.  
6. **Fase 4** modular (experiências completas).

---

## Referência

- Plano estratégico e política de canais: **`pretreino_abalations/PLAN_PRETRAINING_ABLATIONS.md`**
- Código TFC local: **`TFC-pretraining/code/TFC/`**
