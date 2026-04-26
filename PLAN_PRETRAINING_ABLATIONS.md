# Planning document: modular pretraining ablations for animal behavior (IMU)

## Original prompt (as sent)

> ok, I wanna run an experiment, you will help me to think and to plan it.
>
> Estou explorando pretrieno para aprendizado de representacoes para a task de reconhecimento de comportamento em animais. A ideia inicial era usar o PatchTST para ver os resultados. Quando coloquei para todar (tanto no repo oficial quanto na versao do hugging face) ficou claro que o pretreino simplesmente nao aprende, ficndo perto de 1 sempre.
>
> Tentei mudar hiperparametro, modelo, dados... nada funcionou muito bem. Entao testei com um outro dado `data/etth/ETTm1.csv`, que foi usado no paper, e ele funcionou super bem. Entao, alguams coisas podem ser a causa disso. Pode ser que o dado tenha muito ruido, pode ser que a serie temporal nao seja trivial e precise mudar algo na arquiteura...
>
> para testar, quero fazer algumas coisas, para alem do PatchTST.
>
> Existem alguns modelos que fazem o que quero:
>
> - TFC (<https://github.com/mims-harvard/tfc-pretraining>)
> - Patch TST (<https://huggingface.co/docs/transformers/model_doc/patchtst#transformers.PatchTSTForClassification>)
> - Patch TSMixer (<https://huggingface.co/blog/patchtsmixer>)
>
> E separei alguns dados. `data/dog` esses sao alguns parecidos com o meu e que quero testar. `dataset` esses tambem sao meus.
>
> Ai temos um parecido com esses dados, tambem de IMU, que foi usado pelo TFC e parece ter sido bom, que é o `data/HAR` UCI cujo dado processado pelo TFC esta em `data/processed/HAR`. Adicione o repo com a implementacao do TFC original aqui: `TFC-pretraining`
>
> Voce vai ver as instrucoes de como rodar em: `TFC-pretraining/README.md`
>
> Voce vai ver que ele tambem tem alguns outros modelos legais: `TFC-pretraining/code/baselines`
>
> A implementacao do TFC mesmo fica aqui: `TFC-pretraining/code/TFC`
>
> Alem desses datasets de IMU, separei o `data/etth` que e usado no paper do PatchTST e do PatchTSMixer, entoa e garantido que ele funciona nesses modelos. Para termos como base, mas nao conseguimos fazer classificao neles.
>
> O que eu qeuro fazer é o seguinte. Seguindo como foi feito em `src/`, mas agora em uma pasta totalemnte nova, chamada pretreino_abalations, quero criar uma arquitetura modular para testar os diferentes tipos de modelos, loss e tudo mais.
>
> Basicamente, como voce pode ver, temos algumas partes.
>
> O encoder, no qual temos como possibilidades, por exemplo, uma resnet padrao, ou um encoder convencional, como usado no TFC. Ou por exemplo, o patchTST e o PatchTSMixer, que sao arquiteturas de encoder.
>
> Ai no metodo de aprendizado, vamos ter por exemlo, simclr padrao com usa loss, TFC (o TFC ;e um metodo de aprendizado, nao um encoder) e outros tipos, como no baseline do TFC deixa claro.
>
> E no final, ainda, vamos ter um forma de tranformar a representacao aprendida em classificacao, aqui podemso fazer de varias formas. varias cabecas de classificacao difernetes.
>
> Quero uma pipeline padrao que supotes os testes com todos os modelos que mencionei e os datasets.
>
> para isso, vamos precisar entender qual o formato de dataset processado que vamso poder usar par atodos, como configurar as abstracoes do pipeline, como salvar resultados, como rodar...
>
> Vale mencionar que eu gostaria de saber, f1-score, acucaracya da classificao (quando tem), matriz de confusao, mas tambem um sne da represtacao aprendida logo aos o pretreino e depois do classificador final (considerando que vamos mduar o peso do encocer). Quero tudo dentro de uma folder para cada run.
>
> Gostari de ter a opcao de congelaro encoder e mduar so o peso na head de classificacao ou fazer o full fine tuning. (deve ser um arg na hora de rodar)
>
> Sua funcao agora é me falar o que acha dessa ideia, possiveix porblema e o que ta faltando.
>
> Mais importante ainda, olhe os respos el inks que te mandar, para comecar a planejar como podemos fazer isso. algumas pergutnas relevantes ainda, alem do obvio e das ja mencionadas sao:
>
> - Quais encoder devriamos usar para testar?
> - Quais loss?
> - Algo alem de full finetuning e congelar encoder?
> - Devemos usar o mesmo dado para tudo ou ter diferentes versoes?
> - O que fazer para garantir que a comparacao é jsuta?
> - Como isolar bem o test set nas diferrentes runs?

---

## Verdict on the idea

The plan is **sound and worth doing**. You already did the right diagnostic: **PatchTST-style pretraining “works” on ETTh/ETTm-style data but collapses or stalls on your IMU behavior windows**, which strongly suggests a **domain / scaling / objective mismatch**, not necessarily a broken implementation.

A **single modular package** with swappable **encoder × pretraining objective × downstream head × finetuning mode** is the right way to separate “which representation works for noisy animal IMU?” from “which codebase happened to train.”

**Main risks** (manageable with discipline):

1. **Combinatorial explosion**: encoders × losses × heads × datasets × seeds. You need a **small core grid** first, then expand.
2. **Apples-to-oranges**: TFC uses **two encoders + TF-C loss**; SimCLR uses **one encoder + NT-Xent**; PatchTST pretraining is often **masked reconstruction**. They are not the same compute or inductive bias—fairness is about **protocol**, not identical loss.
3. **Input conventions**: Hugging Face PatchTST/PatchTSMixer typically expect **`(B, T, C)`** (`past_values`); TFC-style code and your HAR `.pt` use **`(B, C, T)`**. A **single internal tensor layout** (choose one) with thin adapters avoids silent bugs.
4. **TFC “method” vs “encoder”**: TF-C is a **training objective** tying **time and frequency** branches. Porting it on top of arbitrary encoders (e.g. PatchTST) is research-grade work; for v1, treat **TFC as its own module** (reuse `TFC-pretraining/code/TFC` logic) rather than forcing every encoder into TF-C day one.

---

## What is still missing (before coding)

- **Canonical dataset spec**: one **documented** contract (shapes, splits, label map, subject/group id for leakage checks). All loaders emit the same `Batch` object.
- **Explicit “sanity run” suite**: ETTh forecasting/MAE (already known good) + HAR supervised baseline + your smallest dog split—to catch pipeline bugs in minutes.
- **Pretraining success metrics beyond loss**: e.g. **linear probe on train-val** during pretrain, **kNN on frozen embeddings**, or **CKA** between two runs—so you detect collapse early (loss ≈ constant).
- **Compute budget table**: max steps/epoch per method so you do not compare a 10-epoch SimCLR to a 200-epoch TFC run without noting it.
- **Repro manifest**: `config.yaml` + git commit + data hash (or split file path) + seed **copied into every run directory** (you asked for one folder per run—store the manifest there).

---

## Unified processed format (recommendation)

**Target tensor for all IMU classification experiments:**

- `x`: `float32` tensor **`[N, T, C]`** (match HF time-series models) *or* `[N, C, T]` if you standardize on TFC—pick **one** and stick to it in code with `permute`. Para a **grelha comparativa principal**, fixar **`C=1`** (ver **Política de canais** abaixo).
- `y`: `int64` class index.
- `group` / `subject_id` / `calf_id` (when available): **integer or string id** for **group-aware splits** (see test isolation).

**Storage:**

- **Training speed**: `.pt` or **memory-mapped** `.npy` for windows; keep **Parquet** as **provenance** + optional feature columns.
- **Metadata sidecar**: JSON with `label2id`, `sampling_rate`, `window_length`, `stride`, `split_policy`, `hash`.

**ETTh/ETTm:**

- Use as **pretraining / forecasting sanity** only (as you said, not classification). Optionally **multivariate forecasting** or **channel-independent MAE** to validate PatchTST/PatchTSMixer wiring—not to compare directly to behavior F1.

---

## Política de canais (decisão fixa do projeto)

**Manter um único canal em todos os métodos da grelha principal**, alinhado com `TFC-pretraining/code/TFC/dataloader.py`: após `(N, C, T)`, usar **`X[:, :1, :L]`** — só o **canal 0** entra no pré-treino, fine-tune e avaliação comparáveis. **Não alterar** arquiteturas nem losses dos métodos para suportar multi-canal nesta fase.

**O que isto obriga na prática**

1. **Definir e documentar o eixo físico do canal 0** em todos os datasets (ex.: ordem fixa `accX, accY, accZ` nos tensores e **canal 0 = accX** — ou outra convenção, mas **uma só** para HAR, AcTBeCalf, dog, etc.).
2. **Normalizar antes dos modelos**: ou gravar `.pt` já com `C=1`, ou aplicar `x = x[:, :1, :]` num único sítio no loader partilhado (evita runs acidentais com `C=3` misturados na mesma tabela de resultados).
3. **PatchTST / PatchTSMixer (Hugging Face)**: `num_input_channels=1`, `past_values` com shape **`(B, T, 1)`** nas experiências desta política.
4. **Manifesto por run** (`config.yaml` ou equivalente): campos explícitos, por exemplo `num_input_channels: 1`, `channel_policy: first_only`, `channel0_physical_axis: accX` (ajustar ao teu naming).

**Trade-off aceite**

- Perde-se acoplamento **entre eixos** no encoder (não há Y/Z na mesma frente). Comportamentos que dependem fortemente da relação triaxial podem ficar mais difíceis; isso é uma **escolha metodológica** documentada, não um bug da pipeline.

**Fora de âmbito (outra grelha)**

- Qualquer experiência futura com `C>1` deve ser **outra família de runs** (novo `channel_policy`), nunca misturada silenciosamente com esta.

---

## Modular pipeline (conceptual)

Mirror `src/hybrid_activity_recognition/` patterns (`trainer`, `dataloader`, `models`) but stricter interfaces:

| Component | Responsibility |
|-----------|----------------|
| `datasets/` | Load canonical windows; **no** model-specific quirks beyond padding/masking. |
| `encoders/` | Map `x` → representation `z` (vector or sequence pooling). |
| `pretext/` | Self-supervised head + **loss** (SimCLR, MAE, TF-C, TS-TCC, …). |
| `downstream/` | Classification head(s); optional **prototype / cosine** head. |
| `finetune/` | `freeze_encoder`, LR schedules, layer-wise LR, gradual unfreeze. |
| `eval/` | Accuracy, macro/micro F1, confusion matrix, **t-SNE/UMAP** (with seed), calibration if needed. |

**TFC repo map (local):** `TFC-pretraining/code/TFC` (method), `TFC-pretraining/code/baselines/` (SimCLR, TS-TCC, TS2Vec, Mixing-up, CLOCS—different objectives and assumptions).

**Hugging Face:**

- [PatchTST docs](https://huggingface.co/docs/transformers/model_doc/patchtst): `PatchTSTForPretraining`, `PatchTSTForClassification`, `PatchTSTModel` (encoder-only).
- [PatchTSMixer blog](https://huggingface.co/blog/patchtsmixer): forecasting pretrain; analogous classification heads exist in Transformers—check your installed `transformers` version for `PatchTSMixerFor*` classes.

---

## Which encoders to test (prioritized)

**Tier A — must-have for IMU behavior:**

1. **Lightweight CNN / TFC-style temporal encoder** (from TFC stack)—strong inductive bias for smooth IMU.
2. **ResNet-1D / MSDN-style** (if you already have it in-repo)—cheap baseline.
3. **PatchTST backbone** (`PatchTSTModel`) with **MAE** and optionally **contrastive** if you add a projector (not always in HF).
4. **PatchTSMixer backbone**—often **more stable / cheaper** than full Transformer on short windows.

**Tier B — after A is stable:**

5. **TS2Vec-style dilated CNN encoder** (from `baselines/TS2vec`) if you adopt their hierarchical loss.
6. **Transformer encoder** (non-patch) only if window length grows (your animal windows may stay short; patching may be unnecessary).

**Not necessarily “one encoder for all objectives”:** TF-C wants **time + frequency** pathways; SimCLR wants **two views**. The modular design should allow **encoder(s) per method**, not force PatchTST through TF-C on day one.

---

## Which losses / pretraining methods

Map methods to **families** (for fair reporting):

| Family | Examples | Notes |
|--------|----------|--------|
| **Contrastive (instance)** | SimCLR, InfoNCE / NT-XEnt | Needs **strong augmentations** tuned for IMU (jitter, scaling, time-shift, channel dropout). |
| **Temporal consistency** | TS-TCC (in TFC baselines) | Uses **weak/strong** augmentations; different from SimCLR. |
| **Masked reconstruction** | PatchTST `PatchTSTForPretraining` | Your ETTh success suggests the **objective is fine in general**; IMU may need **normalization**, **patch length**, or **mask ratio** changes. |
| **Time–frequency consistency** | **TFC (TF-C)** | Method-specific; two branches + alignment loss. |
| **Hierarchical / subseries** | TS2Vec | Multi-scale contrastive. |

**Practical suggestion:** start with **(1) MAE/PatchTST pretrain**, **(2) SimCLR with IMU augmentations**, **(3) TFC** on **HAR only** to validate port, then move to **dog / AcTBeCalf**.

---

## Beyond “freeze encoder” vs “full finetune”

Add these **explicit modes** (CLI flags):

1. **Linear probe**: encoder frozen; train single linear (or small MLP) — measures **representation quality**.
2. **Partial unfreeze**: last *k* blocks / last layer norm only (common for Transformers).
3. **Discriminative LR**: **lower LR** on encoder, **higher** on head (often better than binary freeze/full).
4. **Gradual unfreeze**: freeze → unfreeze blocks over epochs.
5. **Head-only then full** (two-stage): cheap exploratory schedule recommended in many transfer papers.

These are **standard** and often outperform a single “full finetune from scratch on head” experiment.

---

## Same data for everything vs different versions

- **One canonical preprocessing** per dataset family (window length, normalization, filter). Store **version id** in the manifest.
- **Optional derived views** are fine (e.g. bandpass vs raw) but treat them as **separate dataset ids** in results tables—do not mix in one row without labeling.
- **ETTh/ETTm**: **sanity / pretrain for HF models**; **not** the same task as behavior classification—use a **separate benchmark section** in reports.

---

## Fair comparison checklist

1. **Same splits** across methods (same `test_indices.pt` or hashable split file).
2. **Same evaluation code** (macro F1 for imbalanced behavior labels).
3. **Same training steps** *or* report **compute** (epochs × batch × model FLOPs estimate)—if you match wall-clock poorly, note it.
4. **Same augmentation budget** where applicable (contrastive methods need more augment diversity than MAE).
5. **Early stopping on val** with **single test evaluation** at end—or fixed epoch budget with **multiple seeds**.
6. **Hyperparameter search policy**: either **fixed budget per method** or **small search per method**—document which.

---

## Test-set isolation (especially for animals)

**Goal:** no window from the same **animal/session/recording** appears in train and test.

- Use **`calf_id` / dog_id / session** when available: **group split** (GroupKFold or custom).
- For HAR UCI, follow TFC’s note: some datasets are **pre-shuffled**; still **fix a split file** and reuse everywhere.
- **Never** tune on test. **Val** for LR / early stopping; **test** once per finalized checkpoint (or report **nested** CV if data is tiny—expensive).

Persist: `splits/{dataset}/train.json`, `val.json`, `test.json` (window ids or `(subject, time range)`).

---

## Per-run output directory layout (suggested)

```
runs/
  {timestamp}_{dataset}_{encoder}_{method}_{finetune}_{seed}/
    config.yaml              # full hyperparams + CLI
    environment.txt          # pip freeze or conda env export
    metrics.json             # val/test acc, macro_f1, per-class f1
    confusion_matrix.png
    confusion_matrix_test.npy
    embeddings/
      post_pretrain_train.npy / .pt   # optional subsample
      post_pretrain_val.npy
      post_finetune_test.npy
    plots/
      tsne_post_pretrain.png
      tsne_post_finetune.png
    checkpoints/
      pretrain_best.pt
      finetune_best.pt
    logs/
      train.log
```

**t-SNE / UMAP:** subsample for clarity (e.g. 5k points), **fix seed**, **perplexity** tuned to sample size; color by **true label**; save **2–3 runs** with same hyperparameters to check stability.

---

## References (expanded)

| Resource | URL | What to use it for |
|----------|-----|-------------------|
| TF-C paper (NeurIPS 2022) | [arXiv:2206.08496](https://arxiv.org/abs/2206.08496) | Objective definition (time–frequency consistency), transfer settings, baseline claims. |
| TFC code + data | [github.com/mims-harvard/TFC-pretraining](https://github.com/mims-harvard/TFC-pretraining) | Local: `TFC-pretraining/`. |
| PatchTST (HF) | [huggingface.co/docs/transformers/model_doc/patchtst](https://huggingface.co/docs/transformers/model_doc/patchtst) | `PatchTSTModel`, `PatchTSTForPretraining`, `PatchTSTForClassification`; input `past_values` shape `(B, T, C)`. |
| PatchTST blog | [huggingface.co/blog/patchtst](https://huggingface.co/blog/patchtst) | Intuition, patching, channel independence. |
| PatchTSMixer (HF) | [huggingface.co/docs/transformers/model_doc/patchtsmixer](https://huggingface.co/docs/transformers/model_doc/patchtsmixer) | **`PatchTSMixerForTimeSeriesClassification`**, `PatchTSMixerForPretraining`, `PatchTSMixerForPrediction`; note `head_aggregation` (`max_pool`, `avg_pool`, …). |
| PatchTSMixer blog | [huggingface.co/blog/patchtsmixer](https://huggingface.co/blog/patchtsmixer) | Forecasting + linear probe vs full finetune workflow (MSE-centric). |
| Transformers source (mixer) | [modeling_patchtsmixer.py (main)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/patchtsmixer/modeling_patchtsmixer.py) | Exact forward signatures and tensor ranks (`last_hidden_state` as `B × nvars × num_patch × d_model`). |

---

## Exploração adicional (internet + repositório local)

Esta secção complementa o plano com o que foi revisto **na web** e **no teu clone**, para não ficar só nos uploads iniciais.

### Web (papers e documentação oficial)

- **TF-C (paper):** o abstract formaliza o problema de *domain mismatch* em séries temporais e define **Time-Frequency Consistency** como propriedade desejável: vizinhança no tempo e no domínio frequencial da **mesma** amostra devem ficar próximas no espaço latente; vizinhanças de **outras** amostras, mais longe. O modelo é **decomponível** (ramos tempo e frequência) com estimação contrastiva em cada ramo. Isto distingue TF-C de “um SimCLR genérico” mesmo quando ambos usam NT-Xent.
- **PatchTST (HF):** documentação confirma uso para **pretraining** (máscaras `random` / `forecast`), **classificação** e **regressão**; tensores de entrada em formato **(batch, sequence_length, num_input_channels)**.
- **PatchTSMixer (HF):** a classe de classificação chama-se **`PatchTSMixerForTimeSeriesClassification`** (não só “ForClassification”). Suporta **pretrain**, **forecast**, **classificação**, **regressão**. A doc descreve `target_values` com shape `(batch,)` para classificação. Há parâmetro **`head_aggregation`** (`max_pool`, `avg_pool`, `use_last`, etc.) relevante para comparar com pooling do teu encoder custom.
- **Versão do `transformers`:** em versões antigas a doc de PatchTSMixer pode não existir; fixar versão mínima no `requirements` do novo pacote.

### Código local: `TFC-pretraining/code/TFC` (leitura detalhada)

- **`model.py` — classe `TFC`:** dois **`TransformerEncoder` de 2 camadas** (tempo e frequência), cada um seguido de um **projector** MLP (`→ 256 → 128`). Não é ResNet: o backbone “oficial” nesta pasta é **Transformer** sobre sequências já alinhadas (`configs.TSlength_aligned`). O classificador downstream `target_classifier` concatena representações e usa MLP com `sigmoid` intermédio.
- **`dataloader.py` — ponto crítico para comparabilidade:**
  - Garante layout **`(N, C, T)`** com **C no eixo 1**.
  - **Corta para um só canal:** `X_train = X_train[:, :1, :int(config.TSlength_aligned)]` — ou seja, **só o primeiro canal** entra no pré-treino TFC tal como está no código.
  - Domínio frequencial: **`torch.fft.fft(x).abs()`** (magnitude; mesma length que o tempo).
  - Modo `pre_train`: devolve tuplos `(x, y, aug_time, x_f, aug_freq)`; augmentações vêm de `DataTransform_TD` / `DataTransform_FD`.
- **`trainer.py` — composição da loss no pré-treino:** usa **`NTXentLoss_poly`** (variante NT-Xent) para:
  - `loss_t`: contraste **temporal** entre embeddings `h_t` e `h_t_aug`;
  - `loss_f`: contraste na **frequência**;
  - `l_TF`: alinhar **`z_t` e `z_f`** (projetados) da mesma amostra;
  - `loss_c`: termos estilo **triplet** que mantêm `l_TF` menor que cruzamentos `z_t` com `z_f_aug`, etc. (ver linhas ~145–150).
  - Fine-tune: também treina **KNN(k=5)** em embeddings de validação como alternativa ao MLP (como no README atualizado do repo).
- **`main.py`:** `subset = True` por defeito — atenção: o README do upstream menciona subset pequeno para debug; para experiências sérias convém **desligar subset** e confirmar `configs` / caminhos `../../datasets/{name}`.
- **`model_patchtst.py`:** ficheiro **experimental / placeholder** no teu tree: comenta integração PatchTST+TFC mas o encoder PatchTST está **desativado**; cai em `TransformerEncoder` como fallback. **Não assumas** que TFC+PatchTST está pronto sem completar este módulo.

### Código local: `ablations/patchtst/` (já alinhado com AcTBeCalf)

- **`scripts/_common.sh`:** define `TRAIN_PARQUET` / `TEST_PARQUET` por defeito para **`dataset/processed/AcTBeCalf/windowed_{train,test}.parquet`**, chama `python -m hybrid_activity_recognition.main` com `--mode pretrain` e hiperparâmetros via env (`PRETRAIN_MASK_RATIO`, etc.). `BATCH_SIZE` default **2000** (muito alto vs TFC típico — reflecte pipeline HF/PatchTST, não TFC).
- **`README.md`:** lista suites (`sem_pretreino`, `tamanho_modelo`, `mixing`, …), smoke run, consolidação com `summarize_results.py`.

### Dados no workspace (o que existe de facto)

- **`dataset/`:** no glob atual: `AcTBeCalf.csv`, `processed/AcTBeCalf/*.parquet`, `tsfel_feature_manifest.json` — não há árvore grande de múltiplos datasets sob `dataset/`; **dog** e **HAR** estão sob **`data/`**.
- Implicação: o plano modular deve aceitar **vários roots** (`data/processed/HAR`, `data/dog`, `dataset/processed/AcTBeCalf`, `data/etth`) com um **registry** de datasets, não assumir uma única pasta.

### Implicações para o teu `pretreino_abalations`

1. **Canais:** decisão fixa — **apenas canal 0** (`[:, :1, :]`), como no TFC stock; ver secção **Política de canais (decisão fixa do projeto)** acima. Sem alterar métodos para multi-canal nesta grelha.
2. **Alinhar layouts:** TFC usa `(B, C, T)`; HF PatchTST/PatchTSMixer usa `(B, T, C)` — normalizar numa camada única `to_model_layout`.
3. **PatchTSMixer:** usar doc **`PatchTSMixerForTimeSeriesClassification`** + eventual **`PatchTSMixerForPretraining`** para MAE, e reportar `head_aggregation` nos resultados.
4. **Não confundir** “loss do PatchTST (reconstrução)” com “loss do TFC (NT-Xent + TF + triplet)” — métricas de “loss ~1” podem ser escalas diferentes; para collapse, usar **linear probe** ou **kNN** em embeddings.

---

## Baselines em `TFC-pretraining/code/baselines`: o que faz sentido testar vs adaptar

Percorrido: README + ficheiros de entrada (`main.py`, `train.py`, `train_model.py`) de cada pasta. Resumo orientado ao teu caso (**IMU / comportamento animal**, `.pt` HAR, parquet windowed AcTBeCalf, PatchTST em PyTorch).

### Tabela rápida

| Método | Stack | Formato de dados (no repo) | Faz sentido para ti? | Facilidade de adaptar ao teu pipeline unificado |
|--------|--------|----------------------------|----------------------|--------------------------------------------------|
| **TF-C (TFC)** | PyTorch | `train.pt` → `{"samples": (N,C,T), "labels"}`; dataloader atual usa **1 canal** + FFT | **Sim** | **Média** |
| **TS-TCC** | PyTorch | Igual: `train.pt`, `val.pt`, `test.pt` dict | **Sim** | **Alta** (mesmo contrato `.pt`; cuidado com deps antigas — melhor extrair loss/augment) |
| **TS2Vec** | PyTorch | UCR: `(n_inst, T, n_features)` | **Sim** | **Média** (loader custom para as tuas janelas) |
| **SimCLR (esta pasta)** | **TensorFlow** | `.npy` + `transformations.py` | Ideia sim; código TF pouco atrativo | **Baixa** integração; **Alta** reimplementação PyTorch + transforms |
| **Mixing-up** | — | Notebook/removido neste clone | Pouco | **Baixa** |
| **CLOCS** | PyTorch | ECG patient-specific | Pouco para animal por defeito | **Baixa** |

### Recomendação prática (ordem)

1. TS-TCC ou NT-Xent + augments IMU em PyTorch (contrato `.pt` igual ao HAR).
2. TFC.
3. TS2Vec com loader fino.
4. Evitar SimCLR-TF como caminho principal; CLOCS/Mixing-up como prioridade baixa.

### Detalhes úteis

- **TS-TCC**: modos `self_supervised`, `fine_tune`, `train_linear` alinham com congelar encoder vs fine-tune.
- **TS2Vec**: classificação em `(bs, window_len, n_channels)` — compatível com mentalidade `(B,T,C)` do PatchTST/HF.
- **SimCLR** (paper HAR): ênfase em **qual conjunto de transformações** usar — relevante para diagnóstico de “pré-treino não aprende” no IMU.

---

## Suggested first milestone (keep scope small)

1. **Data adapter**: HAR `.pt` + one **dog** CSV/parquet windows → canonical `TensorDataset` / `DataLoader`, com **`C=1`** conforme a política de canais.
2. **Three runs**: supervised-only baseline; **linear probe** after random init vs after pretrain (same encoder family).
3. **One pretrain objective** that worked on ETTh wired to **your IMU** with **diagnostics** (collapse detection).
4. Expand to **TS-TCC / NT-Xent+augment (PyTorch)**, then **TFC**, then **TS2Vec** — evitar integrar SimCLR-TensorFlow como caminho principal; ver tabela de baselines acima.

This sequence minimizes time lost debugging five objectives at once.

---

## Note on folder naming

You wrote **`pretreino_abalations`**. Consider renaming to **`pretrain_ablations`** (English + spelling) to avoid typos in imports and paths; if you keep the Portuguese name, use it **consistently** everywhere.
