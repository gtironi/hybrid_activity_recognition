# Plano: Grade de Experimentos Completa + PatchTST como Encoder

## Prompt original do usuario

> Olhe o que ja produzi sobre o repositorio em PLAN_modular_hybrid.md e PLAN_raw_input_hybrid.md.
> Vamos continuar a fazer isso, seguindo a estrutura deles.
>
> Quero usar o patchtst como encoder no ramo de deep learning. Para comparar com o cnn+lstm no
> artigo.
>
> Quero tambem ter um baseline que seja cnn+lstm, quero ter um baseline que seja patchtst apenas e o
> modelo hibrido patchtst + features tsfel. Como tem hoje nos modelos robust_hybrid. Fazendo fusao
> no final.
>
> Todos os modelos (robust_hybrid, hybrid_cnn_lstm, patchtst) devem ter esses 2 casos podendo ser
> rodados e testados para comparacao:
>
> 1. So o ramo deep learning, recebendo apenas x, y, z na janela (sem o hibrido, sem fusao no final)
> 2. Hibrido (ramo deep learning + ramo tsfel, com fusao antes da cabeca de classificacao)
>
> O encoder sempre recebe apenas x, y, z. O TSFEL fica exclusivamente no ramo separado de fusao.
>
> Deve ser tudo controlado por um comando .sh que pode ser rodado.
>
> A principal coisa aqui e entender como colocar o patchtst no experimento. Ele difere dos outros
> porque tem etapa de pretreino. Por agora, deve-se usar o dataset que temos (sem label) para o
> pretreino antes de treinar a arquitetura completa. Ou seja, vamos ter uma etapa de pretreino do
> patchtst e depois usar o checkpoint salvo como pesos iniciais do encoder no treino supervisionado.
>
> No futuro, vou usar outro dado sem label para o pretreino. O codigo deve estar preparado para
> isso, podendo so mudar o caminho do arquivo no .sh.
>
> Tambem, quero poder so indicar o arquivo de checkpoint a ser usado, caso o pretreino ja tenha
> sido feito e eu queira pular essa etapa.
>
> Estruture o codigo de forma que o PatchTST fique atras de uma interface propria do projeto.
> Assim, se no futuro precisarmos do repositorio oficial, a troca do backend do encoder sera
> localizada e nao exigira redesenhar o resto do experimento.

---

## Contexto: o que ja existe

Referencia aos outros dois planos:

- **PLAN_modular_hybrid.md** — define a interface modular (`SignalEncoder`, `TsfelBranch`,
  `FusionModule`, `ClassificationHead`) e o container `HybridModel`. Os encoders existentes
  (`CNNLSTMEncoder`, `RobustCNNLSTMEncoder`) ja tem contratos definidos.

- **PLAN_raw_input_hybrid.md** — define `RawWindowDataset`, `prepare_raw_dataloaders`, e
  `prepare_tsfel_per_window.py`. O encoder recebe sinal bruto janelado em memoria.

Este documento expande esses dois planos com:
1. A grade 3×2 de experimentos (3 familias de encoder × 2 modos)
2. `PatchTSTEncoder` como implementacao de `SignalEncoder`
3. Pre-treino self-supervised do PatchTST e gerenciamento de checkpoints
4. Script `.sh` que orquestra todos os experimentos

---

## 1. A grade de experimentos

### 3 familias de encoder

| ID | Familia | Encoder backbone |
|----|---------|-----------------|
| `cnn_lstm` | HybridCNNLSTM | 2 blocos Conv1D + BiLSTM 2 camadas |
| `robust` | RobustHybridModel | 3 blocos Conv1D + BiLSTM 1 camada |
| `patchtst` | PatchTST | Patches + Transformer encoder (HF) |

### 2 modos

| ID | Nome | Deep encoder recebe | Ramo TSFEL separado | Fusao |
|----|------|---------------------|---------------------|-------|
| `deep_only` | Baseline DL puro | xyz (3 canais) | Nao | Nao |
| `hybrid` | Hibrido DL + TSFEL | xyz (3 canais) | Sim | Sim |

O encoder **sempre** recebe apenas `(B, 3, T)`. O TSFEL entra exclusivamente pelo ramo separado
no modo `hybrid`.

### Grade resultante: 6 configuracoes

```
              deep_only        hybrid
cnn_lstm      cnn_lstm_deep    cnn_lstm_hybrid
robust        robust_deep      robust_hybrid
patchtst      patchtst_deep    patchtst_hybrid
```

Cada celula e um experimento separado com seu proprio checkpoint e metricas.

### Interpretacao cientifica (ablation)

| Comparacao | O que testa |
|------------|------------|
| `cnn_lstm_deep` vs `cnn_lstm_hybrid` | Ganho do ramo TSFEL sobre CNN+LSTM puro |
| `robust_deep` vs `robust_hybrid` | Ganho do ramo TSFEL sobre Robust CNN+LSTM puro |
| `patchtst_deep` vs `patchtst_hybrid` | Ganho do ramo TSFEL sobre PatchTST puro |
| `cnn_lstm_hybrid` vs `patchtst_hybrid` | PatchTST vs CNN+LSTM no modo hibrido |
| `cnn_lstm_deep` vs `patchtst_deep` | PatchTST vs CNN+LSTM como encoder puro |

---

## 2. Como funciona cada modo no forward pass

### Modo `deep_only`

```python
# x_signal:   (B, 3, T)
# x_features: ignorado — passado para manter interface uniforme, modelo nao usa
z_sig  = encoder(x_signal)    # (B, enc_dim)
logits = head(z_sig)          # (B, num_classes)
```

O modelo nao tem `tsfel_branch` nem `fusion`. `x_features` e aceito na assinatura mas ignorado
no forward. Isso permite que o `Trainer` use sempre `model(x_sig, x_feat)` sem alteracao.

### Modo `hybrid`

```python
# x_signal:   (B, 3, T)
# x_features: (B, K) — features TSFEL pre-computadas por janela
z_sig   = encoder(x_signal)         # (B, enc_dim)
z_ts    = tsfel_branch(x_features)  # (B, tsfel_dim)
z_fused = fusion(z_sig, z_ts)       # (B, enc_dim + tsfel_dim) via ConcatFusion
logits  = head(z_fused)             # (B, num_classes)
```

Estado arquitetural atual do repositorio. O encoder nao sabe que existe TSFEL.

---

## 3. Como o modo e controlado no codigo

### Parametro `input_mode` no `HybridModel`

```python
class HybridModel(nn.Module):
    def __init__(
        self,
        encoder: SignalEncoder,
        tsfel_branch: TsfelBranch | None,   # None para deep_only
        fusion: FusionModule | None,         # None para deep_only
        head: ClassificationHead,
        input_mode: Literal["deep_only", "hybrid"],
    ): ...

    def forward(self, x_signal: Tensor, x_features: Tensor) -> Tensor:
        z_sig = self.encoder(x_signal)

        if self.input_mode == "deep_only":
            return self.head(z_sig)

        z_ts    = self.tsfel_branch(x_features)
        z_fused = self.fusion(z_sig, z_ts)
        return self.head(z_fused)
```

### Factory `build_hybrid_model`

```python
def build_hybrid_model(
    encoder_name: str,       # "cnn_lstm" | "robust" | "patchtst"
    input_mode: str,         # "deep_only" | "hybrid"
    num_classes: int,
    n_tsfel_feats: int,
    **encoder_kwargs,
) -> HybridModel:
    encoder      = make_encoder(encoder_name, in_channels=3, ...)
    tsfel_branch = MLPTsfelBranch(n_tsfel_feats, ...) if input_mode == "hybrid" else None
    fusion       = ConcatFusion() if input_mode == "hybrid" else None
    head_in_dim  = encoder.output_dim + (tsfel_branch.output_dim if tsfel_branch else 0)
    head         = MLPHead(head_in_dim, num_classes)
    return HybridModel(encoder, tsfel_branch, fusion, head, input_mode)
```

O encoder tem sempre `in_channels=3`. A factory nunca varia isso.

---

## 4. PatchTST como SignalEncoder

### Principio: interface propria do projeto

O PatchTST (Hugging Face `transformers`) fica encapsulado atras de `PatchTSTEncoder`, que
implementa a interface `SignalEncoder` do projeto. Se no futuro o backend mudar para o
repositorio oficial (https://github.com/PatchTST/PatchTST), apenas `PatchTSTEncoder` muda.
O resto do experimento — `HybridModel`, `Trainer`, scripts `.sh` — nao precisa saber.

### Wrapper `PatchTSTEncoder`

**Arquivo**: `src/.../models/modular/encoders.py`

```python
from transformers import PatchTSTConfig, PatchTSTModel

class PatchTSTEncoder(SignalEncoder):
    """
    Wrapper do PatchTST (Hugging Face) como SignalEncoder.

    HF PatchTST espera (B, T, C); nosso projeto usa (B, C, T).
    O wrapper faz a transposicao internamente. O encoder sempre recebe
    apenas os 3 canais de acelerometro (accX, accY, accZ).

    Args:
        context_length:  tamanho da janela em timesteps (default 75)
        patch_length:    tamanho de cada patch (default 8)
        patch_stride:    stride entre patches (default 8, sem overlap)
        d_model:         dimensao do transformer (default 128)
        num_heads:       cabecas de atencao (default 4)
        num_layers:      camadas do transformer (default 3)
        dropout:         dropout geral (default 0.1)
        pretrained_path: caminho para checkpoint de pretreino (opcional)
    """

    def __init__(
        self,
        context_length: int = 75,
        patch_length: int = 8,
        patch_stride: int = 8,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        pretrained_path: str | None = None,
    ):
        super().__init__()
        config = PatchTSTConfig(
            num_input_channels=3,         # sempre xyz
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            attention_dropout=dropout,
            ff_dropout=dropout,
            path_dropout=dropout,
            pooling_type="mean",
            channel_attention=False,      # channel independence padrao PatchTST
        )
        self._backbone = PatchTSTModel(config)
        self._d_model = d_model

        if pretrained_path:
            self.load_pretrained_encoder(pretrained_path)

    @property
    def output_dim(self) -> int:
        return self._d_model

    def forward(self, x_signal: Tensor) -> Tensor:
        # x_signal: (B, 3, T)  →  HF espera (B, T, 3)
        x = x_signal.permute(0, 2, 1)            # (B, T, 3)
        out = self._backbone(past_values=x)
        # last_hidden_state: (B, num_patches * 3, d_model) com channel independence
        # mean pooling → (B, d_model)
        return out.last_hidden_state.mean(dim=1)

    def load_pretrained_encoder(self, path: str) -> None:
        """Carrega pesos do backbone a partir de checkpoint de pretreino."""
        state = torch.load(path, map_location="cpu")
        # PatchTSTForPretraining salva o backbone com prefixo "model."
        encoder_state = {
            k[len("model."):]: v
            for k, v in state.items()
            if k.startswith("model.")
        }
        self._backbone.load_state_dict(encoder_state, strict=False)
        # strict=False porque PatchTSTModel nao tem a cabeca de reconstrucao
```

### Numero de patches para janela de 75 amostras

Com `patch_length=8`, `patch_stride=8` (sem overlap entre patches):
- `num_patches = floor((75 - 8) / 8) + 1 = 9` patches por canal
- Com 3 canais e channel independence: `9 × 3 = 27` tokens no transformer
- `d_model = 128` → `output_dim = 128`

Com `patch_length=5`, `patch_stride=5`:
- `num_patches = floor((75 - 5) / 5) + 1 = 15` patches → `45` tokens
- Mais fine-grained, mais compute

**Recomendacao**: `patch_length=8, patch_stride=8`. Citavel via Nie et al. (2023) — o paper
PatchTST recomenda configuracoes que resultem em 8-16 patches por canal.

---

## 5. Pre-treino self-supervised do PatchTST

### Motivacao

O PatchTST pode ser pre-treinado via masked reconstruction (MAE — Masked Auto-Encoding).
Patches aleatorios sao mascarados e o modelo aprende a reconstrui-los a partir do contexto.
Isso permite usar dados sem label para inicializar o encoder antes do fine-tuning supervisionado,
potencialmente melhorando a representacao aprendida.

### Fluxo completo

```
Parquet bruto sem label (mesmo formato que train.parquet, colunas: calfId, segId, accX, accY, accZ)
    ↓
PretrainWindowDataset: janelamento em memoria, sem TSFEL, sem label → (3, T) por janela
    ↓
PretrainTrainer.pretrain(): treina PatchTSTForPretraining por N epocas (MAE loss)
    ↓
patchtst_pretrained.pt salvo em experiments/patchtst_pretrain/
    ↓
PatchTSTEncoder(pretrained_path="patchtst_pretrained.pt")
    ↓
HybridModel(encoder=patchtst_encoder, ...)
    ↓
Trainer.train_supervised(): fine-tuning supervisionado com ou sem ramo TSFEL
```

### Formato do dado de pretreino

Parquet **janelado de sinais** gerado por `prepare_windowed_parquet.py --no-label`.
Segue a convencao de colunas padrao definida em `PLAN_raw_input_hybrid.md`:
`subject, segment_id, window_start, acc_x=[...], acc_y=[...], acc_z=[...]`
Sem coluna `label`, sem features TSFEL.

Pode ser gerado a partir do proprio `train.parquet` ou de qualquer outra fonte de dados
brutos que siga a convencao `datetime, subject, acc_x, acc_y, acc_z`. Trocar a fonte
de pretreino no futuro e so mudar o caminho no `.sh`.

### Dataset de pretreino

**Arquivo**: `src/.../data/pretrain_dataset.py`

```python
class PretrainWindowDataset(Dataset):
    """
    Le parquet janelado sem label (gerado com --no-label).
    Retorna apenas o tensor de sinal (C, T) — sem TSFEL, sem label.
    Usa detect_signal_cols() de utils/schema.py para detectar canais.
    """
    def __init__(self, windowed_df: pd.DataFrame):
        from hybrid_activity_recognition.utils.schema import detect_signal_cols
        sig_cols = detect_signal_cols(windowed_df)  # ["acc_x", "acc_y", "acc_z"]
        # stack cada coluna (lista de floats) → (N, C, T)
        arrays = [np.stack(windowed_df[c].values) for c in sig_cols]
        self.signals = torch.tensor(np.stack(arrays, axis=1), dtype=torch.float32)

    def __getitem__(self, idx) -> Tensor:
        return self.signals[idx]   # (C, T) — sem label, sem features
```

### Trainer de pretreino

**Arquivo**: `src/.../training/pretrain_trainer.py`

```python
class PretrainTrainer:
    def __init__(self, encoder_config: PatchTSTConfig, device, output_dir): ...

    def pretrain(
        self,
        pretrain_dl: DataLoader,          # DataLoader de PretrainWindowDataset
        epochs: int = 50,
        lr: float = 1e-4,
        mask_ratio: float = 0.4,
        checkpoint_name: str = "patchtst_pretrained.pt",
    ) -> str:
        """Treina PatchTSTForPretraining e retorna caminho do checkpoint."""
        from transformers import PatchTSTForPretraining
        model = PatchTSTForPretraining(self.encoder_config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        ...
        # Loop: out = model(past_values=x); out.loss.backward()
        torch.save(model.state_dict(), ckpt_path)
        return str(ckpt_path)
```

### Pular o pretreino

Se o checkpoint ja existe ou se o usuario quer usar pesos externos:

```bash
# No .sh: se PATCHTST_CHECKPOINT estiver preenchido, o pretreino e pulado
PATCHTST_CHECKPOINT="experiments/patchtst_pretrain/patchtst_pretrained.pt"
```

No `main.py`:
```python
if args.patchtst_checkpoint:
    # Carrega diretamente, sem pretreino
    encoder = PatchTSTEncoder(pretrained_path=args.patchtst_checkpoint, ...)
else:
    # Roda pretreino, salva checkpoint, depois carrega
    ckpt = run_pretrain(args)
    encoder = PatchTSTEncoder(pretrained_path=ckpt, ...)
```

---

## 6. Estrutura do script `.sh`

### Arquivo: `run_experiments.sh`

```bash
#!/bin/bash
set -e

# ============================================================
# Configuracao — ajuste os caminhos aqui
# ============================================================

TRAIN_PARQUET="dataset/processed/AcTBeCalf/train.parquet"
TEST_PARQUET="dataset/processed/AcTBeCalf/test.parquet"
TSFEL_PARQUET_TRAIN="dataset/processed/AcTBeCalf/tsfel_train.parquet"
TSFEL_PARQUET_TEST="dataset/processed/AcTBeCalf/tsfel_test.parquet"

# Dado para pretreino do PatchTST (sem label; pode ser o proprio train.parquet)
PRETRAIN_PARQUET="$TRAIN_PARQUET"

# Checkpoint do PatchTST: se vazio, roda pretreino; se preenchido, pula
PATCHTST_CHECKPOINT=""

EPOCHS=50
BATCH=64
SEED=42
DEVICE="cuda"

# ============================================================
# Etapa 0: Pre-treino PatchTST (roda uma unica vez)
# ============================================================

if [ -z "$PATCHTST_CHECKPOINT" ]; then
    echo "=== Pre-treino PatchTST ==="
    PATCHTST_CHECKPOINT="experiments/patchtst_pretrain/patchtst_pretrained.pt"
    PYTHONPATH=src python -m hybrid_activity_recognition.main \
        --mode pretrain \
        --model patchtst \
        --pretrain_parquet  "$PRETRAIN_PARQUET" \
        --output_dir        experiments/patchtst_pretrain \
        --pretrain_epochs   "$EPOCHS" \
        --seed              "$SEED" \
        --device            "$DEVICE"
fi

echo "Checkpoint PatchTST: $PATCHTST_CHECKPOINT"

# ============================================================
# Funcao helper: roda um experimento
# ============================================================

run_experiment() {
    local MODEL=$1        # cnn_lstm | robust | patchtst
    local MODE=$2         # deep_only | hybrid
    local OUT="experiments/${MODEL}_${MODE}"
    mkdir -p "$OUT"
    echo "=== ${MODEL} / ${MODE} ==="

    EXTRA_ARGS=""
    if [ "$MODEL" = "patchtst" ]; then
        EXTRA_ARGS="--patchtst_checkpoint $PATCHTST_CHECKPOINT"
    fi

    PYTHONPATH=src python -m hybrid_activity_recognition.main \
        --mode               supervised \
        --model              "$MODEL" \
        --input_mode         "$MODE" \
        --labeled_parquet_train  "$TRAIN_PARQUET" \
        --labeled_parquet_test   "$TEST_PARQUET" \
        --tsfel_parquet_train    "$TSFEL_PARQUET_TRAIN" \
        --tsfel_parquet_test     "$TSFEL_PARQUET_TEST" \
        --output_dir             "$OUT" \
        --epochs             "$EPOCHS" \
        --batch_size         "$BATCH" \
        --seed               "$SEED" \
        --device             "$DEVICE" \
        $EXTRA_ARGS \
        2>&1 | tee "${OUT}/train.log"
}

# ============================================================
# Grade de experimentos: 3 encoders x 2 modos = 6 experimentos
# ============================================================

for MODEL in cnn_lstm robust patchtst; do
    for MODE in deep_only hybrid; do
        run_experiment "$MODEL" "$MODE"
    done
done

echo "=== Todos os experimentos concluidos ==="
```

---

## 7. Novos parametros em `main.py`

| Parametro | Tipo | Descricao |
|-----------|------|-----------|
| `--input_mode` | `{deep_only,hybrid}` | Modo do modelo |
| `--model` | `{cnn_lstm,robust,patchtst}` | Familia do encoder |
| `--tsfel_parquet_train` | str | Parquet TSFEL por janela (treino); necessario se `hybrid` |
| `--tsfel_parquet_test` | str | Parquet TSFEL por janela (teste); necessario se `hybrid` |
| `--patchtst_checkpoint` | str | Checkpoint de pretreino do PatchTST (opcional; pula pretreino) |
| `--pretrain_parquet` | str | Parquet bruto para pretreino (apenas se `--mode pretrain`) |
| `--patchtst_d_model` | int | Dimensao do transformer (default 128) |
| `--patchtst_num_layers` | int | Camadas do transformer (default 3) |
| `--patchtst_patch_length` | int | Tamanho do patch (default 8) |
| `--patchtst_patch_stride` | int | Stride do patch (default 8) |
| `--pretrain_epochs` | int | Epocas de pretreino (default 50) |
| `--pretrain_mask_ratio` | float | Fracao de patches mascarados no MAE (default 0.4) |

---

## 8. Estrutura de arquivos: o que criar / modificar

### Arquivos NOVOS

| Arquivo | Conteudo |
|---------|----------|
| `src/.../utils/windowing.py` | `generate_windows()` compartilhado entre datasets e script TSFEL |
| `src/.../data/raw_dataset.py` | `RawWindowDataset`: janelamento em memoria, retorna `(signal, features, label)` |
| `src/.../data/raw_dataloader.py` | `prepare_raw_dataloaders()`: normaliza, divide, retorna DataLoaders |
| `src/.../data/pretrain_dataset.py` | `PretrainWindowDataset`: janelas sem label para pretreino PatchTST |
| `src/.../models/modular/base.py` | ABCs: `SignalEncoder`, `TsfelBranch`, `FusionModule`, `ClassificationHead` |
| `src/.../models/modular/model.py` | `HybridModel` com `input_mode: {"deep_only", "hybrid"}` |
| `src/.../models/modular/encoders.py` | `CNNLSTMEncoder`, `RobustCNNLSTMEncoder`, `PatchTSTEncoder` |
| `src/.../models/modular/tsfel_branches.py` | `MLPTsfelBranch` |
| `src/.../models/modular/fusion.py` | `ConcatFusion` |
| `src/.../models/modular/heads.py` | `MLPHead`, `LinearHead` |
| `src/.../models/modular/__init__.py` | `build_hybrid_model(encoder_name, input_mode, ...)` factory |
| `src/.../training/pretrain_trainer.py` | `PretrainTrainer` com metodo `pretrain()` |
| `scripts/prepare_tsfel_per_window.py` | Janela parquet bruto, extrai TSFEL, salva manifest |
| `run_experiments.sh` | Script shell orquestrando os 6 experimentos + pretreino |

### Arquivos que MUDAM

| Arquivo | O que muda |
|---------|-----------|
| `src/.../main.py` | Novos args (tabela secao 7); novo modo `pretrain`; `build_model` via factory modular |
| `src/.../models/hybrid_cnn_lstm/__init__.py` | Re-exporta de `models/modular` para backward compat |
| `src/.../models/robust_hybrid/__init__.py` | Re-exporta de `models/modular` para backward compat |

### Arquivos que NAO mudam

`training/trainer.py`, `training/loss.py`, `training/metrics.py`, `training/augment.py`,
`data/dataloader.py` (pipeline windowed legado permanece funcional), `layers/`.

---

## 9. Detalhes criticos de implementacao

### 9.1 Encoder sempre recebe (B, 3, T)

O encoder nao tem conhecimento de TSFEL. Sempre recebe 3 canais. Nao ha logica de tiling
nem concatenacao no forward. O TSFEL so existe no ramo separado do modo `hybrid`.

### 9.2 Compatibilidade de checkpoint pretreino com encoder de classificacao

`PatchTSTForPretraining` e `PatchTSTModel` compartilham o backbone com os mesmos nomes de
parametro no HF. A transferencia:

```python
pretrain_state = torch.load("patchtst_pretrained.pt")
encoder_state = {
    k[len("model."):]: v
    for k, v in pretrain_state.items()
    if k.startswith("model.")
}
patchtst_backbone.load_state_dict(encoder_state, strict=False)
```

`strict=False` e necessario porque `PatchTSTModel` nao tem a cabeca de reconstrucao do MAE.

### 9.3 `output_dim` do PatchTSTEncoder

Com `pooling_type="mean"` e channel independence:
- `last_hidden_state.shape = (B, num_patches × 3, d_model)`
- Apos `mean(dim=1)`: `(B, d_model)`
- `output_dim = d_model = 128` (default)

A factory usa `encoder.output_dim` para calcular `head_in_dim`:
- Modo `deep_only`: `head_in_dim = encoder.output_dim`
- Modo `hybrid`: `head_in_dim = encoder.output_dim + tsfel_branch.output_dim`

### 9.4 Trainer nao precisa mudar

O `Trainer` atual chama sempre `model(x_sig, x_feat)`. No modo `deep_only`, o modelo aceita
`x_feat` na assinatura mas nao usa no forward. Interface preservada sem nenhuma alteracao.

### 9.5 `dataset.labels` para class weights

O `Trainer.train_supervised` acessa `train_dl.dataset.labels`. O `RawWindowDataset` precisa
expor esse atributo como `np.ndarray` de inteiros, igual ao `CalfHybridDataset` atual.

### 9.6 PatchTST sem pretreino (edge case)

Se `--patchtst_checkpoint` nao for fornecido E `--mode supervised` for usado com `--model
patchtst`, o `main.py` deve lan cuma excecao clara pedindo para rodar o pretreino primeiro
ou fornecer um checkpoint. Nao deve silenciosamente usar pesos aleatorios sem avisar — isso
poderia confundir resultados no artigo.

---

## 10. Resumo das dependencias entre planos

```
PLAN_modular_hybrid.md
  └─ define: SignalEncoder ABC, TsfelBranch ABC, FusionModule ABC, ClassificationHead ABC
  └─ define: HybridModel container
  └─ define: CNNLSTMEncoder, RobustCNNLSTMEncoder
  └─ define: ConcatFusion, MLPHead, MLPTsfelBranch

PLAN_raw_input_hybrid.md
  └─ define: generate_windows() em utils/windowing.py
  └─ define: RawWindowDataset
  └─ define: prepare_raw_dataloaders()
  └─ define: prepare_tsfel_per_window.py

PLAN_experiments_patchtst.md  ← ESTE ARQUIVO
  └─ expande HybridModel com: input_mode ("deep_only" | "hybrid")
  └─ adiciona: PatchTSTEncoder (wrapper HF, sempre 3 canais)
  └─ adiciona: PretrainWindowDataset, PretrainTrainer
  └─ adiciona: run_experiments.sh (6 experimentos + etapa 0 pretreino)
  └─ adiciona: novos args em main.py
```

---

## 11. Checklist de implementacao

### Fase 0: Pre-requisitos (dos outros planos)

- [ ] `utils/windowing.py` com `generate_windows()`
- [ ] `models/modular/base.py` com as 4 ABCs e `output_dim` como `@abstractproperty`
- [ ] `CNNLSTMEncoder`, `RobustCNNLSTMEncoder` implementando `SignalEncoder`

### Fase 1: HybridModel com input_mode

- [ ] `models/modular/model.py`: `HybridModel` com `input_mode: {"deep_only", "hybrid"}`
- [ ] `models/modular/__init__.py`: `build_hybrid_model(encoder_name, input_mode, ...)`
- [ ] Testar que modo `deep_only` ignora `tsfel_branch` e `fusion` sem erros

### Fase 2: PatchTSTEncoder

- [ ] Adicionar `PatchTSTEncoder` em `models/modular/encoders.py`
- [ ] Validar transposicao `(B, 3, T) → (B, T, 3)` no forward
- [ ] Validar `output_dim = d_model` apos mean pooling
- [ ] Testar forward com batch real

### Fase 3: Pre-treino

- [ ] `data/pretrain_dataset.py`: `PretrainWindowDataset` (janelas sem label, sem TSFEL)
- [ ] `training/pretrain_trainer.py`: `PretrainTrainer.pretrain()`
- [ ] Testar que checkpoint salvo carrega corretamente em `PatchTSTEncoder`
- [ ] Verificar `strict=False` na transferencia de pesos

### Fase 4: Integracao em `main.py`

- [ ] Novos argumentos CLI (tabela secao 7)
- [ ] Modo `pretrain` para PatchTST
- [ ] `build_hybrid_model` substituindo `build_model` atual (backward compat)
- [ ] Erro claro se `patchtst` e `supervised` sem checkpoint

### Fase 5: Script de experimentos

- [ ] `run_experiments.sh`: etapa 0 + 6 experimentos
- [ ] Testar que todos os 6 rodam sem erro
- [ ] Verificar que resultados de `cnn_lstm_hybrid` e `robust_hybrid` batem com os modelos atuais

---

## 12. Baseline: TSFEL-only com Random Forest

### Motivacao

Baseline classico totalmente separado de tudo. Nao usa deep learning. Serve para responder:
"o quanto o TSFEL sozinho, sem nenhuma representacao aprendida, consegue classificar?"

Deve ser o experimento mais simples do artigo. Um arquivo Python, roda em segundos.

### O que ele faz

1. Le o parquet janelado ja existente (o mesmo `windowed_train.parquet` que os outros modelos usam)
2. Separa as colunas TSFEL das colunas de sinal
3. Aplica `SelectKBest` (f_classif) para selecionar as K features mais discriminativas
4. Treina um `RandomForestClassifier`
5. Avalia no `windowed_test.parquet`
6. Imprime accuracy, macro F1, weighted F1

### Arquivo: `scripts/tsfel_baseline.py`

```python
"""
Baseline TSFEL-only: SelectKBest + Random Forest.
Roda totalmente separado dos modelos de deep learning.

Uso:
    python scripts/tsfel_baseline.py \
        --train dataset/processed/AcTBeCalf/windowed_train.parquet \
        --test  dataset/processed/AcTBeCalf/windowed_test.parquet \
        --k     50
"""

import argparse
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

_STANDARD_COLS = {"dateTime", "calfId", "calf_id", "segId", "acc_x", "acc_y", "acc_z", "label"}

def feature_cols(df):
    return [c for c in df.columns if c not in _STANDARD_COLS]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test",  required=True)
    p.add_argument("--k",     type=int, default=50)
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--seed",  type=int, default=42)
    args = p.parse_args()

    train = pd.read_parquet(args.train)
    test  = pd.read_parquet(args.test)

    feats = feature_cols(train)

    le = LabelEncoder().fit(train["label"])
    y_train = le.transform(train["label"])
    y_test  = le.transform(test["label"])

    X_train = train[feats].fillna(0).values
    X_test  = test[feats].fillna(0).values

    selector = SelectKBest(f_classif, k=args.k).fit(X_train, y_train)
    X_train  = selector.transform(X_train)
    X_test   = selector.transform(X_test)

    clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"accuracy   : {accuracy_score(y_test, y_pred):.4f}")
    print(f"f1_macro   : {f1_score(y_test, y_pred, average='macro',    zero_division=0):.4f}")
    print(f"f1_weighted: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

if __name__ == "__main__":
    main()
```

### Adicionar no `run_experiments.sh`

```bash
# ============================================================
# Baseline TSFEL-only
# ============================================================

echo "=== Baseline TSFEL-only ==="
python scripts/tsfel_baseline.py \
    --train "$WINDOWED_TRAIN_PARQUET" \
    --test  "$WINDOWED_TEST_PARQUET" \
    --k     50 \
    2>&1 | tee experiments/tsfel_baseline/results.log
```

Adicionar no topo do `.sh`:
```bash
WINDOWED_TRAIN_PARQUET="dataset/processed/AcTBeCalf/windowed_train.parquet"
WINDOWED_TEST_PARQUET="dataset/processed/AcTBeCalf/windowed_test.parquet"
```

### O que NAO fazer

- Nao integrar com PyTorch, DataLoader, nem nenhuma classe do projeto
- Nao usar o pipeline de normalizacao do `dataloader.py`
- Nao criar classes, heranca, ou abstrações
- Sklearn normaliza internamente onde necessario; o Random Forest e invariante a escala

---

## 13. Referencias para o artigo

### PatchTST

- Nie, Y. et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with
  Transformers." *ICLR 2023*. — Paper original do PatchTST.

### MAE para pre-treino de series temporais

- He, K. et al. (2022). "Masked Autoencoders Are Scalable Vision Learners." *CVPR 2022*.
  — Inspiracao do mecanismo de mascaramento MAE.

- Dong, Z. et al. (2023). "SimMTM: A Simple Pre-Training Framework for Masked Time-Series
  Modeling." *NeurIPS 2023*. — Pre-treino MAE especificamente para series temporais.

### CNN+LSTM e hibrido DL+handcrafted (dos outros planos)

- Ordonez, F.J. & Roggen, D. (2016). "Deep Convolutional and LSTM Recurrent Neural Networks
  for Multimodal Wearable Activity Recognition." *Sensors*, 16(1), 115.

- Phuong, N.H. & Phuc, D.T. (2021). "Feature fusion using deep learning for smartphone based
  human activity recognition." *Int. J. Information Technology*.

- Barandas, M. et al. (2020). "TSFEL: Time Series Feature Extraction Library." *SoftwareX*.

- Mao, A. et al. (2023). "Deep learning-based animal activity recognition with wearable
  sensors." *Computers and Electronics in Agriculture*, 211.
