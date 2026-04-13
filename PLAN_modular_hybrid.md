# Plano: Arquitetura Híbrida Modular

## Prompt original do usuário

Olhe para o meu projeto hoje. Principalmente focado no @src/hybrid_activity_recognition/__init__.py

e coisas dentro desse diretorio.

Foque no hydrid_cnn_lstm e no robust_hybrid.

A ideia dos dois é juntar a extração de features dos modelos de deep learning com as features estatisticas extraidas pelo tsfel.

Vamos primeiramente analisar o que está sendo feito hoje.

ignore completamente o fixmatch. Foque nos dois que mandei.

Hoje, se não me engano, o input desse smodelos é um dataset janelado, com váriso x y z e alguams features do tsefel. então o modelo de deep learning aprende em cima disso e depois junta com o tsfel com uma fusão.

Confirme se é isso mesmo.

A ideia aqui é continuar com essa ideia de fusão de representação do deep leaning com o tsfel. Após a fusão, vem uam baceça de classificação (que deve poder ser desacoplada e trocada, para testar varias coisas).

Veja, é muito claro que temos um arquitetura de encoder para extrair um reseptação puramente por deep leanrning, um ramo paralelo com um tsfel (no nosso caso, tem uma porjeção para garantir o tamanho) uma forma de fusão para juntar as duas coisas (hoje só concatena) e um head de classificação que usa a representação fundida e pode propagar o gradiente para os dois ramos.

Quero manter essa interface, mas podendo varias o enconder, o ramo do tsfel, a fusao e o head de classificação.

Exploque o repostiorio e entenda como el funciona hoje, anotando coisas relevantes para criar isso que descrevi apra voce.

Ainda, escheva, no topo do arquivo que voce vai crair, exatamente esse pormpt que mandei para voce, com essas palavras ou muito proximo.

faça a exploração e escreva um arquivo markdown que voce usará depois para implemntar, nao implemnta agora, estamso apenas nos preparando.

---

## Estado atual confirmado

A arquitetura hoje funciona exatamente como o usuário descreveu. Os dois modelos (`HybridCNNLSTM` e
`RobustHybridModel`) têm a mesma estrutura lógica de quatro blocos:

```
x_signal  ──► [ Signal Encoder ]──────────────────┐
                                                   ▼
                                             [ Fusion ] ──► [ Classification Head ] ──► logits
                                                   ▲
x_features ──► [ TSFEL Branch (MLP Projection) ]──┘
```

Porém os blocos estão **hardcoded dentro de cada modelo**. O `HybridCNNLSTM` instancia
`HybridCNNLSTMSignalBranch`, `TsfelMLPBranch`, `ConcatFusion` e `MLPClassificationHead`
diretamente no seu `__init__`. O `RobustHybridModel` faz o mesmo com `RobustCNNLSTMSignalBranch`.
Não há como variar esses componentes sem criar um terceiro arquivo de modelo do zero.

---

## O que precisa mudar

Criar um único modelo container (`HybridModel`) que recebe os quatro blocos como argumentos, junto
com uma coleção de implementações concretas para cada bloco. O contrato de cada bloco é definido
por uma classe base (ABC ou protocolo simples). A escolha dos blocos acontece fora do modelo, via
fábrica ou instanciação explícita.

---

## Contratos (interfaces) de cada bloco

### 1. `SignalEncoder`

```python
class SignalEncoder(nn.Module):
    @property
    def output_dim(self) -> int: ...

    def forward(self, x_signal: Tensor) -> Tensor:
        # x_signal : (B, 3, T)   — 3 eixos, T timesteps
        # retorna  : (B, output_dim)
        ...
```

`output_dim` é obrigatório porque `Fusion` e `HybridModel` precisam saber o tamanho do embedding
sem fazer um forward pass.

### 2. `TsfelBranch`

```python
class TsfelBranch(nn.Module):
    @property
    def output_dim(self) -> int: ...

    def forward(self, x_features: Tensor) -> Tensor:
        # x_features : (B, n_features_tsfel)
        # retorna    : (B, output_dim)
        ...
```

A projeção linear interna garante que `output_dim` é fixo independentemente do número de features
TSFEL do dataset.

### 3. `FusionModule`

```python
class FusionModule(nn.Module):
    @property
    def output_dim(self) -> int: ...

    def forward(self, z_signal: Tensor, z_tsfel: Tensor) -> Tensor:
        # z_signal : (B, encoder.output_dim)
        # z_tsfel  : (B, tsfel_branch.output_dim)
        # retorna  : (B, output_dim)
        ...
```

O `output_dim` do `FusionModule` alimenta o `ClassificationHead`.

### 4. `ClassificationHead`

```python
class ClassificationHead(nn.Module):
    def forward(self, z_fused: Tensor) -> Tensor:
        # z_fused : (B, fusion.output_dim)
        # retorna : (B, num_classes)  — logits
        ...
```

---

## Modelo container

```python
class HybridModel(nn.Module):
    def __init__(
        self,
        encoder: SignalEncoder,
        tsfel_branch: TsfelBranch,
        fusion: FusionModule,
        head: ClassificationHead,
    ): ...

    def forward(self, x_signal: Tensor, x_features: Tensor) -> Tensor:
        z_sig  = self.encoder(x_signal)       # (B, encoder.output_dim)
        z_ts   = self.tsfel_branch(x_features) # (B, tsfel_branch.output_dim)
        z      = self.fusion(z_sig, z_ts)      # (B, fusion.output_dim)
        return self.head(z)                    # (B, num_classes)
```

O gradiente flui naturalmente para os dois ramos via backprop normal. Nenhuma mudança no trainer
é necessária.

---

## Implementações concretas planejadas

### Signal Encoders

| Nome | Descrição | Dimensões relevantes |
|------|-----------|----------------------|
| `CNNLSTMEncoder` | Migra o branch do `HybridCNNLSTM` (2 conv blocks + BiLSTM 2 camadas, last timestep) | CNN out=128, LSTM hidden=64 → `output_dim = 128` |
| `RobustCNNLSTMEncoder` | Migra o branch do `RobustHybridModel` (3 conv blocks + BiLSTM 1 camada, concat h_n) | CNN out=256, LSTM hidden=128 → `output_dim = 256` |
| `TransformerEncoder` | (futuro) CNN stem + Transformer + CLS token | configurável |
| `PureCNNEncoder` | (futuro) CNN puro sem recorrência, global avg pool | configurável |

### TSFEL Branches

| Nome | Descrição |
|------|-----------|
| `MLPTsfelBranch` | `Linear → BN → ReLU → Dropout` (único bloco). Migra comportamento atual. |
| `DeepMLPTsfelBranch` | (futuro) Duas camadas ocultas. |
| `IdentityTsfelBranch` | (futuro) Sem projeção, passa direto. Útil quando `n_features_tsfel` já tem o tamanho desejado. |

### Fusion Modules

| Nome | Descrição | `output_dim` |
|------|-----------|--------------|
| `ConcatFusion` | Concatenação simples (comportamento atual) | `enc_dim + tsfel_dim` |
| `GatedFusion` | (futuro) Porta de atenção aprendível que pondera os dois ramos antes de concatenar | configurável |
| `AdditiveFusion` | (futuro) Soma elemento-a-elemento (requer `enc_dim == tsfel_dim`) | `enc_dim` |
| `CrossAttentionFusion` | (futuro) Cross-attention entre os dois embeddings | configurável |

### Classification Heads

| Nome | Descrição |
|------|-----------|
| `MLPHead` | `Linear → ReLU → Dropout → Linear` (comportamento atual) |
| `LinearHead` | `Linear` apenas. Mínimo de parâmetros. |
| `DeepMLPHead` | (futuro) Duas camadas ocultas. |

---

## Estrutura de arquivos proposta

```
src/hybrid_activity_recognition/
├── models/
│   ├── hybrid_cnn_lstm/          ← manter para compatibilidade reversa (só re-exporta)
│   ├── robust_hybrid/            ← manter para compatibilidade reversa (só re-exporta)
│   └── modular/                  ← NOVO
│       ├── __init__.py
│       ├── base.py               ← ABCs: SignalEncoder, TsfelBranch, FusionModule, ClassificationHead
│       ├── model.py              ← HybridModel container
│       ├── encoders.py           ← CNNLSTMEncoder, RobustCNNLSTMEncoder, (futuro: Transformer, PureCNN)
│       ├── tsfel_branches.py     ← MLPTsfelBranch, (futuro: Deep, Identity)
│       ├── fusion.py             ← ConcatFusion, (futuro: Gated, Additive, CrossAttention)
│       └── heads.py              ← MLPHead, LinearHead
```

A pasta `layers/` existente (`signal_branch.py`, `tsfel_branch.py`, `fusion.py`, `heads.py`) pode
ser mantida ou ter seus módulos migrados para `models/modular/`. Como os módulos existentes não têm
base class, a migração consiste em:
1. Criar as ABCs em `base.py`
2. Fazer as classes existentes herdarem das ABCs e expor `output_dim`
3. Mover para os novos arquivos

---

## Factory / builder

Para manter a CLI (`main.py`) funcionando sem mudar a assinatura do `build_model`, criar uma função
fábrica em `models/modular/__init__.py`:

```python
def build_hybrid_model(
    encoder_name: str,
    tsfel_branch_name: str,
    fusion_name: str,
    head_name: str,
    num_classes: int,
    n_features_tsfel: int,
    **kwargs,
) -> HybridModel: ...
```

Os nomes antigos (`hybrid_cnn_lstm`, `robust_hybrid`) podem ser mapeados para as combinações
equivalentes de blocos, mantendo backward compatibility no `main.py`.

---

## Notas de implementação

- `output_dim` deve ser `@property` implementado pelas ABCs como `@abstractmethod`, não um
  argumento do construtor. Isso evita dessincronias.
- `FusionModule` recebe os dois `output_dim` no seu `__init__` (via `encoder.output_dim` e
  `tsfel_branch.output_dim`) para poder criar layers internas com o tamanho certo. O `HybridModel`
  é responsável por passar esses valores ao instanciar o fusion.
- O `Trainer` existente não precisa de nenhuma mudança, pois a assinatura
  `model(x_sig, x_feat) → logits` é preservada.
- Inicialização de pesos (Kaiming) pode virar um método utilitário `apply_kaiming_init(module)`
  em `utils/` e ser chamado opcionalmente por qualquer encoder.

---

## O que NÃO muda

- `CalfHybridDataset` e o pipeline de normalização (sem alterações)
- `Trainer` (sem alterações — interface do forward já é compatível)
- `training/loss.py`, `training/metrics.py`, `training/augment.py`
- Formato dos dados de entrada: `(B, 3, T)` para sinais, `(B, n_feats)` para TSFEL
