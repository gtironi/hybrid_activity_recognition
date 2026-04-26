# PatchTST – Notas para Experimentos em HAR (AcTBeCalf)

## 1. Notas do artigo original

**Referência:**
- Yuqi Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers* (ICLR 2023, PatchTST).

### 1.1 Objetivo e contexto
- Foco principal: *long-term forecasting* em séries temporais multivariadas (benchmarks ETT, Electricity, Exchange, Weather, Traffic, Illness etc.).
- Secundário: auto-supervisão (self-supervised) e transferência de representações para outras tarefas de forecasting.
- Não é um artigo de HAR, mas a arquitetura é genérica para séries temporais contínuas, o que é compatível com sensores inerciais.

### 1.2 Ideias centrais de arquitetura
- **Patching (subseries-level tokens)**:
  - O histórico de tamanho $L$ é segmentado em patches de comprimento $P$ (e stride $S$), produzindo $N$ patches por canal.
  - Cada patch é um pequeno sub-sinal local ("64 palavras" → analogia com NLP/ViT).
  - Benefícios: (i) retém semântica local; (ii) reduz custo quadrático da atenção (menos tokens); (iii) permite olhar janelas mais longas histórico, mantendo complexidade controlada.
- **Channel-independence**:
  - Cada canal (variável) é tratado como uma série univariada.
  - Um único backbone Transformer é compartilhado entre canais (mesmos pesos), com projeção de entrada e cabeça específicas por tarefa.
  - Isso simplifica a modelagem multivariada e melhora generalização quando há muitos canais.
- **Encoder Transformer simples**:
  - Backbone é basicamente um Transformer encoder padrão (multi-head self-attention, FFN, LayerNorm, residuals), sem arquiteturas complexas como Autoformer/FEDformer.
  - Patching + channel-independence são as mudanças estruturais chave.

### 1.3 Detalhes de entrada/saída relevantes
- **Entrada**:
  - Para forecasting: janelas de contexto de tamanho $L$ (ex.: 96, 192, 336, 512). Cada canal → sequência univariada.
  - Essa sequência é dividida em patches de tamanho $P$ (ex.: 16, 24) e stride $S$ (normalmente $S = P$ ou um valor fixo). Cada patch é projetado linearmente em um embedding $d_{model}$.
  - Posição é representada via embeddings posicionais 1D sobre patches.
- **Saída**:
  - Para forecasting: gera embeddings de patches e os projeta para horizontes de previsão (multi-step). Existem cabeças específicas para horizontes diferentes.
  - Para *representation learning*, a saída de patches pode ser agregada (p.ex. pooling) e usada em outras tarefas (fine-tuning).
  - Para fine-tuning em classificação/regressão, a prática geral é usar pooling (global average) ou um token especial para obter um vetor fixo.

### 1.4 Normalização e pré-processamento
- **Normalização por canal (z-score)**:
  - O artigo discute a importância de normalizar cada variável (canal) com média/variância do *treino*.
  - Em benchmarks de forecasting, é comum usar normalização global por canal (usar média/variância calculadas sobre todo o conjunto de treino para aquele canal).
- **RevIN / normalizações avançadas**:
  - PatchTST em si não depende estruturalmente de RevIN, mas o artigo discute que normalização adequada é crítica para bom desempenho em long-term forecasting.
  - Em muitos experimentos, normalização simples por canal já é suficiente, desde que aplicada de forma consistente entre treino/val/test.
- **Relevância para o nosso caso**:
  - Para HAR com janelas curtas de sensores (como AcTBeCalf), normalização só global pode ser insuficiente quando há variação lenta de nível por sujeito/sessão.
  - Estratégias mais locais (por janela, por sujeito, ou combinação global+local) podem ser importantes para permitir que o PatchTST foque mais em formas/variações do sinal, não em offsets.

### 1.5 Self-supervised pretraining (auto-supervisão)
- O artigo propõe usar PatchTST como backbone para tarefas de **reconstrução/forecasting mascarados**:
  - Tarefas de mascaramento de patches (similar ao masked language modeling / masked image modeling): partes da sequência são mascaradas e o modelo aprende a reconstruir.
  - Tarefas de forecasting auto-supervisionado (predizer horizontes futuros a partir do contexto) também são exploradas.
- **Resultado central**:
  - Em grandes datasets de séries temporais, pré-treinar PatchTST com tarefas auto-supervisionadas e depois fazer fine-tuning supervisionado pode superar o treinamento puramente supervisionado.
  - Transfer learning: representações pré-treinadas em um dataset podem ser transferidas para outros datasets de forecasting, com boa performance.
- **Relevância para nós**:
  - Em HAR, pré-treino auto-supervisionado em janelas de acelerômetro (reconstruir partes da janela, predizer futuros curtos) pode ajudar, mas:
    - É crucial que a *tarefa de pretreino*, a *normalização* e a *janela de contexto* sejam coerentes com a tarefa final de classificação.
    - Se a loss de reconstrução/forecasting não desce (ou não aprende estrutura útil), o pretreino pode virar ruído e atrapalhar o fine-tuning.

### 1.6 Hiperparâmetros importantes
- **Comprimento do contexto ($L$)**:
  - PatchTST se beneficia de contextos relativamente longos (até centenas de passos), porque patching reduz custo da atenção.
  - Para HAR, isso se traduz em quantos segundos de histórico queremos considerar (ex.: 2s, 4s, 10s), ajustando compatibilidade com AcTBeCalf.
- **Comprimento do patch ($P$) e stride ($S$)**:
  - Valores típicos em forecasting: $P ∈ \{16, 24\}$, stride frequentemente igual a $P$.
  - Patches muito curtos → muitos tokens, custo alto; patches muito longos → perda de resolução local.
- **Dimensão do embedding ($d_{model}$), número de camadas e heads**:
  - No paper, o default para datasets "médios/grandes" é aproximadamente: $d_{model} = 128$, 3 camadas Transformer, 16 heads, dropout em torno de 0.2.
  - Para datasets menores (ILI, ETTh1, ETTh2), eles reduzem o tamanho do modelo para algo como $d_{model} = 16$, 4 heads, mantendo 3 camadas, justamente para mitigar overfitting.
  - Em geral, overparameterização pode dificultar treinamento quando o dataset é pequeno (caso típico em HAR), então faz sentido testar tamanhos menores de modelo no nosso cenário.

### 1.7 Resultados empíricos relevantes
- PatchTST supera Transformers anteriores (Autoformer, FEDformer, Informer etc.) em várias benchmarks de forecasting.
- Pré-treino auto-supervisionado mostra ganhos maiores em datasets grandes.
- Em tarefas menores, ganhos do pré-treino são mais modestos – indicando que a utilidade do pré-treino depende fortemente de:
  - tamanho do dataset,
  - qualidade da tarefa auto-supervisionada,
  - alinhamento entre pretreino e tarefa downstream.

### 1.8 Consequências diretas para o nosso HAR
- O artigo sugere que **dois pontos são cruciais**:
  1. **Desenho do patch** (tamanho, stride, contexto) precisa respeitar a escala temporal dos padrões relevantes (p.ex., ciclos de caminhada, repetições de exercício).
  2. **Normalização consistente** (idealmente por canal, eventualmente combinando global + local) é crítica para estabilidade de treinamento.
- Além disso:
  - Em problemas menores, **treinar do zero** um PatchTST bem-configurado pode ser tão bom (ou melhor) que pretreino mal configurado.
  - Se o pretreino não converge bem (loss quase constante, pouca queda), é melhor desconfiar da tarefa de pretreino/normalização antes de culpar a arquitetura.

## 2. Hipóteses e perguntas para experimentos em AcTBeCalf

Esta seção não assume nada como "verdade absoluta"; são **questões** guiadas pelo paper original e pelo nosso setup de HAR (PatchTST deep-only, sem TSFEL) que queremos responder com experimentos.

### 2.1 Channel-independence vs mistura de canais

- No paper, PatchTST é **channel-independent**: cada canal é tratado como série univariada, com pesos compartilhados entre séries.
- No nosso wrapper HuggingFace atual:
  - Usamos `channel_attention=False` no `PatchTSTConfig`.
  - A saída `last_hidden_state` tem shape `(B, C, num_patches, d_model)` e depois fazemos média em canais e patches → `(B, d_model)` antes do head (`MLPHead` ou `LinearHead`).
- Perguntas que queremos testar empiricamente:
  - O que acontece se **misturarmos canais explicitamente**?
    - Variante A: ligar `channel_attention=True` no backbone (deixar o modelo aprender atenção entre canais).
    - Variante B: mudar o pooling final para **concatenar canais** (por exemplo, pooling em patches mas manter `C * d_model` como vetor de entrada do head), deixando o head aprender a mistura entre eixos.
  - Em HAR com apenas 3 eixos (x,y,z), será que a filosofia de channel-independence continua vantajosa, ou essa falta de mistura explícita entre eixos está reduzindo nossa acurácia final?

### 2.2 Tamanho do modelo (d_model, heads, camadas)

- O artigo mostra dois regimes:
  - Modelos padrão: $d_{model} = 128$, 3 camadas, 16 heads, dropout ≈ 0.2.
  - Modelos "compactos" para datasets pequenos: $d_{model} = 16$, 3 camadas, 4 heads (e MLP menor), para reduzir overfitting.
- Nosso dataset HAR (AcTBeCalf) é relativamente pequeno comparado a Weather/Traffic/Electricity.
- Perguntas que queremos transformar em experimentos:
  - O que acontece se **mantivermos** um PatchTST maior (ex.: $d_{model} = 128$, 3 camadas, 8–16 heads) vs. **diminuirmos** (ex.: $d_{model} ∈ \{32, 16\}$, heads menores)?
  - Como isso afeta:
    - convergência da loss de pretreino (MAE mascarado),
    - overfitting na classificação (gap treino/val/test),
    - métricas finais (acc, macro-F1) nas classes de AcTBeCalf?
- A hipótese, inspirada pelo paper, é que modelos muito grandes podem estar facilitando overfitting e tornando o pretreino pouco estável; mas vamos tratar isso como algo a ser **medido** (não assumido) via ablações.

### 2.3 Normalização global vs por janela (RevIN‑like)

- O artigo discute que **instance normalization** (RevIN) — normalizar cada instância/janela individualmente para média 0 e desvio 1 e depois desnormalizar — ajuda a mitigar shift de distribuição entre treino e teste.
- Hoje, nosso pretreino usa normalização **global** por canal (média/std no dataset de treino) em [src/hybrid_activity_recognition/data/pretrain_dataset.py](src/hybrid_activity_recognition/data/pretrain_dataset.py), o que pode não capturar bem variações por sujeito/sessão.
- Perguntas para experimento:
  - O que acontece com a curva de loss de pretreino se mudarmos para **normalização por janela** (por exemplo, z-score por `(B, C, T)` antes de alimentar o PatchTST, estilo RevIN, sem ou com desnormalização na saída)?
  - Essa mudança melhora a estabilidade da loss (deixa de ficar "travada" em ~0.98) e a qualidade das representações para classificação downstream?

### 2.4 Pré-treino vs treinamento do zero

- O paper mostra ganhos claros de pré-treino em datasets grandes, mas ganhos menores (ou neutros) em datasets pequenos.
- No nosso caso, já observamos que:
  - a loss de pretreino parece não descer tanto quanto gostaríamos,
  - o PatchTST deep-only com checkpoint pré-treinado ainda generaliza pior que CNN‑LSTM em alguns cenários.
- Perguntas para ablações futuras:
  - Comparar, com tudo igual (normalização, tamanho de modelo, splits):
    - PatchTST **treinado do zero** vs
    - PatchTST **com pré-treino MAE**, medindo acurácia/macro-F1 e comportamento de overfitting.
  - Ver se algum regime de normalização + tamanho de modelo torna o pré-treino realmente vantajoso.
