# Review Técnico Do `src/` Para O Artigo

Escopo desta revisão: apenas o código atual em `src/hybrid_activity_recognition/`, assumindo que a tese central do artigo é:

> "Uma arquitetura híbrida, combinando ramo de sinal bruto e ramo de features TSFEL, funciona melhor do que cada ramo isolado."

Este documento está escrito como se eu fosse um reviewer tentando avaliar se o repositório, no estado atual, sustenta bem essa tese.

## Resumo Executivo

Hoje, o repositório já tem uma proposta arquitetural clara:

- ramo de sinal com `CNN1D + BiLSTM`;
- ramo TSFEL com `Linear + BatchNorm + ReLU + Dropout`;
- fusão por concatenação;
- cabeça MLP para classificação;
- modos `supervised`, `finetune`, `fixmatch` e `test`.

O problema principal não é "a arquitetura não faz sentido". Ela faz.

O problema principal é de **credibilidade experimental**. No estado atual, os maiores riscos são:

1. o protocolo de split ainda pode ser fraco para HAR se não for claramente por sujeito/sessão;
2. a tese "híbrido > isolados" ainda não está devidamente testada, porque o `src/` atual só expõe modelos híbridos;
3. o ganho do híbrido pode ser confundido com mais capacidade, mais informação de entrada ou pipeline de treino diferente;
4. o modo `fixmatch` adiciona complexidade metodológica antes de a tese principal estar limpa e bem estabelecida.

Se eu estivesse revisando o paper, eu diria:

- a ideia é plausível;
- o código está organizado o suficiente para suportar um paper;
- mas faltam ablações e protocolo experimental para transformar isso em evidência convincente.

## O Que A Arquitetura Atual Realmente É

### Modelos disponíveis

Hoje o `src/` expõe dois modelos:

- `HybridCNNLSTM`
- `RobustHybridModel`

Ambos são híbridos. Não há, no `src/`, versões "signal-only" ou "TSFEL-only" prontas para treino via `main.py`.

### Ramo de sinal

O ramo de sinal é baseado em convoluções 1D seguidas de BiLSTM:

- `HybridCNNLSTMSignalBranch`: menor, mais didático;
- `RobustCNNLSTMSignalBranch`: CNN mais profunda e BiLSTM de 1 camada.

### Ramo TSFEL

O ramo TSFEL não entra cru na fusão. Ele passa por uma projeção MLP curta:

- `Linear(in_features, hidden_dim)`
- `BatchNorm1d`
- `ReLU`
- `Dropout`

Isso significa que o híbrido não é apenas "concatenar TSFEL com sinal"; ele aprende uma representação tabular antes da fusão.

### Fusão

A fusão atual é **concatenação simples**:

- `z = torch.cat((z_sig, z_ts), dim=1)`

Não há:

- atenção entre ramos;
- mecanismo de gating;
- soma ponderada aprendida;
- cross-modal interaction explícita antes da cabeça.

### Cabeça

Depois da concatenação, a classificação é feita por uma MLP curta:

- `Linear(fusion_dim, hidden_dim)`
- `ReLU`
- `Dropout`
- `Linear(hidden_dim, num_classes)`

Isso é importante para o paper porque, hoje, a mistura efetiva entre modalidades acontece quase toda nesta cabeça.

## Julgamento Como Reviewer

### O Que Faz Sentido E É Defensável

As seguintes escolhas são defensáveis e coerentes:

- combinar sinal bruto com features manuais TSFEL;
- usar TSFEL como ramo complementar, e não como substituto do ramo temporal;
- projetar TSFEL para uma dimensão fixa antes da fusão;
- usar uma fusão simples primeiro, antes de testar mecanismos mais sofisticados;
- reportar `macro-F1`, e não apenas accuracy.

Em outras palavras: a **ideia-base do híbrido é válida**.

### O Que Ainda Não Está Convincente

Se a tese é "o híbrido funciona melhor que sozinho", então o código ainda não demonstra isso com rigor, porque:

1. não há baseline interno forte para o ramo de sinal sozinho;
2. não há baseline interno forte para TSFEL sozinho;
3. não está isolado o efeito da fusão vs o efeito de mais parâmetros;
4. não está isolado o efeito da arquitetura vs o efeito do regime de treino (`supervised`, `finetune`, `fixmatch`).

Como reviewer, eu aceitaria a arquitetura como candidata, mas ainda não aceitaria a conclusão forte.

## Principais Problemas Hoje

## 1. Problema Mais Grave: protocolo de split pode não ser suficiente para HAR

O caminho legado `--labeled_parquet` faz split estratificado por label em janelas.

Isso é perigoso em HAR quando múltiplas janelas vêm do mesmo animal, sessão ou trecho contínuo. Nesse cenário, treino e teste podem partilhar padrões muito parecidos do mesmo indivíduo, o que geralmente infla as métricas.

Mesmo no pipeline com `train.parquet` e `test.parquet` separados, ainda há um ponto delicado:

- se a validação vier de `val_fraction` dentro de `df_train`, ela continua sendo um split por label, não necessariamente por grupo.

Para paper, isto precisa ficar muito claro:

- o split precisa ser por sujeito, por sessão, por grupo temporal, ou por um protocolo que impeça vazamento semântico;
- se isso já é garantido antes do `src/`, o artigo precisa explicar isso explicitamente;
- se não é garantido, a principal ameaça à validade externa está aqui.

### O que eu cobraria no paper

- tabela com número de sujeitos em treino/val/test;
- tabela com número de janelas por split;
- explicação explícita de que nenhum sujeito aparece em mais de um split, se esse for o caso;
- se houver validação interna, ela também deve respeitar agrupamento.

## 2. A tese central ainda não está testada de forma justa

O repositório atual treina apenas híbridos. Isso não basta para afirmar que o híbrido supera os ramos isolados.

Você precisa de pelo menos estes três experimentos supervisionados, no mesmo split:

1. `signal-only`
2. `tsfel-only`
3. `hybrid`

E esses três precisam ser comparados:

- com o mesmo protocolo de dados;
- com o mesmo número de seeds;
- com as mesmas métricas;
- com controle razoável de capacidade.

Sem isso, um reviewer pode dizer:

> "Talvez o ganho venha apenas de dar mais informação ao modelo, ou de aumentar a dimensão da representação, e não de uma fusão realmente útil."

## 3. Hoje a comparação entre `HybridCNNLSTM` e `RobustHybridModel` não prova a tese

Esses dois modelos não são uma ablação limpa do mecanismo híbrido.

Ao comparar os dois, você muda ao mesmo tempo:

- a profundidade do ramo de sinal;
- o tamanho do `hidden_lstm`;
- a dimensão do ramo TSFEL;
- o dropout;
- a inicialização.

Isso é útil para procurar um modelo melhor, mas não para responder:

> "A componente híbrida ajuda?"

Ou seja:

- comparar os dois ajuda a escolher **qual híbrido** é melhor;
- não ajuda a provar que **ser híbrido** é melhor.

## 4. `FixMatch` hoje parece secundário para a tese principal

O `src/` tem suporte a `fixmatch`, o que é interessante, mas para o artigo atual isso pode mais atrapalhar do que ajudar, se a tese principal ainda não estiver consolidada.

Como reviewer, eu perguntaria:

> "O paper é sobre arquitetura híbrida supervisionada, ou sobre aprendizado semi-supervisionado em HAR?"

Se a resposta for a primeira, então `FixMatch` deve aparecer como:

- experimento secundário;
- extensão opcional;
- ou material suplementar.

Não como base da mensagem principal.

Além disso, há um ponto metodológico real:

- o não rotulado é normalizado com estatísticas próprias do conjunto não rotulado;
- o rotulado é normalizado com estatísticas do treino rotulado.

Isso pode introduzir desalinhamento entre os dois fluxos.

## 5. Remoção de classes raras precisa ser muito bem reportada

O loader remove classes raras do treino abaixo de um mínimo de contagem. Depois, exemplos de validação/teste com labels fora do encoder são descartados.

Isso pode ser aceitável como engenharia para treinar, mas em artigo científico precisa ser muito explícito:

- quantas classes originais existiam;
- quantas foram removidas;
- quantas janelas foram descartadas em val/test;
- por que isso foi feito;
- se os resultados são reportados no problema completo ou num subproblema filtrado.

Se isso ficar implícito, parece manipulação involuntária do protocolo.

## 6. A fusão atual é simples demais para uma alegação forte sobre "melhor arquitetura híbrida"

Hoje a fusão é concatenação seguida por MLP.

Isso é um baseline sólido, mas se você quer argumentar não só que "o híbrido ajuda", mas também que encontrou a **melhor arquitetura híbrida**, então falta explorar alternativas relevantes de fusão.

Sem isso, a conclusão mais honesta hoje seria:

> "Uma arquitetura híbrida simples baseada em concatenação já melhora os baselines unimodais."

Isso é defensável.

Já afirmar:

> "Esta é a melhor arquitetura híbrida"

não está sustentado ainda.

## Ablações Mais Relevantes

Se eu tivesse de priorizar poucas ablações com maior retorno científico, seriam estas.

## A. Ablação central: unimodal vs híbrido

Obrigatórias:

1. `signal-only`
2. `tsfel-only`
3. `hybrid`

Essas três já precisam existir no paper principal.

### Como eu faria de modo justo

- `signal-only`: usar exatamente o mesmo ramo de sinal do híbrido e uma cabeça equivalente;
- `tsfel-only`: usar exatamente o mesmo ramo TSFEL e uma cabeça equivalente;
- `hybrid`: usar o modelo atual.

Assim você reduz o argumento de reviewer de que o baseline foi artificialmente enfraquecido.

## B. Ablação de fusão

Você precisa testar pelo menos:

1. concatenação simples (baseline atual);
2. concatenação + gating escalar por ramo;
3. soma ponderada aprendida, se as dimensões forem alinhadas;
4. eventualmente uma atenção simples entre ramos, se quiser uma opção mais moderna.

Não precisa virar um paper de fusão multimodal. Mas pelo menos duas ou três variantes fazem sentido.

### O que isso responde

Responde se o ganho vem:

- de ter duas fontes de informação;
- ou de um mecanismo de interação mais inteligente entre elas.

## C. Ablação do ramo TSFEL

Hoje o ramo TSFEL usa projeção linear para `hidden_dim`.

Isso sugere testes simples:

1. TSFEL cru direto para a cabeça;
2. TSFEL com projeção linear atual;
3. TSFEL com MLP de duas camadas.

Se a projeção atual já ganhar, você consegue defender:

> "Não basta concatenar features feitas à mão; vale aprender uma representação delas."

## D. Ablação do ramo de sinal

Você hoje tem duas versões do ramo de sinal, mas dentro de dois híbridos diferentes.

O ideal é isolar:

1. backbone de sinal menor;
2. backbone de sinal mais robusto;
3. mesmo ramo TSFEL;
4. mesma cabeça.

Isso ajuda a responder:

- o ganho vem da multimodalidade?
- ou vem quase todo da robustez do backbone temporal?

## E. Ablação do regime de treino

Para o argumento principal, eu faria a história em camadas:

1. primeiro, mostrar que o híbrido supervisionado funciona;
2. depois, opcionalmente, mostrar se `finetune` melhora;
3. por último, opcionalmente, mostrar se `fixmatch` melhora.

Não misture logo no início:

- arquitetura;
- pré-aquecimento;
- semi-supervisão;
- diferentes losses;
- diferentes checkpoints.

Isso dificulta a causalidade.

## F. Ablação de orçamento

Esta é muito importante e frequentemente ignorada.

Se o híbrido tem mais parâmetros e recebe mais informação, ele naturalmente pode ganhar.

Então eu testaria:

1. híbrido atual;
2. signal-only com cabeça aumentada para orçamento similar;
3. tsfel-only com cabeça aumentada para orçamento similar.

Não precisa ser perfeito, mas deve haver uma tentativa honesta de controlar capacidade.

## O Que Eu Testaria A Mais

## 1. Múltiplas seeds

Hoje o código aceita seed, mas para paper eu reportaria:

- média;
- desvio-padrão;
- idealmente intervalo de confiança.

No mínimo 3 seeds. Melhor 5, se couber no tempo.

Sem isso, qualquer ganho pequeno fica pouco confiável.

## 2. Resultados por classe

Além de accuracy e macro-F1, eu incluiria:

- F1 por classe;
- matriz de confusão;
- discussão de quais comportamentos o híbrido ajuda mais.

Isto é especialmente importante porque a principal narrativa do híbrido costuma ser:

> "cada modalidade resolve ambiguidades diferentes"

Sem análise por classe, essa narrativa fica só intuitiva.

## 3. Teste estatístico ou pelo menos efeito consistente

Se o ganho do híbrido for pequeno, eu tentaria mostrar consistência:

- híbrido ganha em quase todas as seeds;
- híbrido ganha em quase todos os folds ou sujeitos;
- híbrido melhora particularmente as classes difíceis.

Sem isso, um reviewer pode dizer que é ruído experimental.

## 4. Baseline clássico forte

Como você usa TSFEL, eu certamente incluiria pelo menos um baseline clássico:

- `RandomForest` sobre TSFEL;
- ou `SVM` sobre TSFEL.

Motivo: isso ancora o paper no mundo tradicional de HAR com features manuais.

Se o híbrido não superar um RF forte em TSFEL, isso é uma discussão importante.
Se superar, isso dá muita credibilidade.

## 5. Probe com embedding aprendido

Como experimento secundário, faz sentido testar:

- extrair o embedding fundido do deep model;
- treinar um `RandomForest` ou `LogisticRegression` em cima desse embedding.

Isso não substitui o modelo principal, mas ajuda a entender:

- se a representação aprendida é boa;
- se o MLP está realmente sendo a melhor cabeça;
- ou se o backbone está gerando um espaço já facilmente separável.

## 6. Robustez a ausência de modalidade

Se quiser fortalecer muito o argumento híbrido, eu faria:

- treino híbrido normal;
- teste com TSFEL zerado;
- teste com sinal zerado;
- teste normal.

Isso mostra dependência e complementaridade entre modalidades.

É um experimento forte quando bem interpretado.

## O Que Eu Não Priorizaria Agora

Para o artigo atual, eu não gastaria muita energia com:

- dezenas de variações finas dos hiperparâmetros de augmentação do `FixMatch`;
- comparar muitas taxas de dropout sem hipótese clara;
- testar uma grande família de losses se a pergunta principal é arquitetural;
- explorar mecanismos muito sofisticados de fusão antes de estabelecer concatenação vs unimodal;
- substituir a cabeça por modelos não diferenciáveis como eixo principal do paper.

Essas coisas podem entrar depois, mas têm retorno científico menor agora.

## O Que Faz Sentido Dizer No Paper Hoje

No estado atual do código, a versão mais honesta e forte da narrativa seria:

1. existe uma arquitetura híbrida clara e bem definida;
2. ela combina representações complementares de sinal bruto e TSFEL;
3. a primeira hipótese a testar é se isso supera ramos isolados sob protocolo rigoroso;
4. a concatenação é um baseline de fusão simples, interpretável e sólido;
5. variantes mais robustas e semi-supervisionadas são extensões, não a prova principal.

Ou seja: a mensagem central do paper deve começar simples.

## O Que Eu Criticaria Numa Submissão Se Não For Tratado

Se o artigo fosse submetido hoje sem novos testes, minhas críticas principais seriam:

1. **Falta baseline unimodal justo.**
   Sem `signal-only` e `tsfel-only`, a tese central fica incompleta.

2. **Risco de vazamento ou protocolo de split insuficiente para HAR.**
   Se não houver split por sujeito/sessão claramente demonstrado, eu desconfiaria bastante das métricas.

3. **Capacidade e informação não foram controladas.**
   O híbrido pode ganhar simplesmente por ver mais informação e ter representação maior.

4. **A conclusão sobre "melhor arquitetura híbrida" é forte demais para a evidência atual.**
   Hoje o código mostra uma arquitetura híbrida plausível, não ainda a melhor.

5. **FixMatch pode confundir a narrativa principal.**
   Se entrar cedo demais, parece que o paper tenta vender ao mesmo tempo arquitetura, SSL e pipeline de treino.

## Plano Experimental Recomendado

Se eu tivesse de transformar este repositório num paper convincente com o menor número de experimentos adicionais, eu faria:

## Fase 1: credibilidade básica

1. definir split rigoroso por sujeito/sessão;
2. rodar `signal-only`, `tsfel-only` e `hybrid`;
3. repetir com 3 a 5 seeds;
4. reportar `macro-F1`, `weighted-F1`, accuracy e matriz de confusão.

## Fase 2: provar qual híbrido é melhor

1. comparar `HybridCNNLSTM` vs `RobustHybridModel`;
2. controlar minimamente o orçamento;
3. testar 2 ou 3 variantes de fusão;
4. testar ablação do ramo TSFEL com e sem projeção.

## Fase 3: extensões

1. `finetune`;
2. `fixmatch`;
3. embedding + classificador clássico;
4. robustez quando uma modalidade falha.

## Minha Recomendação Final

Se o objetivo é **dar credibilidade científica** para a tese do híbrido, a prioridade não é inventar uma arquitetura mais complexa agora.

A prioridade é:

1. **protocolo de split forte**;
2. **baselines unimodais justos**;
3. **ablação de fusão simples mas limpa**;
4. **múltiplas seeds e análise por classe**.

Se isso for bem feito, mesmo uma fusão por concatenação já pode render um paper defensável.

Se isso não for feito, mesmo uma arquitetura híbrida mais sofisticada continuará vulnerável como submissão.

## Decisão Como Reviewer

### Estado atual

- arquitetura: promissora;
- organização do código: boa;
- tese principal: ainda insuficientemente demonstrada;
- risco metodológico: moderado a alto, dependendo do split real.

### Para ficar convincente

O mínimo que eu exigiria para acreditar na conclusão principal seria:

1. split por sujeito/sessão claramente demonstrado;
2. `signal-only` vs `tsfel-only` vs `hybrid`;
3. resultados em múltiplas seeds;
4. análise por classe;
5. comparação entre variantes híbridas com narrativa causal limpa.

Se você fizer isso, o trabalho deixa de ser "uma ideia boa com código razoável" e passa a ser "uma evidência experimental crível de que o híbrido ajuda".
