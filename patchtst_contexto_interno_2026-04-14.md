# Contexto Interno Relevante: PatchTST no Projeto AcTBeCalf

Data: 14 de abril de 2026
Escopo: consolidar, de forma técnica e rastreável, o contexto acumulado sobre resultados, hipóteses, limitações e plano de experimentação para PatchTST neste repositório.

## 1. Resumo Executivo

- O comportamento recente do PatchTST ficou inconsistente em comparação com outras famílias de modelo (ex.: CNN/LSTM e robust variants), especialmente no cenário deep-only após pretreino.
- Há evidência de que o pretreino MAE, como está hoje no pipeline, pode não estar transferindo bem para a tarefa supervisionada final.
- A hipótese mais forte é que a performance está sendo governada por combinação de:
  - capacidade/escala do backbone (d_model, layers, heads),
  - granularidade de patch (patch_length, patch_stride),
  - regime de pretreino (mask_ratio, pretrain_lr, pretrain_epochs),
  - e possíveis limitações estruturais (mistura entre canais e estratégia de pooling/head).
- Decisão operacional: testar primeiro tudo que já é varrível via CLI (sem tocar no src), em busca faseada e com orçamento controlado.

## 2. Evidências Observadas no Projeto

Fonte principal:
- logs de execução em `logs/run_all.log`.

Padrões observados durante a análise:
- Mudança forte de desempenho entre runs de dias diferentes, sugerindo alta sensibilidade a configuração/protocolo.
- Em algumas execuções, PatchTST deep-only colapsa após pretreino, enquanto versões híbridas e/ou outros modelos se mantêm melhores.
- Curvas/indicadores de pretreino com evolução limitada em certos cenários (indicando possível mismatch com o objetivo downstream).

Leitura técnica resultante:
- O problema não parece ser apenas "falta de época".
- O ponto crítico provável é interação entre desenho da arquitetura e regime de treino.

## 3. Mapeamento Técnico do Código (o que já existe)

Arquivos-chave mapeados:
- `src/hybrid_activity_recognition/main.py`
- `src/hybrid_activity_recognition/models/modular/encoders.py`
- `src/hybrid_activity_recognition/training/pretrain_trainer.py`
- `scripts/experiments/_common.sh`
- `scripts/experiments/pretrain_patchtst.sh`
- `scripts/experiments/run_patchtst.sh`

Parâmetros PatchTST já expostos para sweep direto via CLI:
- `patchtst_d_model`
- `patchtst_num_heads`
- `patchtst_num_layers`
- `patchtst_patch_length`
- `patchtst_patch_stride`
- `patchtst_dropout`
- `pretrain_mask_ratio`
- `pretrain_epochs`
- `pretrain_lr`
- `epochs`
- `batch_size`
- `lr`
- `seed`

Limites atuais (não totalmente exploráveis sem adaptar código):
- `channel_attention` e mecanismos explícitos de channel-mixing no encoder atual.
- alternativas de pooling/head para preservar melhor estrutura inter-canais/eixos.
- variações mais profundas de normalização por janela integradas no fluxo principal.
- filtros top-K de classes embutidos no pipeline padrão.

## 4. Hipóteses Prioritárias (para validação experimental)

H1. Escala de modelo domina mais que aumento simples de treino.
- Ajustes de `d_model`, `num_layers`, `num_heads` podem explicar mais variação do que apenas subir `epochs`.

H2. Patching está subótimo para o sinal atual.
- `patch_length` e `patch_stride` podem estar comprimindo ou fragmentando demais padrões temporais relevantes.

H3. Faixa ótima de pretreino é intermediária.
- `pretrain_mask_ratio` e `pretrain_lr` possivelmente têm ótimo em faixa média; extremos podem deteriorar transferência.

H4. Há acoplamento forte entre pretreino e fine-tuning.
- Melhor pretreino não garante melhor downstream se `lr/batch/epochs` supervisionados não forem compatíveis.

## 5. Plano de Grid Search (sem mexer em src)

Estratégia escolhida:
- busca faseada (A -> B -> C -> D), com orçamento rápido na fase inicial.
- evitar grid cartesiano total para não explodir custo sem ganho interpretável.

### Fase A: Arquitetura (orçamento rápido)

Objetivo:
- identificar top configurações estruturais com treino fixo.

Espaço recomendado:
- `d_model`: 64, 128, 256
- `num_layers`: 2, 3, 4
- `num_heads`: 2, 4
- `patch_length`: 4, 8
- `patch_stride`: 4, 8

Fixos nesta fase:
- `pretrain_epochs=100`
- `pretrain_lr=1e-3`
- `pretrain_mask_ratio=0.4`
- `epochs=150`
- `batch_size=256`
- `lr=1e-3`
- `seed=42`

Execução prática (alvo):
- 12-24 runs (subamostrado dentro do espaço acima), com seleção de top-N.

### Fase B: Pretreino (em cima do top-N da Fase A)

Espaço recomendado:
- `pretrain_mask_ratio`: 0.2, 0.4, 0.6
- `pretrain_lr`: 5e-4, 1e-3, 2e-3
- `pretrain_epochs`: 50, 100, 150

### Fase C: Fine-tuning (em cima dos vencedores da Fase B)

Espaço recomendado:
- `lr`: 5e-4, 1e-3, 2e-3
- `batch_size`: 128, 256
- `epochs`: 100, 150, 200

### Fase D: Robustez/Reprodutibilidade

Espaço recomendado:
- `seed`: 42, 123, 456

Objetivo:
- separar ganho real de variância de seed.

## 6. Métricas, Ranking e Critérios de Avanço

Métricas principais:
- macro F1 (primária)
- accuracy (secundária)

Critério de avanço entre fases:
- selecionar top-N (recomendado N=3) por macro F1.
- desempate por estabilidade (variação menor entre runs/seeds).

Sinais de alerta:
- aumento de train sem ganho em val/test (overfit).
- ganho pequeno porém instável entre seeds.
- sensibilidade extrema a um único hiperparâmetro.

## 7. Logging, Rastreabilidade e Execução Longa

Diretrizes operacionais:
- nomes de run codificando hiperparâmetros centrais (fácil grep e comparação).
- log por suíte + log por execução.
- saída consolidada para ranking tabular final.
- manter padrão compatível com nohup para runs longos.

Estado atual do ambiente observado:
- execução longa em background já usada no projeto (`nohup ... run_all.sh ...`).

## 8. O que foi decidido até aqui

Decisões ativas:
- foco em experimento isolado e controlado antes de qualquer refactor em `src`.
- começar por busca faseada com orçamento rápido.
- usar evidência empírica para priorizar futuras mudanças estruturais.

Não decidido ainda (próxima rodada):
- forma exata de amostragem da Fase A (subset específico das combinações).
- regra de early-stop de campanha (tempo limite por suíte, limite de runs ruins consecutivos, etc.).

## 9. Próximos Passos Recomendados

1. Materializar scripts da busca faseada em árvore isolada (ex.: `ablations/patchtst`).
2. Executar Fase A (12-24 runs) e consolidar ranking.
3. Rodar Fase B e C apenas com top-3.
4. Validar robustez em seeds múltiplas (Fase D).
5. Só então decidir mudanças estruturais em `src` (channel-mixing, pooling/head, etc.).

## 10. Referências Internas

- `patchtst_study.md`
- `logs/run_all.log`
- `scripts/experiments/_common.sh`
- `scripts/experiments/pretrain_patchtst.sh`
- `scripts/experiments/run_patchtst.sh`
- `src/hybrid_activity_recognition/main.py`
- `src/hybrid_activity_recognition/models/modular/encoders.py`
- `src/hybrid_activity_recognition/training/pretrain_trainer.py`

Plano:

Plan: Ablações PatchTST Isoladas
Vamos criar uma árvore nova em ablations/patchtst, sem tocar em src, copiando a lógica de execução atual (scripts, naming, resume, logs) e separando suites por tipo de teste, com um runner único para nohup.

Steps

Fase Base: criar a raiz ablations/patchtst com subpastas para scripts, preprocess, logs, results e docs.
Fase Base: criar um _common.sh local espelhando a infraestrutura atual (paths, run naming, resume por checkpoint, DONE marker, tee de logs).
Fase Suite: criar comando dedicado para teste sem pretreino (supervisionado deep_only sem checkpoint), com sweep opcional de seed.
Fase Suite: criar comando dedicado para teste de tamanho de modelo (grade controlada de d_model, layers, heads, dropout, lr).
Fase Suite: criar comando dedicado para teste de normalização, com preprocess local que gera variantes de parquet e dispara os treinos por variante.
Fase Suite: criar comando dedicado para teste top-k (K = 5, 7, 10, 13, 15, 18), filtrando classes mais frequentes no treino e aplicando filtro consistente no teste.
Fase Suite: criar comando dedicado para teste de mixing em dois blocos.
Linha seguinte: bloco imediato com o que já é possível pela CLI atual.
Linha seguinte: bloco marcado como pendente técnica para channel attention e pooling alternativo, implementado apenas dentro de ablations/patchtst, sem alterar src.
Fase Orquestração: criar um comando único run_all_patchtst_ablations.sh que chama todas as suites em ordem e grava log mestre para uso com nohup.
Fase Relatório: criar sumarizador local para extrair acc e macro_f1 dos logs e gerar tabela final em results.
Fase Docs: criar README da raiz de ablações com comando de cada suite + comando único + exemplo de execução com nohup.
Comandos que o plano vai entregar

run_teste_sem_pretreino.sh
run_teste_tamanho_modelo.sh
run_teste_mixing.sh
run_teste_normalizacao.sh
run_teste_topk.sh
run_all_patchtst_ablations.sh
Relevant files

patchtst_study.md — requisitos científicos e hipóteses.
scripts/experiments/_common.sh — padrão de helper, naming e resume.
scripts/experiments/run_patchtst.sh — fluxo atual patchtst.
scripts/experiments/pretrain_patchtst.sh — fluxo de pretreino.
src/hybrid_activity_recognition/main.py — flags já disponíveis.
src/hybrid_activity_recognition/models/modular/encoders.py — hardcode atual de channel attention e pooling.
src/hybrid_activity_recognition/training/pretrain_trainer.py — hardcode atual no pretreino.
src/hybrid_activity_recognition/data/pretrain_dataset.py — normalização global no pretreino.
src/hybrid_activity_recognition/data/dataloader.py — normalização atual no supervisionado.
Verification

Cada suite, em smoke run curto, precisa gerar output_dir, train.log, checkpoint, best e DONE.
O runner único precisa gerar log mestre e logs por suite em ablations/patchtst/logs.
O teste sem pretreino precisa comprovar que rodou sem checkpoint.
O teste de tamanho de modelo deve alterar apenas hiperparâmetros de arquitetura.
Top-k e normalização devem registrar metadados da variante usada em cada run.
O sumarizador precisa gerar tabela final comparando suites, variantes e seeds.
Decisions

Escopo desta fase: 100% isolado em ablations/patchtst.
Nada em src será alterado agora.
O que a CLI já suporta será feito só com shell orchestration.
O que não está exposto na CLI será implementado apenas dentro da nova árvore de ablações.