# PatchTST Ablations (Isolado)

Este diretório executa ablações do PatchTST sem alterar `src`.

## Estrutura

- `scripts/`: orquestração das suítes.
- `preprocess/`: geração de variantes de dados (normalização e top-k).
- `logs/`: logs por suíte e log mestre.
- `results/`: consolidação final.

## Suítes disponíveis

- `scripts/run_teste_sem_pretreino.sh`
- `scripts/run_teste_tamanho_modelo.sh`
- `scripts/run_teste_mixing.sh`
- `scripts/run_teste_normalizacao.sh`
- `scripts/run_teste_topk.sh`
- `scripts/run_all_patchtst_ablations.sh`
- `scripts/run_smoke_patchtst_ablations.sh`

## Execução rápida

Do diretório raiz do projeto:

```bash
bash ablations/patchtst/scripts/run_teste_sem_pretreino.sh
bash ablations/patchtst/scripts/run_teste_tamanho_modelo.sh
```

Execução completa em background:

```bash
nohup bash ablations/patchtst/scripts/run_all_patchtst_ablations.sh > ablations/patchtst/logs/run_all_patchtst_ablations.log 2>&1 &
```

Smoke run (pipeline completo em modo reduzido):

```bash
bash ablations/patchtst/scripts/run_smoke_patchtst_ablations.sh
```

## Variáveis de ambiente úteis

- `DEVICE` (default: `cuda`)
- `SEED` (default: `42`)
- `EPOCHS` (default: `150`)
- `BATCH_SIZE` (default: `256`)
- `LR` (default: `1e-3`)
- `PRETRAIN_EPOCHS` (default: `100`)
- `PRETRAIN_LR` (default: `1e-3`)
- `PRETRAIN_MASK_RATIO` (default: `0.4`)
- `TRAIN_PARQUET`, `TEST_PARQUET`, `PRETRAIN_PARQUET`

## Consolidação de resultados

Após as execuções:

```bash
python ablations/patchtst/results/summarize_results.py
```

Arquivos gerados:
- `ablations/patchtst/results/summary.csv`
- `ablations/patchtst/results/summary.md`
