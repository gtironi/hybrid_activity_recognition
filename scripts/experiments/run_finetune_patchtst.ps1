# EXTENSION PIPELINE - 3rd Phase PatchTST
$ErrorActionPreference = "Stop"

# Hyperparameters
$FinetuneEpochs  = 100 
$Patience        = 50
$Seed            = "2026"
$Device          = "cuda"
$BatchLarge      = 512
$BatchPatch      = 256

# Paths
$PSScriptRootEscaped = $PSScriptRoot
$RepoRoot        = (Resolve-Path (Join-Path $PSScriptRootEscaped "..\..")).Path
$ProcessedDir    = Join-Path $RepoRoot "dataset\processed\AcTBeCalf"
$TrainWindowed   = Join-Path $ProcessedDir "windowed_train.parquet"
$TestWindowed    = Join-Path $ProcessedDir "windowed_test.parquet"

Set-Location -LiteralPath $RepoRoot
$env:PYTHONPATH = "src"

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " Finetune" -ForegroundColor Cyan
Write-Host " Current Patience: $Patience epochs" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan

$robustDir = "experiments/robust_supervised_hybrid_sota"
$robustCkpt = Join-Path $RepoRoot (Join-Path $robustDir "best.pt")

if (Test-Path -LiteralPath $robustCkpt) {
    Write-Host "`n>>> [EXECUTANDO] Retreinando Fase 3 para Robust Hybrid..." -ForegroundColor Green
    $robustArgs = @(
        "-m", "hybrid_activity_recognition.main",
        "--mode", "finetune",
        "--model", "robust",
        "--input_mode", "hybrid",
        "--labeled_parquet_train", $TrainWindowed,
        "--labeled_parquet_test", $TestWindowed,
        "--output_dir", $robustDir,
        "--epochs", "$FinetuneEpochs",
        "--batch_size", "$BatchLarge",
        "--lr", "1e-3",
        "--seed", $Seed,
        "--device", $Device,
        "--checkpoint", $robustCkpt,
        "--signal_norm", "subject",
        "--norm_shrinkage_tau", "50",
        "--fusion", "gated",
        "--label_smoothing", "0.05",
        "--finetune_early_stopping_patience", "$Patience",
        "--cnn_dropout", "0.4",
        "--lstm_dropout", "0.4"
    )
    python @robustArgs
} else {
    Write-Host "AVISO: Checkpoint de origem não encontrado em $robustCkpt" -ForegroundColor Red
}

$patchDir = "experiments/patchtst_supervised_hybrid_sota"
$patchCkpt = Join-Path $RepoRoot (Join-Path $patchDir "best.pt")
$patchFlags = @("--context_length", "75", "--patch_len", "15", "--stride", "5")

if (Test-Path -LiteralPath $patchCkpt) {
    Write-Host "`n>>> [EXECUTANDO] Retreinando Fase 3 para PatchTST Hybrid..." -ForegroundColor Green
    $patchArgs = @(
        "-m", "hybrid_activity_recognition.main",
        "--mode", "finetune",
        "--model", "patchtst",
        "--input_mode", "hybrid",
        "--labeled_parquet_train", $TrainWindowed,
        "--labeled_parquet_test", $TestWindowed,
        "--output_dir", $patchDir,
        "--epochs", "$FinetuneEpochs",
        "--batch_size", "$BatchPatch",
        "--lr", "1e-4",
        "--seed", $Seed,
        "--device", $Device,
        "--checkpoint", $patchCkpt,
        "--signal_norm", "subject",
        "--norm_shrinkage_tau", "50",
        "--fusion", "gated",
        "--label_smoothing", "0.05",
        "--finetune_early_stopping_patience", "$Patience"
    )
    $allPatchArgs = $patchArgs + $patchFlags
    python @allPatchArgs
} else {
    Write-Host "AVISO: Checkpoint de origem não encontrado em $patchCkpt" -ForegroundColor Red
}

$newFinetunedPatchCkpt = Join-Path $RepoRoot (Join-Path $patchDir "finetuned_best.pt")
if (Test-Path -LiteralPath $newFinetunedPatchCkpt) {
    Write-Host "`n>>> [EXECUTANDO] Recalculando Stacking Meta-Learner com a nova Fase 3..." -ForegroundColor Green
    $ensArgs = @(
        "-m", "hybrid_activity_recognition.main",
        "--mode", "ensemble",
        "--model", "patchtst",
        "--input_mode", "hybrid",
        "--labeled_parquet_train", $TrainWindowed,
        "--labeled_parquet_test", $TestWindowed,
        "--output_dir", (Join-Path $patchDir "ensemble_stacking"),
        "--checkpoint", $newFinetunedPatchCkpt,
        "--ensemble_method", "stacking",
        "--ensemble_deep_weight", "0.5",
        "--seed", $Seed,
        "--device", $Device,
        "--signal_norm", "subject",
        "--norm_shrinkage_tau", "50",
        "--fusion", "gated"
    )
    $allEnsArgs = $ensArgs + $patchFlags
    python @allEnsArgs
}

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host " Finished. Results are above" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan