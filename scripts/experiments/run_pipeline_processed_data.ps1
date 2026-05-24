$ErrorActionPreference = "Stop"

$Epochs          = 2
$FinetuneEpochs  = 2
$PretrainEpochs  = 2
$Seed            = "2026"
$Device          = "cuda"
$BatchLarge      = 512
$BatchPatch      = 128

$PSScriptRootEscaped = $PSScriptRoot
$RepoRoot        = (Resolve-Path (Join-Path $PSScriptRootEscaped "..\..")).Path
$ProcessedDir    = Join-Path $RepoRoot "dataset\processed\AcTBeCalf"
$TrainWindowed   = Join-Path $ProcessedDir "windowed_train.parquet"
$TestWindowed    = Join-Path $ProcessedDir "windowed_test.parquet"
$PretrainRaw     = Join-Path $RepoRoot "dataset\processed\pretrain_raw_windowed.parquet"

Set-Location -LiteralPath $RepoRoot
$env:PYTHONPATH = "src"

$CommonScript = Join-Path $PSScriptRootEscaped "_pipeline_common.ps1"
. $CommonScript

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " Pipeline PROCESSED -> ML (SOTA flags, on-the-fly norm/aug)" -ForegroundColor Cyan
Write-Host " Repo: $RepoRoot" -ForegroundColor Cyan
Write-Host " Epochs: supervised=$Epochs finetune=$FinetuneEpochs pretrain=$PretrainEpochs" -ForegroundColor Cyan
Write-Host " Start: $(Get-Date)" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan

if (-not (Test-Path -LiteralPath $TrainWindowed)) {
    throw "Train Parquet not found: $TrainWindowed . Run run_pipeline_raw_data.ps1 or get the parquets on the dataset folder"
}
if (-not (Test-Path -LiteralPath $TestWindowed)) {
    throw "Test Parquet not found: $TestWindowed"
}

Write-Step "Getting Parquets" "DarkYellow"
Write-Host "  Train: $TrainWindowed" -ForegroundColor DarkGray
Write-Host "  Test:  $TestWindowed" -ForegroundColor DarkGray
if (Test-Path -LiteralPath $PretrainRaw) {
    Write-Host "  MAE raw: $PretrainRaw" -ForegroundColor DarkGray
} else {
    Write-Host "  MAE raw: (opcional) $PretrainRaw not found - PatchTST with no MAE in raw" -ForegroundColor Yellow
}

Write-Step "Grade ML (pretrain + supervised + finetune + baselines + ensemble)" "Magenta"
Invoke-MlExperimentGrid `
    -TrainParquet $TrainWindowed `
    -TestParquet $TestWindowed `
    -PretrainRawParquet $PretrainRaw `
    -Epochs $Epochs `
    -FinetuneEpochs $FinetuneEpochs `
    -PretrainEpochs $PretrainEpochs `
    -Seed $Seed `
    -Device $Device `
    -BatchLarge $BatchLarge `
    -BatchPatch $BatchPatch

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " Pipeline PROCESSED" -ForegroundColor Green
Write-Host " End: $(Get-Date)" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan