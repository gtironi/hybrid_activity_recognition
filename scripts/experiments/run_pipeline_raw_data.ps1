# =============================================================================
# Full Pipeline (CSV -> Parquets -> ML)
# =============================================================================

$ErrorActionPreference = "Stop"

$Epochs          = 150
$FinetuneEpochs  = 30
$PretrainEpochs  = 10
$Seed            = "2026"
$Device          = "cuda"
$BatchLarge      = 512
$BatchPatch      = 256
$WindowLen       = 75
$WindowStride    = 37

$PSScriptRootEscaped = $PSScriptRoot
$RepoRoot        = (Resolve-Path (Join-Path $PSScriptRootEscaped "..\..")).Path
$CsvLabeled      = Join-Path $RepoRoot "dataset\AcTBeCalf.csv"
$CsvRaw          = Join-Path $RepoRoot "dataset\Time_Adj_Raw_Data.csv"
$ProcessedDir    = Join-Path $RepoRoot "dataset\processed\AcTBeCalf"
$TrainLong       = Join-Path $ProcessedDir "train.parquet"
$TestLong        = Join-Path $ProcessedDir "test.parquet"
$TrainWindowed   = Join-Path $ProcessedDir "windowed_train.parquet"
$TestWindowed    = Join-Path $ProcessedDir "windowed_test.parquet"
$FeatureManifest = Join-Path $ProcessedDir "tsfel_feature_manifest.json"
$PretrainRaw     = Join-Path $RepoRoot "dataset\processed\pretrain_raw_windowed.parquet"

Set-Location -LiteralPath $RepoRoot
$env:PYTHONPATH = "src"

$CommonScript = Join-Path $PSScriptRootEscaped "_pipeline_common.ps1"
. $CommonScript

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " Pipeline RAW -> Processed -> ML (SOTA flags)" -ForegroundColor Cyan
Write-Host " Repo: $RepoRoot" -ForegroundColor Cyan
Write-Host " Epochs: supervised=$Epochs finetune=$FinetuneEpochs pretrain=$PretrainEpochs" -ForegroundColor Cyan
Write-Host " Start: $(Get-Date)" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan

Write-Step "dataset_processing.py (CSV -> train/test parquet)" "Magenta"
if ((Test-Path -LiteralPath $TrainLong) -And (Test-Path -LiteralPath $TestLong)) {
    Write-Host ">>> [CACHE] train.parquet and test.parquet already exist. No split." -ForegroundColor Yellow
} else {
    if (-not (Test-Path -LiteralPath $CsvLabeled)) { throw "CSV not found: $CsvLabeled" }
    $args = @("scripts/dataset_processing.py", "--csv", $CsvLabeled, "--out-dir", (Join-Path $RepoRoot "dataset\processed"), "--split-by", "behavior", "--subject-column", "calfId", "--behavior-column", "behaviour")
    python @args
}

Write-Step "window_raw_for_pretrain.py (Time_Adj_Raw_Data.csv)" "Magenta"
if (Test-Path -LiteralPath $PretrainRaw) {
    Write-Host ">>> [CACHE] pretrain_raw_windowed.parquet already exist. No window making." -ForegroundColor Yellow
} else {
    if (-not (Test-Path -LiteralPath $CsvRaw)) { throw "CSV not found: $CsvRaw" }
    $pretrainDir = Split-Path $PretrainRaw -Parent
    if (-not (Test-Path -LiteralPath $pretrainDir)) { New-Item -ItemType Directory -Path $pretrainDir -Force | Out-Null }
    $args = @("scripts/window_raw_for_pretrain.py", "--input", $CsvRaw, "--output", $PretrainRaw, "--window_len", "$WindowLen", "--stride", "$WindowStride")
    python @args
}

Write-Step "prepare_windowed_parquet.py (train / discover)" "Magenta"
if ((Test-Path -LiteralPath $TrainWindowed) -And (Test-Path -LiteralPath $FeatureManifest)) {
    Write-Host ">>> [CACHE] windowed_train.parquet and manifesto TSFEL already exist. No extraction." -ForegroundColor Yellow
} else {
    if (-not (Test-Path -LiteralPath $TrainLong)) { throw "No train: $TrainLong" }
    $args = @("scripts/prepare_windowed_parquet.py", "--input", $TrainLong, "--output", $TrainWindowed, "--feature-manifest-out", $FeatureManifest, "--window-size", "$WindowLen", "--overlap", "0.5", "--top-n", "75")
    python @args
}

Write-Step "prepare_windowed_parquet.py (test / apply manifest)" "Magenta"
if (Test-Path -LiteralPath $TestWindowed) {
    Write-Host ">>> [CACHE] windowed_test.parquet already exist." -ForegroundColor Yellow
} else {
    if (-not (Test-Path -LiteralPath $TestLong)) { throw "Saida esperada ausente: $TestLong" }
    if (-not (Test-Path -LiteralPath $FeatureManifest)) { throw "Manifest ausente: $FeatureManifest" }
    $args = @("scripts/prepare_windowed_parquet.py", "--input", $TestLong, "--output", $TestWindowed, "--feature-manifest-in", $FeatureManifest, "--window-size", "$WindowLen", "--overlap", "0.5")
    python @args
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
Write-Host " Pipeline RAW finished." -ForegroundColor Green
Write-Host " End: $(Get-Date)" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan