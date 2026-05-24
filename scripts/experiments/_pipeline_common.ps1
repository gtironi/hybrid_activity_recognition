# Shared helpers for run_pipeline_raw_data.ps1 and run_pipeline_processed_data.ps1
# Dot-source from the experiment scripts:  . "$PSScriptRoot\_pipeline_common.ps1"

function Write-Step {
    param(
        [string]$Message,
        [string]$Color = "Cyan"
    )
    Write-Host ""
    Write-Host ">>> $Message" -ForegroundColor $Color
    Write-Host "    $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor DarkGray
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Comando falhou (exit=$LASTEXITCODE): python $($Arguments -join ' ')"
    }
}

function Get-SotaTrainFlags {
    param(
        [string]$Model
    )
    $flags = @(
        "--signal_norm", "subject",
        "--norm_shrinkage_tau", "50",
        "--fusion", "gated",
        "--loss_criterion", "focal",
        "--apply_augmentation",
        "--adversarial_subject_alignment",
        "--adversarial_beta", "0.1",
        "--val_fraction", "0.1"
    )
    if ($Model -eq "robust") {
        $flags += @("--cnn_dropout", "0.4", "--lstm_dropout", "0.4")
    }
    return $flags
}

function Get-PatchTstFlags {
    return @(
        "--context_length", "75",
        "--patch_len", "15",
        "--stride", "5",
        "--mask_ratio", "0.5"
    )
}

function Invoke-MlExperimentGrid {
    param(
        [string]$TrainParquet,
        [string]$TestParquet,
        [string]$PretrainRawParquet,
        [int]$Epochs,
        [int]$FinetuneEpochs,
        [int]$PretrainEpochs,
        [string]$Seed,
        [string]$Device,
        [int]$BatchLarge = 512,
        [int]$BatchPatch = 128
    )

    $patchFlags = Get-PatchTstFlags

    # --- TS2Vec (CNN-LSTM + Robust) ---
    foreach ($model in @("cnn_lstm", "robust")) {
        Write-Step "Pretrain TS2Vec: $model" "Green"
        $ts2vecOut = "experiments/ts2vec_pretrain_${model}_sota"
        Invoke-Python -Arguments @(
            "-m", "hybrid_activity_recognition.main",
            "--mode", "pretrain_ts2vec",
            "--model", $model,
            "--pretrain_parquet", $TrainParquet,
            "--output_dir", $ts2vecOut,
            "--pretrain_epochs", "$PretrainEpochs",
            "--batch_size", "$BatchLarge",
            "--pretrain_lr", "1e-3",
            "--seed", $Seed,
            "--device", $Device
        )
        $initCkpt = Join-Path $ts2vecOut "ts2vec_best.pt"

        foreach ($mode in @("deep_only", "hybrid")) {
            $out = "experiments/${model}_supervised_${mode}_sota"
            $trainFlags = Get-SotaTrainFlags -Model $model
            $extra = @()
            if (Test-Path $initCkpt) { $extra = @("--init_encoder_from", $initCkpt) }

            Write-Step "Supervised $model / $mode" "Green"
            # CORREÇÃO: Somando os arrays dentro de uma subexpressão $()
            $baseArgs = @(
                "-m", "hybrid_activity_recognition.main",
                "--mode", "supervised",
                "--model", $model,
                "--input_mode", $mode,
                "--labeled_parquet_train", $TrainParquet,
                "--labeled_parquet_test", $TestParquet,
                "--output_dir", $out,
                "--epochs", "$Epochs",
                "--batch_size", "$BatchLarge",
                "--lr", "1e-3",
                "--seed", $Seed,
                "--device", $Device
            )
            Invoke-Python -Arguments ($baseArgs + $trainFlags + $extra)

            Write-Step "Finetune $model / $mode" "Green"
            $fineArgs = @(
                "-m", "hybrid_activity_recognition.main",
                "--mode", "finetune",
                "--model", $model,
                "--input_mode", $mode,
                "--labeled_parquet_train", $TrainParquet,
                "--labeled_parquet_test", $TestParquet,
                "--output_dir", $out,
                "--epochs", "$FinetuneEpochs",
                "--batch_size", "$BatchLarge",
                "--lr", "1e-4",
                "--seed", $Seed,
                "--device", $Device,
                "--checkpoint", (Join-Path $out "best.pt"),
                "--signal_norm", "subject",
                "--norm_shrinkage_tau", "50",
                "--fusion", "gated",
                "--label_smoothing", "0.05",
                "--finetune_early_stopping_patience", "15"
            )
            $dropoutFlags = if ($model -eq "robust") { @("--cnn_dropout", "0.4", "--lstm_dropout", "0.4") } else { @() }
            Invoke-Python -Arguments ($fineArgs + $dropoutFlags)
        }
    }

    # --- PatchTST MAE (raw unlabeled) ---
    if (Test-Path $PretrainRawParquet) {
        Write-Step "Pretrain PatchTST MAE (raw)" "Green"
        $maeRawOut = "experiments/patchtst_pretrain_raw_sota"
        $maeArgs = @(
            "-m", "hybrid_activity_recognition.main",
            "--mode", "pretrain",
            "--pretrain_parquet", $PretrainRawParquet,
            "--output_dir", $maeRawOut,
            "--pretrain_epochs", "$PretrainEpochs",
            "--batch_size", "$BatchPatch",
            "--pretrain_lr", "1e-3",
            "--seed", $Seed,
            "--device", $Device
        )
        Invoke-Python -Arguments ($maeArgs + $patchFlags)
        $patchCkpt = Join-Path $maeRawOut "best.pt"
    } else {
        Write-Host "AVISO: $PretrainRawParquet nao encontrado; PatchTST supervised sem checkpoint MAE." -ForegroundColor Yellow
        $patchCkpt = ""
    }

    foreach ($mode in @("deep_only", "hybrid")) {
        $out = "experiments/patchtst_supervised_${mode}_sota"
        $trainFlags = Get-SotaTrainFlags -Model "patchtst"
        $extra = @()
        if ($patchCkpt -and (Test-Path $patchCkpt)) {
            $extra = @("--patchtst_checkpoint", $patchCkpt)
        }

        Write-Step "Supervised patchtst / $mode" "Green"
        $pSuperArgs = @(
            "-m", "hybrid_activity_recognition.main",
            "--mode", "supervised",
            "--model", "patchtst",
            "--input_mode", $mode,
            "--labeled_parquet_train", $TrainParquet,
            "--labeled_parquet_test", $TestParquet,
            "--output_dir", $out,
            "--epochs", "$Epochs",
            "--batch_size", "$BatchPatch",
            "--lr", "1e-3",
            "--seed", $Seed,
            "--device", $Device
        )
        Invoke-Python -Arguments ($pSuperArgs + $patchFlags + $trainFlags + $extra)

        Write-Step "Finetune patchtst / $mode" "Green"
        $pFinetuneArgs = @(
            "-m", "hybrid_activity_recognition.main",
            "--mode", "finetune",
            "--model", "patchtst",
            "--input_mode", $mode,
            "--labeled_parquet_train", $TrainParquet,
            "--labeled_parquet_test", $TestParquet,
            "--output_dir", $out,
            "--epochs", "$FinetuneEpochs",
            "--batch_size", "$BatchPatch",
            "--lr", "1e-4",
            "--seed", $Seed,
            "--device", $Device,
            "--checkpoint", (Join-Path $out "best.pt"),
            "--signal_norm", "subject",
            "--norm_shrinkage_tau", "50",
            "--fusion", "gated",
            "--label_smoothing", "0.05",
            "--finetune_early_stopping_patience", "15"
        )
        Invoke-Python -Arguments ($pFinetuneArgs + $patchFlags)
    }

    # --- Baselines ---
    Write-Step "TSFEL Random Forest baseline" "Green"
    Invoke-Python -Arguments @(
        "-m", "random_forest_baseline.tsfel_baseline",
        "--train", $TrainParquet,
        "--test", $TestParquet,
        "--output_dir", "experiments/tsfel_baseline_rf_sota",
        "--seed", $Seed
    )

    Write-Step "TSFEL MLP baseline" "Green"
    Invoke-Python -Arguments @(
        "-m", "hybrid_activity_recognition.main",
        "--mode", "supervised",
        "--model", "tsfel_mlp",
        "--input_mode", "tsfel_only",
        "--labeled_parquet_train", $TrainParquet,
        "--labeled_parquet_test", $TestParquet,
        "--output_dir", "experiments/tsfel_mlp_baseline_sota",
        "--epochs", "$Epochs",
        "--batch_size", "$BatchLarge",
        "--seed", $Seed,
        "--device", $Device,
        "--signal_norm", "subject",
        "--loss_criterion", "focal"
    )

    # --- Ensemble stacking (PatchTST hybrid + RF) ---
    $hybridOut = "experiments/patchtst_supervised_hybrid_sota"
    $ckptFinetuned = Join-Path $hybridOut "finetuned_best.pt"
    $ckptBest = Join-Path $hybridOut "best.pt"
    $ensembleCkpt = if (Test-Path $ckptFinetuned) { $ckptFinetuned } elseif (Test-Path $ckptBest) { $null } else { $null }

    if ($ensembleCkpt) {
        Write-Step "Ensemble stacking (PatchTST hybrid + TSFEL RF)" "Green"
        $ensArgs = @(
            "-m", "hybrid_activity_recognition.main",
            "--mode", "ensemble",
            "--model", "patchtst",
            "--input_mode", "hybrid",
            "--labeled_parquet_train", $TrainParquet,
            "--labeled_parquet_test", $TestParquet,
            "--output_dir", (Join-Path $hybridOut "ensemble_stacking"),
            "--checkpoint", $ensembleCkpt,
            "--ensemble_method", "stacking",
            "--ensemble_deep_weight", "0.5",
            "--seed", $Seed,
            "--device", $Device,
            "--signal_norm", "subject",
            "--norm_shrinkage_tau", "50",
            "--fusion", "gated"
        )
        Invoke-Python -Arguments ($ensArgs + $patchFlags)
    } else {
        Write-Host "AVISO: checkpoint PatchTST hybrid ausente; ensemble ignorado." -ForegroundColor Yellow
    }
}