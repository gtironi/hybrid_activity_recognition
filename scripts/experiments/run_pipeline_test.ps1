# configs
$env:PYTHONPATH = "src"
$train_parquet  = "dataset/processed/AcTBeCalf/windowed_train.parquet"
$test_parquet   = "dataset/processed/AcTBeCalf/windowed_test.parquet"
$seed           = "2026"
$device         = "cuda"
$epochs          = "300"
$pretrain_epochs = "100"

Write-Host "=== Iniciando Pipeline de Experimentos Completa via PowerShell ===" -ForegroundColor Cyan
Write-Host "Horário de Início: $(Get-Date)" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------"

# ------------------------------------------------------------------------------
# CNN + LSTM
# ------------------------------------------------------------------------------
Write-Host "`n>>> [1/5] Executando bloco: cnn_lstm..." -ForegroundColor Green

# 1.1 Pre-training
python -m hybrid_activity_recognition.main --mode pretrain_ts2vec --model cnn_lstm --pretrain_parquet $train_parquet --output_dir "experiments/ts2vec_pretrain_cnn_lstm" --pretrain_epochs $pretrain_epochs --batch_size 512 --seed $seed --device $device

# 1.2 Supervised training
python -m hybrid_activity_recognition.main --mode supervised --model cnn_lstm --input_mode deep_only --labeled_parquet_train $train_parquet --labeled_parquet_test $test_parquet --output_dir "experiments/cnn_lstm_supervised_deep_only" --epochs $epochs --batch_size 512 --seed $seed --device $device
python -m hybrid_activity_recognition.main --mode supervised --model cnn_lstm --input_mode hybrid --labeled_parquet_train $train_parquet --labeled_parquet_test $test_parquet --output_dir "experiments/cnn_lstm_supervised_hybrid" --epochs $epochs --batch_size 512 --seed $seed --device $device


# ------------------------------------------------------------------------------
# ROBUST (CNN + LSTM)
# ------------------------------------------------------------------------------
Write-Host "`n>>> [2/5] Executando bloco: robust..." -ForegroundColor Green

# 2.1 Pre-training
python -m hybrid_activity_recognition.main --mode pretrain_ts2vec --model robust --pretrain_parquet $train_parquet --output_dir "experiments/ts2vec_pretrain_robust" --pretrain_epochs $pretrain_epochs --batch_size 512 --seed $seed --device $device

# 2.2 Supervised training
python -m hybrid_activity_recognition.main --mode supervised --model robust --input_mode deep_only --labeled_parquet_train $train_parquet --labeled_parquet_test $test_parquet --output_dir "experiments/robust_supervised_deep_only" --epochs $epochs --batch_size 512 --seed $seed --device $device
python -m hybrid_activity_recognition.main --mode supervised --model robust --input_mode hybrid --labeled_parquet_train $train_parquet --labeled_parquet_test $test_parquet --output_dir "experiments/robust_supervised_hybrid" --epochs $epochs --batch_size 512 --seed $seed --device $device


# ------------------------------------------------------------------------------
# PATCHTST (TRANSFORMER)
# ------------------------------------------------------------------------------
Write-Host "`n>>> [3/5] Executando bloco: patchtst..." -ForegroundColor Green

# 3.1 Pre-training MAE
python -m hybrid_activity_recognition.main --mode pretrain --pretrain_parquet $train_parquet --output_dir "experiments/patchtst_pretrain" --pretrain_epochs $pretrain_epochs --batch_size 128 --seed $seed --device $device

# 3.2 Supervised training
python -m hybrid_activity_recognition.main --mode supervised --model patchtst --input_mode deep_only --labeled_parquet_train $train_parquet --labeled_parquet_test $test_parquet --output_dir "experiments/patchtst_supervised_deep_only" --epochs $epochs --batch_size 128 --seed $seed --device $device
python -m hybrid_activity_recognition.main --mode supervised --model patchtst --input_mode hybrid --labeled_parquet_train $train_parquet --labeled_parquet_test $test_parquet --output_dir "experiments/patchtst_supervised_hybrid" --epochs $epochs --batch_size 128 --seed $seed --device $device


# ------------------------------------------------------------------------------
# BASELINE RANDOM FOREST
# ------------------------------------------------------------------------------
Write-Host "`n>>> [4/5] Executando bloco: tsfel_baseline (Random Forest)..." -ForegroundColor Green
python -m random_forest_baseline.tsfel_baseline --train $train_parquet --test $test_parquet --output_dir "experiments/tsfel_baseline_rf" --seed $seed

# ------------------------------------------------------------------------------
# BASELINE DEEP MLP
# ------------------------------------------------------------------------------
Write-Host "`n>>> [5/5] Executando bloco: tsfel_mlp..." -ForegroundColor Green
python -m hybrid_activity_recognition.main --mode supervised --model tsfel_mlp --input_mode tsfel_only --labeled_parquet_train $train_parquet --labeled_parquet_test $test_parquet --output_dir "experiments/tsfel_mlp_baseline" --epochs $epochs --batch_size 512 --seed $seed --device $device


Write-Host "`n--------------------------------------------------------"
Write-Host "=== Todos os Experimentos de Teste Concluídos com Sucesso! ===" -ForegroundColor Cyan
Write-Host "Horário de Término: $(Get-Date)" -ForegroundColor Yellow