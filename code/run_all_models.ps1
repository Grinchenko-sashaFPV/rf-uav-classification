# Послiдовно тренує 4 моделi на 37 класах RFUAV.
# Усi результати зберiгаються у models/ та results/ з тегами.
# Запуск з C:\thesis:  .\code\run_all_models.ps1

$env:ALL_CLASSES = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$models = @(
    "resnet50",
    "mobilenetv3_small_100",
    "efficientnet_b0",
    "vit_small_patch16_224"
)

$logDir = "results\training_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$startTotal = Get-Date
Write-Host "=== START BATCH ===" -ForegroundColor Cyan
Write-Host "Start time: $startTotal" -ForegroundColor Cyan
Write-Host ""

foreach ($model in $models) {
    $env:MODEL_NAME = $model
    $logFile = "$logDir\${model}_log.txt"

    $start = Get-Date
    Write-Host "=== [$start] Training: $model ===" -ForegroundColor Yellow

    python code\train.py 2>&1 | Tee-Object -FilePath $logFile

    $end = Get-Date
    $duration = $end - $start
    Write-Host "=== [$end] Done $model in $($duration.TotalMinutes.ToString('F1')) min ===" -ForegroundColor Green
    Write-Host ""

    Start-Sleep -Seconds 5
}

Write-Host "=== EVALUATION PHASE ===" -ForegroundColor Cyan
foreach ($model in $models) {
    $env:MODEL_NAME = $model
    Write-Host "Evaluating $model..." -ForegroundColor Yellow
    python code\evaluate.py 2>&1 | Tee-Object -FilePath "$logDir\${model}_eval.txt"
    python code\evaluate_snr.py 2>&1 | Tee-Object -FilePath "$logDir\${model}_snr.txt"
}

$endTotal = Get-Date
$totalDuration = $endTotal - $startTotal
Write-Host ""
Write-Host "=== BATCH COMPLETE ===" -ForegroundColor Cyan
Write-Host "Total time: $($totalDuration.TotalMinutes.ToString('F1')) min" -ForegroundColor Cyan