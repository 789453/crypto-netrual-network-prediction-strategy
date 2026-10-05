param([ValidateSet('cache','smoke','screen','confirm','all','analysis','predict','report','test')] [string]$Stage='all')
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
(& 'D:/Total_Tools/miniforge3/Scripts/conda.exe' 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate universal
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
$env:PYTHONUNBUFFERED='1'
python -c "import sys,torch; assert torch.cuda.is_available(); print(sys.executable,torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {throw 'CUDA required'}
if ($Stage -eq 'test') {python -m pytest -q --junitxml=outputs/review_v5/tests.xml}
elseif ($Stage -eq 'analysis') {python scripts/analyze_review_research.py}
elseif ($Stage -eq 'predict') {python scripts/predict_review.py}
elseif ($Stage -eq 'report') {python scripts/build_review_report.py}
else {python scripts/run_review_research.py --stage $Stage}
if ($LASTEXITCODE -ne 0) {throw "v5 $Stage failed"}
