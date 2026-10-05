param([ValidateSet('data','models','baselines','analysis','report','test','all')] [string]$Stage='all')
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
(& 'D:/Total_Tools/miniforge3/Scripts/conda.exe' 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate universal
$env:PYTHONPATH=(Join-Path (Get-Location) 'src')
$env:PYTHONUNBUFFERED='1'
python -c "import sys,torch; assert torch.cuda.is_available(); print(sys.executable,torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {throw 'CUDA required'}
if ($Stage -eq 'test') {python -m pytest -q --junitxml=outputs/alpha_strategy/tests.xml}
else {python scripts/run_alpha_strategy.py --stage $Stage}
if ($LASTEXITCODE -ne 0) {throw "alpha $Stage failed"}
