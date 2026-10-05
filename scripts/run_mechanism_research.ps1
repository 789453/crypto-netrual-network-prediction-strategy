param([ValidateSet('cache','smoke','research','report','test')] [string]$Stage = 'research')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
# Scripts/activate is a POSIX script on this installation. Use the equivalent PowerShell hook.
(& 'D:/Total_Tools/miniforge3/Scripts/conda.exe' 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate universal
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTHONUNBUFFERED = '1'
python -c "import sys,torch; assert torch.cuda.is_available(), 'CUDA required'; print(sys.executable,torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA environment verification failed' }
switch ($Stage) {
    'cache' { python scripts/build_mechanism_cache.py }
    'smoke' { python scripts/run_mechanism_research.py --smoke }
    'research' { python scripts/run_mechanism_research.py }
    'report' { python scripts/build_mechanism_report.py }
    'test' { python -m pytest -q }
}
if ($LASTEXITCODE -ne 0) { throw "Stage $Stage failed with exit $LASTEXITCODE" }
