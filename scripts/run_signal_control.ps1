param([ValidateSet('research','test','report','all')][string]$Stage='all')
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
& 'D:/Total_Tools/miniforge3/shell/condabin/conda-hook.ps1'
conda activate universal
if ($LASTEXITCODE -ne 0) { throw 'conda universal activation failed' }
New-Item -ItemType Directory -Force -Path 'outputs/signal_control_2026-10-05' | Out-Null
if ($Stage -in @('test','all')) {
    python -m pytest tests/test_control_position.py tests/test_probability_position.py tests/test_alpha_strategy.py -q --junitxml=outputs/signal_control_2026-10-05/tests.xml
    if ($LASTEXITCODE -ne 0) { throw 'control/account tests failed' }
}
if ($Stage -in @('research','all')) {
    python scripts/run_signal_control.py
    if ($LASTEXITCODE -ne 0) { throw 'signal control research failed' }
    python scripts/run_signal_control_extension.py
    if ($LASTEXITCODE -ne 0) { throw 'core participation extension failed' }
    python scripts/run_signal_core_ablation.py
    if ($LASTEXITCODE -ne 0) { throw 'core-only ablation failed' }
    python scripts/analyze_signal_control.py
    if ($LASTEXITCODE -ne 0) { throw 'signal response/risk evidence failed' }
}
if ($Stage -in @('report','all')) {
    python scripts/build_signal_control_report.py
    if ($LASTEXITCODE -ne 0) { throw 'signal control report failed' }
    python scripts/audit_signal_control.py
    if ($LASTEXITCODE -ne 0) { throw 'signal control final audit failed' }
}
