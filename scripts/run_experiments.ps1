$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$python = 'D:\Total_Tools\miniforge3\envs\universal\python.exe'
$runs = @(
    @{ Name = 'asym4h_seed20261003'; Mode = 'asym4h'; Seed = 20261003; Blocks = 3 },
    @{ Name = 'logratio4h_seed20261003'; Mode = 'logratio4h'; Seed = 20261003; Blocks = 3 },
    @{ Name = 'return4h_seed20261003'; Mode = 'return4h'; Seed = 20261003; Blocks = 3 },
    @{ Name = 'asym1h_seed20261003'; Mode = 'asym1h'; Seed = 20261003; Blocks = 3 },
    @{ Name = 'joint4h_seed20261011'; Mode = 'joint4h'; Seed = 20261011; Blocks = 3 },
    @{ Name = 'joint4h_seed20261023'; Mode = 'joint4h'; Seed = 20261023; Blocks = 3 },
    @{ Name = 'joint4h_depth2_seed20261003'; Mode = 'joint4h'; Seed = 20261003; Blocks = 2 },
    @{ Name = 'joint4h_no_market_seed20261003'; Mode = 'joint4h'; Seed = 20261003; Blocks = 3; Flag = '--no-market' },
    @{ Name = 'joint4h_no_stats_seed20261003'; Mode = 'joint4h'; Seed = 20261003; Blocks = 3; Flag = '--no-stats' }
)
foreach ($run in $runs) {
    $output = Join-Path 'outputs\models' $run.Name
    if (Test-Path (Join-Path $output 'summary.json')) {
        Write-Output "SKIP $($run.Name): completed"
        continue
    }
    Write-Output "START $($run.Name)"
    $extraArgs = @()
    if ($run.ContainsKey('Flag')) { $extraArgs += $run.Flag }
    & $python -u scripts\train_network.py --mode $run.Mode --seed $run.Seed `
        --fast-blocks $run.Blocks @extraArgs --output $output --max-epochs 6 `
        --train-stride-hours 2 --batch-size 256 2>&1 |
        Tee-Object -FilePath (Join-Path 'outputs' ($run.Name + '.console.log'))
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $($run.Name)" }
}
