#requires -Version 5.1
<#
.SYNOPSIS
  Walk custom_nodes/ and `pip install -r requirements.txt` for each pack
  that has one. Also installs ComfyScript editable (it uses pyproject.toml).

.DESCRIPTION
  Called by setup.bat phase [5/6].
  - Skips __pycache__, .git, hidden dirs
  - Logs each pack's result (installed / no-reqs / failed)
  - Returns non-zero exit if any pack fails (setup.bat will warn but continue)
#>

$ErrorActionPreference = 'Continue'
$cnDir = Join-Path $PSScriptRoot '..\custom_nodes'
$cnDir = (Resolve-Path $cnDir).Path

$results = @()
$failed  = @()

Get-ChildItem $cnDir -Directory | Where-Object { $_.Name -notin '__pycache__','ComfyScript' } | ForEach-Object {
    $pack = $_.Name
    $reqFile = Join-Path $_.FullName 'requirements.txt'

    if (-not (Test-Path $reqFile)) {
        $results += [PSCustomObject]@{ Pack=$pack; Status='no-reqs' }
        return
    }

    Write-Host "  -> $pack ..." -NoNewline
    $out = & pip install -U -r $reqFile 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host " ok"
        $results += [PSCustomObject]@{ Pack=$pack; Status='installed' }
    } else {
        Write-Host " FAILED"
        $results += [PSCustomObject]@{ Pack=$pack; Status='failed' }
        $failed  += [PSCustomObject]@{ Pack=$pack; Output=($out | Out-String) }
    }
}

# ComfyScript: editable install with [default] extras (transpiler + nodes import)
$cs = Join-Path $cnDir 'ComfyScript'
if (Test-Path (Join-Path $cs 'pyproject.toml')) {
    Write-Host "  -> ComfyScript (editable, [default] extras) ..." -NoNewline
    & pip install -e "$cs[default]" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host " ok"
        $results += [PSCustomObject]@{ Pack='ComfyScript'; Status='installed (editable)' }
    } else {
        Write-Host " FAILED"
        $results += [PSCustomObject]@{ Pack='ComfyScript'; Status='failed' }
        $failed  += [PSCustomObject]@{ Pack='ComfyScript'; Output='(see pip output above)' }
    }
}

Write-Host ""
Write-Host "  Summary:"
$counts = $results | Group-Object Status | Select-Object Name, Count
$counts | ForEach-Object { Write-Host "    $($_.Name): $($_.Count)" }

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "  Failed packs (re-run pip manually):" -ForegroundColor Yellow
    $failed | ForEach-Object { Write-Host "    - $($_.Pack)" -ForegroundColor Yellow }
    exit 1
}

exit 0
