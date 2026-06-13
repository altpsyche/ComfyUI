#requires -Version 5.1
<#
.SYNOPSIS
  Walk custom_nodes/ and provision each pack: `pip install -r requirements.txt`
  (if present) then run `install.py` (if present, the ComfyUI-Manager convention).
  Also installs ComfyScript editable (it uses pyproject.toml).

.DESCRIPTION
  Called by setup.bat phase [5/6].
  - Skips __pycache__, .git, hidden dirs
  - For each pack: installs requirements.txt, THEN runs install.py if present
    (e.g. comfyui_text_to_pose's install.py clones its t2p model library;
    Impact-Pack/Subpack fetch their extra deps). install.py runs from the pack dir.
  - Logs each pack's result (installed / no-reqs / install.py / failed)
  - Returns non-zero exit if any pack fails (setup.bat will warn but continue)
#>

$ErrorActionPreference = 'Continue'
$cnDir = Join-Path $PSScriptRoot '..\custom_nodes'
$cnDir = (Resolve-Path $cnDir).Path

$results = @()
$failed  = @()

Get-ChildItem $cnDir -Directory | Where-Object { $_.Name -notin '__pycache__','ComfyScript' } | ForEach-Object {
    $pack    = $_.Name
    $reqFile = Join-Path $_.FullName 'requirements.txt'
    $instPy  = Join-Path $_.FullName 'install.py'
    $hasReq  = Test-Path $reqFile
    $hasInst = Test-Path $instPy

    if (-not $hasReq -and -not $hasInst) {
        $results += [PSCustomObject]@{ Pack=$pack; Status='no-reqs' }
        return
    }

    if ($hasReq) {
        Write-Host "  -> $pack (requirements.txt) ..." -NoNewline
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

    # install.py: ComfyUI-Manager runs this on install; mirror it so a plain setup.bat
    # provisions packs that need more than pip (e.g. comfyui_text_to_pose clones its t2p lib).
    if ($hasInst) {
        Write-Host "  -> $pack (install.py) ..." -NoNewline
        Push-Location $_.FullName
        $out = & python install.py 2>&1
        $rc  = $LASTEXITCODE
        Pop-Location
        if ($rc -eq 0) {
            Write-Host " ok"
            $results += [PSCustomObject]@{ Pack="$pack (install.py)"; Status='install.py' }
        } else {
            Write-Host " FAILED"
            $results += [PSCustomObject]@{ Pack="$pack (install.py)"; Status='failed' }
            $failed  += [PSCustomObject]@{ Pack="$pack (install.py)"; Output=($out | Out-String) }
        }
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
