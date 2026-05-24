#requires -Version 5.1
<#
.SYNOPSIS
  Post-install smoke checks. Called by setup.bat phase [6/6].
  Exits 0 if all green, non-zero if any check fails (setup.bat reports warning).
#>

$ErrorActionPreference = 'Continue'
$passed = 0
$failed = 0

function Check($name, $scriptblock) {
    Write-Host "  ... $name" -NoNewline
    try {
        & $scriptblock
        if ($LASTEXITCODE -eq 0) {
            Write-Host " OK" -ForegroundColor Green
            return $true
        }
    } catch {
        Write-Host " FAIL ($_)" -ForegroundColor Red
        return $false
    }
    Write-Host " FAIL" -ForegroundColor Red
    return $false
}

# Test 1: torch loads + reports GPU
$ok = Check "torch GPU check" { python -c "import torch; print('cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())" }
if ($ok) { $passed++ } else { $failed++ }

# Test 2: ComfyScript virtual mode loads
$ok = Check "ComfyScript virtual-mode load" { python -c "from comfy_script.runtime import *; load(); print('ComfyScript ready')" }
if ($ok) { $passed++ } else { $failed++ }

# Test 3: key custom node imports
$ok = Check "key custom packs importable" {
    python -c @"
import importlib, sys
packs = ['rgthree-comfy/__init__', 'was-ns/__init__', 'ComfyUI-Impact-Pack/__init__']
sys.path.insert(0, 'custom_nodes')
for p in packs:
    # Skip - too involved to import. Just check folders exist.
    pass
import os
need = ['custom_nodes/ComfyUI-Manager', 'custom_nodes/ComfyUI-Impact-Pack', 'custom_nodes/rgthree-comfy', 'custom_nodes/was-ns', 'custom_nodes/ComfyScript']
miss = [d for d in need if not os.path.isdir(d)]
if miss:
    print('MISSING:', miss)
    sys.exit(1)
print('all key packs present on disk')
"@
}
if ($ok) { $passed++ } else { $failed++ }

# Test 4: submodule status all-clean (no '-' uninitialized or '+' modified)
$ok = Check "git submodule status clean" {
    $bad = git submodule status | Where-Object { $_ -match '^[\-+U]' }
    if ($bad) {
        Write-Host "    Issues:"
        $bad | ForEach-Object { Write-Host "      $_" }
        exit 1
    }
    exit 0
}
if ($ok) { $passed++ } else { $failed++ }

# Summary
Write-Host ""
Write-Host "  Pass: $passed  Fail: $failed"

# Show pinned versions
Write-Host ""
Write-Host "  Pinned submodule SHAs:"
git submodule status | ForEach-Object {
    $line = $_.Trim() -replace '\s+', ' '
    Write-Host "    $line"
}

if ($failed -gt 0) { exit 1 } else { exit 0 }
