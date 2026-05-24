#requires -Version 5.1
<#
.SYNOPSIS
  Install pytorch + torchvision into the active venv, matching detected CUDA.

.DESCRIPTION
  Called by setup.bat phase [4/6].
  Detects CUDA runtime via `nvidia-smi`, picks matching pytorch wheel index URL.
  Falls back to CPU wheels if no NVIDIA GPU.

  Versions are pinned in $TORCH_VERSION / $TORCHVISION_VERSION so onboarding
  is reproducible. Bump these together when migrating.
#>

$ErrorActionPreference = 'Stop'

# Pinned versions — bump together
$TORCH_VERSION       = '2.8.0'
$TORCHVISION_VERSION = '0.23.0'

# Detect CUDA major version
$cudaMajor = $null
try {
    $smi = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $smi) {
        # Map driver version → CUDA index URL bucket
        # Drivers >= 525 support CUDA 12.x; >= 470 support 11.8
        $driverMajor = [int]($smi.Trim().Split('.')[0])
        if     ($driverMajor -ge 555) { $cudaMajor = '128' }   # cu128 (CUDA 12.8)
        elseif ($driverMajor -ge 525) { $cudaMajor = '121' }   # cu121 (CUDA 12.1)
        elseif ($driverMajor -ge 470) { $cudaMajor = '118' }   # cu118 (CUDA 11.8)
        else                          { $cudaMajor = $null  }  # too old, CPU
    }
} catch {
    $cudaMajor = $null
}

if ($cudaMajor) {
    $indexUrl = "https://download.pytorch.org/whl/cu$cudaMajor"
    Write-Host "  -> Installing torch $TORCH_VERSION (cu$cudaMajor) from $indexUrl"
} else {
    $indexUrl = "https://download.pytorch.org/whl/cpu"
    Write-Host "  -> No NVIDIA GPU detected, installing CPU wheels from $indexUrl"
}

# Check if torch already at pinned version
try {
    $existing = & python -c "import torch; print(torch.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $existing) {
        $existing = $existing.Split('+')[0].Trim()
        if ($existing -eq $TORCH_VERSION) {
            Write-Host "  [+] torch $TORCH_VERSION already installed"
            exit 0
        }
        Write-Host "  -> Found torch $existing, upgrading to $TORCH_VERSION"
    }
} catch {}

& pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url $indexUrl
if ($LASTEXITCODE -ne 0) {
    Write-Warning "torch install failed - you may need to install manually"
    exit 1
}
Write-Host "  [+] torch + torchvision installed"
