#requires -Version 5.1
<#
.SYNOPSIS
  Install pytorch + torchvision into the active venv for the requested GPU stack.

.PARAMETER GpuMode
  nvidia       autodetect CUDA via nvidia-smi (default)
  amd-rdna3    Windows AMD RX 7000 (gfx110X)
  amd-rdna35   Windows AMD Strix halo / Ryzen AI Max+ (gfx1151)
  amd-rdna4    Windows AMD RX 9000 (gfx120X)
  intel-xpu    Intel Arc XPU
  cpu          CPU-only wheel

.DESCRIPTION
  Called by setup.bat phase [4/6]. Maps GPU mode to PyTorch index URL and
  --pre flag, then force-reinstalls torch+torchvision to overwrite any
  unwanted wheel left behind by `pip install -r requirements.txt`.
#>
param(
    [string]$GpuMode = 'nvidia'
)

$ErrorActionPreference = 'Stop'

function Resolve-NvidiaCudaMajor {
    try {
        $smi = & nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $smi) { return $null }
        $driverMajor = [int]($smi.Trim().Split('.')[0])
        if     ($driverMajor -ge 580) { return '130' }   # cu130 (CUDA 13.0)
        elseif ($driverMajor -ge 555) { return '128' }   # cu128 (CUDA 12.8)
        elseif ($driverMajor -ge 525) { return '121' }   # cu121 (CUDA 12.1)
        elseif ($driverMajor -ge 470) { return '118' }   # cu118 (CUDA 11.8)
        else                          { return $null  }
    } catch { return $null }
}

# Dispatch on GpuMode
$indexUrl = $null
$preFlag  = ''
$label    = $GpuMode

switch -Regex ($GpuMode) {
    '^nvidia$' {
        $cuda = Resolve-NvidiaCudaMajor
        if ($cuda) {
            $indexUrl = "https://download.pytorch.org/whl/cu$cuda"
            $label = "nvidia cu$cuda"
        } else {
            Write-Warning "nvidia-smi unavailable; falling back to CPU wheel."
            $indexUrl = 'https://download.pytorch.org/whl/cpu'
            $label = 'cpu (nvidia-smi missing)'
        }
    }
    '^amd-rdna3$'  { $indexUrl = 'https://rocm.nightlies.amd.com/v2/gfx110X-all/'; $preFlag = '--pre' }
    '^amd-rdna35$' { $indexUrl = 'https://rocm.nightlies.amd.com/v2/gfx1151/';     $preFlag = '--pre' }
    '^amd-rdna4$'  { $indexUrl = 'https://rocm.nightlies.amd.com/v2/gfx120X-all/'; $preFlag = '--pre' }
    '^intel-xpu$'  { $indexUrl = 'https://download.pytorch.org/whl/xpu' }
    '^cpu$'        { $indexUrl = 'https://download.pytorch.org/whl/cpu' }
    default {
        Write-Error "Unknown --gpu mode: $GpuMode. Valid: nvidia, amd-rdna3, amd-rdna35, amd-rdna4, intel-xpu, cpu"
        exit 1
    }
}

Write-Host "  -> torch install: $label  (index: $indexUrl)"

# Check if torch already on a non-CPU build for nvidia/amd/intel modes
if ($GpuMode -ne 'cpu') {
    try {
        $existingCuda = & python -c "import torch; print(torch.version.cuda or 'cpu')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $existingCuda) {
            $existingCuda = $existingCuda.Trim()
            # For nvidia, cuda version present = OK. For amd/intel, version reports None but
            # the wheel is hip/xpu-flavored - too hard to detect cleanly. Force-reinstall.
            if ($GpuMode -eq 'nvidia' -and $existingCuda -ne 'cpu') {
                Write-Host "  [+] torch already CUDA-enabled (cuda=$existingCuda) - skipping reinstall"
                exit 0
            }
        }
    } catch {}
}

# Force-reinstall the chosen wheel
$args = @('install', '--force-reinstall')
if ($preFlag) { $args += $preFlag }
$args += @('torch', 'torchvision', '--index-url', $indexUrl)

& pip @args
if ($LASTEXITCODE -ne 0) {
    Write-Warning "torch install failed - install manually with: pip $($args -join ' ')"
    exit 1
}
Write-Host "  [+] torch + torchvision installed for $label"
