#requires -Version 5.1
<#
.SYNOPSIS
  Create the LoRA-training venv (kohya sd-scripts) and install Blackwell-ready deps.

.DESCRIPTION
  Optional setup phase (run via: setup.bat --with-trainer). Independent of ComfyUI's own venv:
  sd-scripts needs Python <=3.12 and a pinned torch, so this builds a SEPARATE venv at
  tools/lora_train/.venv via uv. The trainer CODE is the tools/sd-scripts submodule
  (initialized by setup.bat phase [2/6]); this only provisions its Python env.

  The RTX 5080 (Blackwell sm_120) needs CUDA 12.8+ torch; the default kohya torch won't run, so
  we pin torch cu128 (well-tested for sd-scripts on Blackwell). Idempotent; safe to re-run.

.PARAMETER TorchVersion
  torch version to pin (default 2.7.0 — sd-scripts-compatible, Blackwell-capable).
.PARAMETER CudaTag
  pytorch wheel CUDA tag (default cu128).
#>
param(
    [string]$TorchVersion = '2.7.0',
    [string]$CudaTag = 'cu128'
)
# Continue (not Stop): native CLIs (uv/pip/python) write progress + tracebacks to stderr, which
# under 'Stop' would throw NativeCommandError mid-run. We gate every step on $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'

$root  = Split-Path -Parent $PSScriptRoot          # ...\ComfyUI
$sdDir = Join-Path $root 'tools\sd-scripts'
$venv  = Join-Path $root 'tools\lora_train\.venv'
$py    = Join-Path $venv 'Scripts\python.exe'
$reqs  = Join-Path $sdDir 'requirements.txt'

# sd-scripts submodule present?
if (-not (Test-Path $reqs)) {
    Write-Error "tools/sd-scripts not initialized. Run: git submodule update --init tools/sd-scripts"
    exit 1
}

# uv required (system python is too new for the ML stack; uv pins 3.11 for sd-scripts)
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found. Install it (https://docs.astral.sh/uv/), then re-run. uv pins Python 3.11."
    exit 1
}

# 1. venv (idempotent)
if (-not (Test-Path $py)) {
    Write-Host "  -> creating trainer venv (Python 3.11) at tools/lora_train/.venv"
    & uv venv --python 3.11 $venv
    if ($LASTEXITCODE -ne 0) { Write-Error "uv venv failed"; exit 1 }
} else {
    Write-Host "  [+] trainer venv already exists"
}

# 2. torch (skip if already a CUDA build — mirrors install_torch.ps1)
$cuda = & $py -c "import torch; print(torch.version.cuda or 'cpu')" 2>$null
$cuda = "$cuda".Trim()
if ($LASTEXITCODE -eq 0 -and $cuda -and $cuda -ne 'cpu') {
    Write-Host "  [+] trainer torch already CUDA-enabled (cuda=$cuda) - skipping reinstall"
} else {
    Write-Host "  -> installing torch $TorchVersion + torchvision ($CudaTag) for Blackwell"
    & uv pip install --python $py "torch==$TorchVersion" torchvision --index-url "https://download.pytorch.org/whl/$CudaTag"
    if ($LASTEXITCODE -ne 0) { Write-Error "torch install failed"; exit 1 }
}

# 3. sd-scripts requirements + optimizer + WD14-tagger deps.
#    Run from sd-scripts dir so the `-e .` line in requirements.txt resolves to sd-scripts.
#    uv won't upgrade the already-satisfied torch (nothing here pins it), so our Blackwell
#    wheel survives diffusers[torch].
Write-Host "  -> installing sd-scripts requirements (from $sdDir)"
Push-Location $sdDir
try {
    & uv pip install --python $py -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Write-Warning "some sd-scripts requirements failed - see above" }
} finally {
    Pop-Location
}
# onnx/onnxruntime are commented out in requirements.txt but the WD14 tagger needs them (CPU is
# plenty for ~30-60 images and avoids Blackwell EP issues); prodigyopt is our optimizer.
Write-Host "  -> installing tagger + optimizer deps (onnxruntime, onnx, prodigyopt)"
& uv pip install --python $py onnxruntime onnx prodigyopt
if ($LASTEXITCODE -ne 0) { Write-Warning "tagger/optimizer deps reported issues" }

# 4. accelerate default config (non-interactive) + GPU visibility check
$acc = Join-Path $venv 'Scripts\accelerate.exe'
if (Test-Path $acc) { & $acc config default 2>$null }
$dev = & $py -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')" 2>&1
Write-Host "  [+] trainer venv torch sees: $dev"
Write-Host "  [+] trainer ready. Next: see tools/lora_train/README.md (generate data -> caption -> train)."
