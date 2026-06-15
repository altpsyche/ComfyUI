#requires -Version 5.1
<#
.SYNOPSIS
  Download the Qwen-Image-Edit-2511 model stack for the ComfyUI edit-hybrid dataset workflow.

.DESCRIPTION
  Idempotent + portable. Pulls the GGUF diffusion model (Q5 by default), the qwen2.5-vl-7b text
  encoder, the qwen_image VAE, and the Lightning 4-step + multiple-angles LoRAs into the right
  models/ subfolders via the Hugging Face CLI. Skips any file already present. The ComfyUI-GGUF
  custom node (city96) must be installed (setup adds it as a submodule).

  Tuned for a 16 GB GPU: Q5_K_M (~14 GB) fits with the text encoder offloaded to RAM; the Lightning
  LoRA (6 steps / cfg 1.0) keeps generation practical. Pass -Quant Q4_K_M for less VRAM (lower
  quality) or Q6_K if you have headroom.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install_qwen_edit.ps1
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install_qwen_edit.ps1 -Quant Q4_K_M
#>
[CmdletBinding()]
param(
    [ValidateSet('Q4_K_S','Q4_K_M','Q5_K_S','Q5_K_M','Q6_K')] [string]$Quant = 'Q5_K_M',
    [switch]$SkipAnglesLora     # skip the multiple-angles LoRA (still gets Lightning)
)
$ErrorActionPreference = 'Continue'

$repo   = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$models = Join-Path $repo 'models'
$hf     = (Get-Command hf -ErrorAction SilentlyContinue).Source
if (-not $hf) { $hf = (Get-Command huggingface-cli -ErrorAction SilentlyContinue).Source }
if (-not $hf) { Write-Host "[x] Hugging Face CLI not found (need 'hf' or 'huggingface-cli' on PATH)" -ForegroundColor Red; exit 1 }
if (-not (Test-Path (Join-Path $repo 'custom_nodes\ComfyUI-GGUF'))) {
    Write-Host "[!] ComfyUI-GGUF custom node missing -- run: git submodule update --init custom_nodes/ComfyUI-GGUF" -ForegroundColor Yellow
}

# repo, repo-relative source path, destination subfolder under models/, flat destination name
function Get-Model($repoId, $src, $destSub, $destName) {
    if (-not $destName) { $destName = Split-Path $src -Leaf }
    $destDir = Join-Path $models $destSub
    $dest    = Join-Path $destDir $destName
    if (Test-Path $dest) { Write-Host "[=] $destSub/$destName already present"; return }
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    $stage = Join-Path ([System.IO.Path]::GetTempPath()) ("qedl_" + [guid]::NewGuid().ToString('N'))
    Write-Host "[>] $repoId :: $src"
    & $hf download $repoId $src --local-dir $stage
    if ($LASTEXITCODE -ne 0) { Write-Host "[x] download failed: $repoId/$src" -ForegroundColor Red; return }
    Move-Item (Join-Path $stage $src) $dest -Force
    Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "[+] models/$destSub/$destName" -ForegroundColor Green
}

Write-Host "=== Qwen-Image-Edit-2511 model stack (quant $Quant) ===`n"

# Diffusion model (GGUF) -> models/unet/   (UnetLoaderGGUF reads from models/unet)
# NB: repo filenames keep the quant tag UPPER-case (qwen-image-edit-2511-Q5_K_M.gguf).
Get-Model 'unsloth/Qwen-Image-Edit-2511-GGUF' "qwen-image-edit-2511-$Quant.gguf" 'unet'

# Text encoder + VAE -> models/text_encoders, models/vae   (shared Qwen-Image components)
Get-Model 'Comfy-Org/Qwen-Image_ComfyUI' 'split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors' 'text_encoders' 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
Get-Model 'Comfy-Org/Qwen-Image_ComfyUI' 'split_files/vae/qwen_image_vae.safetensors' 'vae' 'qwen_image_vae.safetensors'

# LoRAs -> models/loras/   (Lightning = fast 6-step gen; angles = camera-angle variety)
Get-Model 'lightx2v/Qwen-Image-Edit-2511-Lightning' 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors' 'loras'
if (-not $SkipAnglesLora) {
    Get-Model 'fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA' 'qwen-image-edit-2511-multiple-angles-lora.safetensors' 'loras'
}

Write-Host "`n[+] done. In ComfyUI use: UnetLoaderGGUF (qwen-image-edit-2511-$Quant.gguf),"
Write-Host "    CLIPLoader (qwen_2.5_vl_7b_fp8_scaled, type qwen_image), VAELoader (qwen_image_vae),"
Write-Host "    LoraLoaderModelOnly (Lightning 4-step) -> KSampler 6 steps / cfg 1.0 / euler / simple."
