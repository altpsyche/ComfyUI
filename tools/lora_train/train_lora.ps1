#requires -Version 5.1
<#
.SYNOPSIS
  Train ONE character LoRA with kohya sd-scripts. Portable + parameterized + auto-captioning.

.DESCRIPTION
  Convention: images live in <repo>/output/dataset/<Char>/ (IL_Dataset saves there directly).
  This script validates the folder, auto-captions it (WD14 tagger + prep_captions) if no .txt
  captions exist, derives num_repeats from a target step count, writes the dataset .toml, and
  launches training. No per-character file copies, no hardcoded paths, no manual file moving.

.EXAMPLE
  .\train_lora.ps1 -Char aria -Prune "auburn hair,long hair,green eyes,freckles"
.EXAMPLE
  .\train_lora.ps1 -Char kael -Trigger kaelchar -Steps 2000 -TrainTextEncoder
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$Char,   # dataset folder name + default LoRA name
    [string]$Trigger,                        # caption trigger token (default: <Char>char)
    [string]$Prune = "",                     # exact tags to bake into the trigger (comma list)
    [string]$Base,                           # checkpoint (default: oneObsession in models/checkpoints)
    [int]$Dim = 16,
    [int]$Alpha = 8,
    [ValidateSet('prodigy','adamw','adafactor')] [string]$Optimizer = 'prodigy',  # Blackwell-safe set; NOT adamw8bit (bitsandbytes on sm_120 is unverified)
    [double]$DCoef = 1.0,                     # Prodigy d_coef (try 0.8 to reduce overcook on small sets)
    [int]$Steps = 1500,                      # TARGET total steps; num_repeats is derived from it
    [int]$Epochs = 10,
    [int]$Batch = 2,
    [int]$MinImages = 12,                    # refuse to train on too small a set
    [switch]$TrainTextEncoder,               # also train the TE (drops --network_train_unet_only)
    [switch]$SkipCaption                     # don't auto-caption (captions already prepared)
)
$ErrorActionPreference = 'Continue'          # native CLIs write to stderr; gate on $LASTEXITCODE
$env:PYTHONUTF8 = '1'                        # sd-scripts prints unicode; cp1252 console crashes

# --- portable paths (derive everything from this script's location) ---
$repo  = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$py    = Join-Path $repo 'tools\lora_train\.venv\Scripts\python.exe'
$acc   = Join-Path $repo 'tools\lora_train\.venv\Scripts\accelerate.exe'
$sdDir = Join-Path $repo 'tools\sd-scripts'
$data  = Join-Path $repo "output\dataset\$Char"
$outDir = Join-Path $repo 'models\loras'
# roster defaults (trigger/prune) from the build-generated manifest, unless overridden on the CLI
$rosterFile = Join-Path $repo 'tools\lora_train\roster.json'
if (Test-Path $rosterFile) {
    # NB: WinPS 5.1 ConvertFrom-Json emits the array as one non-enumerated object — iterate with foreach.
    $roster = Get-Content $rosterFile -Raw | ConvertFrom-Json
    foreach ($e in $roster) {
        if ($e.name -eq $Char) {
            if (-not $Trigger) { $Trigger = [string]$e.trigger }
            if (-not $PSBoundParameters.ContainsKey('Prune') -and $e.prune) { $Prune = [string]$e.prune }
            break
        }
    }
}
if (-not $Trigger) { $Trigger = "${Char}char" }
if (-not $Base)    { $Base = Join-Path $repo 'models\checkpoints\oneObsession_v19Atypical.safetensors' }

function Die($msg) { Write-Host "[x] $msg" -ForegroundColor Red; exit 1 }

# --- pre-flight ---
if (-not (Test-Path $py))   { Die "trainer venv missing. Run: setup.bat --with-trainer" }
if (-not (Test-Path $Base)) { Die "checkpoint not found: $Base  (pass -Base <path>)" }
if (-not (Test-Path $data)) { Die "no dataset at $data  -- generate with IL_Dataset (SaveImage prefix 'dataset/$Char')" }
$imgs = @(Get-ChildItem $data -File | Where-Object { $_.Extension -in '.png','.jpg','.jpeg','.webp' })
if ($imgs.Count -lt $MinImages) { Die "only $($imgs.Count) images in $data (need >= $MinImages). Generate/curate more." }
Write-Host "[+] $($imgs.Count) images in output/dataset/$Char"

# --- caption (only if not already done) ---
$haveCaptions = @(Get-ChildItem $data -Filter *.txt -File).Count -gt 0
if ($SkipCaption) {
    Write-Host "[-] -SkipCaption: using existing captions"
} elseif ($haveCaptions) {
    Write-Host "[+] captions already present - skipping tagger (delete .txt to re-tag)"
} else {
    Write-Host "[+] auto-captioning (WD14 tagger, trigger '$Trigger')"
    Push-Location $sdDir
    & $py 'finetune/tag_images_by_wd14_tagger.py' --onnx `
        --repo_id 'SmilingWolf/wd-v1-4-convnextv2-tagger-v2' --batch_size 4 $data
    $rc = $LASTEXITCODE
    Pop-Location
    if ($rc -ne 0) { Die "WD14 tagger failed (rc=$rc)" }
    & $py (Join-Path $PSScriptRoot 'prep_captions.py') $data --trigger $Trigger --prune $Prune
    if ($LASTEXITCODE -ne 0) { Die "prep_captions failed" }
}

# --- derive num_repeats from the target step count ---
#   steps ~= images * repeats * epochs / batch  ->  repeats = round(steps * batch / (images * epochs))
$repeats = [int][math]::Max(1, [math]::Round($Steps * $Batch / ($imgs.Count * $Epochs)))
$actual  = [int]($imgs.Count * $repeats * $Epochs / $Batch)
Write-Host "[+] $($imgs.Count) imgs x $repeats repeats x $Epochs epochs / batch $Batch  ~= $actual steps (target $Steps)"

# --- write the dataset config (generated; not a hand-copied per-char file) ---
$cacheDir = Join-Path $PSScriptRoot '.cache'
New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
$cfgPath  = Join-Path $cacheDir "$Char.toml"
$dataFwd  = $data -replace '\\','/'
@"
# AUTO-GENERATED by train_lora.ps1 for character '$Char' -- do not hand-edit.
[general]
shuffle_caption = true
keep_tokens = 1
caption_extension = ".txt"

[[datasets]]
resolution = 1024
batch_size = $Batch
enable_bucket = true
min_bucket_reso = 768
max_bucket_reso = 1280
bucket_reso_steps = 64

  [[datasets.subsets]]
  image_dir = "$dataFwd"
  num_repeats = $repeats
"@ | Set-Content -Path $cfgPath -Encoding UTF8

# --- optimizer block (Prodigy is the default; AdamW / AdaFactor are Blackwell-safe alternatives.
# NOT AdamW8bit: bitsandbytes on sm_120 is unverified -- the reason Prodigy was chosen. clip_skip is
# deliberately NOT passed: sdxl_train_network ignores it for SDXL.) ---
switch ($Optimizer) {
    'adamw' {
        $optArgs = @('--optimizer_type','AdamW',
                     '--learning_rate','3e-4','--unet_lr','3e-4','--text_encoder_lr','3e-5')
    }
    'adafactor' {
        $optArgs = @('--optimizer_type','Adafactor',
                     '--optimizer_args','relative_step=False','scale_parameter=False','warmup_init=False',
                     '--learning_rate','3e-4','--unet_lr','3e-4','--text_encoder_lr','3e-5')
    }
    default {   # prodigy: auto-tunes LR (lr must be 1.0); d_coef knobs effective LR
        $optArgs = @('--optimizer_type','prodigy',
                     '--learning_rate','1.0','--unet_lr','1.0','--text_encoder_lr','1.0',
                     '--optimizer_args','decouple=True','weight_decay=0.01',"d_coef=$DCoef",'use_bias_correction=True','safeguard_warmup=True')
    }
}

# --- train ---
$trainArgs = @(
    'launch','--num_cpu_threads_per_process','8','sdxl_train_network.py',
    '--pretrained_model_name_or_path',$Base,
    '--dataset_config',$cfgPath,
    '--output_dir',$outDir,'--output_name',"${Char}_v1",
    '--network_module','networks.lora','--network_dim',$Dim,'--network_alpha',$Alpha
)
$trainArgs += $optArgs
$trainArgs += @(
    '--lr_scheduler','cosine',
    '--max_train_epochs',$Epochs,'--save_every_n_epochs','1',
    '--train_batch_size',$Batch,'--gradient_checkpointing',
    '--mixed_precision','bf16','--save_precision','bf16',
    '--cache_latents','--cache_latents_to_disk',
    '--sdpa','--no_half_vae',
    '--min_snr_gamma','5','--seed','42'
)
if (-not $TrainTextEncoder) { $trainArgs += '--network_train_unet_only' }   # 16 GB-safe default

$optDesc = if ($Optimizer -eq 'prodigy') { "prodigy d_coef=$DCoef" } else { $Optimizer }
Write-Host "[+] training ${Char}_v1 (dim $Dim / alpha $Alpha, $optDesc) -> models/loras/${Char}_v1.safetensors"
Push-Location $sdDir                         # sd-scripts imports library.* relative to its dir
& $acc @trainArgs
$rc = $LASTEXITCODE
Pop-Location
if ($rc -ne 0) { Die "training failed (rc=$rc)" }
Write-Host "[+] done. Pick the best epoch via an XY plot of strength {0.5,0.75,0.9} x seeds in IL_1_Base."
