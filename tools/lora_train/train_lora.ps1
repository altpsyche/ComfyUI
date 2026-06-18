#requires -Version 5.1
<#
.SYNOPSIS
  Train ONE character LoRA with kohya sd-scripts. Portable + parameterized + auto-captioning.

.DESCRIPTION
  Convention: images live in <repo>/output/dataset/<Char>/ (the IL_DatasetEdit_<Char> graph saves there directly).
  This script validates the folder, auto-captions it (WD14 tagger + prep_captions) if no .txt
  captions exist, derives num_repeats from a target step count, writes the dataset .toml, and
  launches training. No per-character file copies, no hardcoded paths, no manual file moving.

  Training parameters are DATA-DRIVEN (train.toml) and layered, resolved by train_config.py:
      explicit CLI flag  >  -Profile <name>  >  [train.<char>]  >  [defaults]
  so you can tune dim/steps/LR/resolution/etc. without editing this script. Use -DryRun to print the
  fully-resolved set + the exact accelerate command + the generated dataset TOML, without training.

.EXAMPLE
  .\train_lora.ps1 -Char aria -Prune "auburn hair,long hair,green eyes,freckles"
.EXAMPLE
  .\train_lora.ps1 -Char kael -Trigger kaelchar -Steps 2000 -TrainTextEncoder
.EXAMPLE
  .\train_lora.ps1 -Char nyx -Profile complex -DryRun
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$Char,   # dataset folder name + default LoRA name
    [string]$Trigger,                        # caption trigger token (default: <Char>char)
    [string]$Outfit,                         # roster outfit string; its garments auto-bake into the trigger (default: from roster.json)
    [string]$Prune = "",                     # EXTRA exact/head-noun tags to bake (outfit is auto-pruned from -Outfit)
    [string]$Keep = "",                      # garment tags to KEEP promptable (NOT baked) so they're add/removable at inference (default: from roster.json)
    [string]$Base,                           # checkpoint (default: oneObsession in models/checkpoints)
    [switch]$TrainTextEncoder,               # also train the TE (drops --network_train_unet_only)
    [switch]$SkipCaption,                    # don't auto-caption (captions already prepared)
    [switch]$DryRun,                         # print resolved params + command + TOML, do NOT train
    [switch]$Force,                          # train even if the outfit baked NOTHING (skip the zero-bake guard)

    # --- training params: unset here on purpose so train_config.py supplies the value from train.toml.
    #     Pass any of these to OVERRIDE for one run (highest precedence). ---
    [string]$Profile,                                                   # named preset from train.toml [profiles.*]
    [int]$Dim, [int]$Alpha,
    [ValidateSet('prodigy','adamw','adafactor')] [string]$Optimizer,    # Blackwell-safe set; NOT adamw8bit
    [double]$DCoef,                                                     # Prodigy d_coef (e.g. 0.8 on small sets)
    [int]$Steps, [int]$Epochs, [int]$Batch, [int]$MinImages,
    [string]$Lr, [string]$UnetLr, [string]$TextEncoderLr,              # adamw/adafactor LR override (prodigy stays 1.0)
    [string]$Scheduler, [string]$MinSnr, [string]$SavePrecision,
    [int]$Resolution, [int]$SaveEveryNEpochs, [int]$BucketMin, [int]$BucketMax
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
# roster defaults (trigger/prune/outfit/keep) from the build-generated manifest, unless overridden on the CLI.
# A MODULAR character carries an `outfits` map (and optional `outfit_keep`) instead of a single `outfit`:
# one LoRA, identity always-on (trigger) + a swappable token per outfit -> the modular branch below.
$Outfits = $null; $OutfitKeep = $null
$rosterFile = Join-Path $repo 'tools\lora_train\roster.json'
if (Test-Path $rosterFile) {
    # NB: WinPS 5.1 ConvertFrom-Json emits the array as one non-enumerated object — iterate with foreach.
    $roster = Get-Content $rosterFile -Raw | ConvertFrom-Json
    foreach ($e in $roster) {
        if ($e.name -eq $Char) {
            if (-not $Trigger) { $Trigger = [string]$e.trigger }
            if (-not $PSBoundParameters.ContainsKey('Prune') -and $e.prune) { $Prune = [string]$e.prune }
            if (($e.PSObject.Properties.Name -contains 'outfits') -and $e.outfits) {
                $Outfits = $e.outfits           # PSCustomObject: property name = outfit, value = garments
                $OutfitKeep = $e.outfit_keep    # PSCustomObject: outfit -> garments left promptable (optional)
            } else {
                if (-not $PSBoundParameters.ContainsKey('Outfit') -and $e.outfit) { $Outfit = [string]$e.outfit }
                if (-not $PSBoundParameters.ContainsKey('Keep') -and $e.keep) { $Keep = [string]$e.keep }
            }
            break
        }
    }
}
$Modular = $null -ne $Outfits
if (-not $Trigger) { $Trigger = "${Char}char" }
if (-not $Base)    { $Base = Join-Path $repo 'models\checkpoints\oneObsession_v19Atypical.safetensors' }

function Die($msg) { Write-Host "[x] $msg" -ForegroundColor Red; exit 1 }
function Warn($msg) { Write-Host "[!] $msg" -ForegroundColor Yellow }

# --- the venv must exist to resolve params and (later) train ---
if (-not (Test-Path $py)) { Die "trainer venv missing. Run: setup.bat --with-trainer" }

# --- resolve training params (train.toml defaults < per-char < profile < explicit CLI flags) ---
# Map each CLI flag to its train.toml key; only pass flags the user actually set (so unset = use toml).
$flagMap = [ordered]@{
    Dim='dim'; Alpha='alpha'; Optimizer='optimizer'; DCoef='d_coef';
    Steps='steps'; Epochs='epochs'; Batch='batch'; MinImages='min_images';
    Lr='lr'; UnetLr='unet_lr'; TextEncoderLr='text_encoder_lr';
    Scheduler='lr_scheduler'; MinSnr='min_snr_gamma'; SavePrecision='save_precision';
    Resolution='resolution'; SaveEveryNEpochs='save_every_n_epochs';
    BucketMin='min_bucket_reso'; BucketMax='max_bucket_reso'
}
$setArgs = @()
foreach ($p in $flagMap.Keys) {
    if ($PSBoundParameters.ContainsKey($p)) {
        $setArgs += @('--set', ("{0}={1}" -f $flagMap[$p], (Get-Variable $p).Value))
    }
}
if ($TrainTextEncoder) { $setArgs += @('--set', 'train_text_encoder=true') }
$cfgArgs = @((Join-Path $PSScriptRoot 'train_config.py'), '--char', $Char)
if ($Profile) { $cfgArgs += @('--profile', $Profile) }
$cfgArgs += $setArgs
$cfgJson = & $py @cfgArgs
if ($LASTEXITCODE -ne 0) { Die "train_config.py failed:`n$cfgJson" }
$cfg = $cfgJson | ConvertFrom-Json

# --- pre-flight (soft under -DryRun so you can preview params without a dataset/checkpoint) ---
if (-not (Test-Path $Base)) {
    if ($DryRun) { Warn "checkpoint not found: $Base (pass -Base <path>)" } else { Die "checkpoint not found: $Base  (pass -Base <path>)" }
}
# Build the per-outfit subset list (modular) up front: one subfolder output/dataset/<char>/<outfit>/
# per outfit, with that outfit's garments + promptable-keep from the roster.
$subsets = @()
if ($Modular) {
    foreach ($p in $Outfits.PSObject.Properties) {
        $oname = $p.Name
        $odir  = Join-Path $data $oname
        $ocnt  = 0
        if (Test-Path $odir) {
            $ocnt = @(Get-ChildItem $odir -File | Where-Object { $_.Extension -in '.png','.jpg','.jpeg','.webp' }).Count
        }
        $okeep = ""
        if ($OutfitKeep -and ($OutfitKeep.PSObject.Properties.Name -contains $oname)) { $okeep = [string]$OutfitKeep.$oname }
        $subsets += [pscustomobject]@{ Name=$oname; Garments=[string]$p.Value; Keep=$okeep; Dir=$odir; Count=$ocnt }
    }
}

if ($Modular) {
    Write-Host "[+] modular character '$Char': $($subsets.Count) outfits ($($subsets.Name -join ', '))"
    foreach ($s in $subsets) {
        if ($s.Count -lt [int]$cfg.min_images) {
            if ($DryRun) {
                Warn "outfit '$($s.Name)': $($s.Count) images in $($s.Dir) (need >= $($cfg.min_images))"
                if ($s.Count -eq 0) { $s.Count = [int]$cfg.min_images }   # placeholder so the preview can plan
            } else {
                Die ("outfit '$($s.Name)': only $($s.Count) images in $($s.Dir) (need >= $($cfg.min_images)). " +
                     "Generate with IL_DatasetEdit_${Char}_$($s.Name) (SaveImage prefix 'dataset/$Char/$($s.Name)').")
            }
        } else { Write-Host "[+]   $($s.Name): $($s.Count) images" }
    }
    $imgCount = ($subsets | Measure-Object -Property Count -Sum).Sum
} else {
    $haveData = Test-Path $data
    $imgs = @()
    if ($haveData) {
        $imgs = @(Get-ChildItem $data -File | Where-Object { $_.Extension -in '.png','.jpg','.jpeg','.webp' })
    }
    $imgCount = $imgs.Count
    if (-not $haveData) {
        if ($DryRun) { Warn "no dataset at $data — using min_images ($($cfg.min_images)) as a placeholder for the preview"; $imgCount = [int]$cfg.min_images }
        else { Die "no dataset at $data  -- generate with IL_DatasetEdit (SaveImage prefix 'dataset/$Char')" }
    } elseif ($imgCount -lt [int]$cfg.min_images) {
        if ($DryRun) { Warn "only $imgCount images in $data (need >= $($cfg.min_images))" }
        else { Die "only $imgCount images in $data (need >= $($cfg.min_images)). Generate/curate more." }
    } else {
        Write-Host "[+] $imgCount images in output/dataset/$Char"
    }
}

# --- caption (only if not already done; skipped entirely on -DryRun) ---
# Captions each folder, then runs prep_captions to bake the outfit/identity into the trigger. For a
# modular character this happens per outfit subfolder with a TWO-token trigger "<trigger>, <char>_<outfit>"
# (identity always-on + this outfit's swappable token), the identity pruned on every frame and this
# outfit's garments baked into its token; for a single-outfit character it's the original single call.
function Invoke-Prep($folder, $trigger, $outfit, $keep, $label) {
    $haveCaptions = @(Get-ChildItem $folder -Filter *.txt -File).Count -gt 0
    if ($SkipCaption) {
        Write-Host "[-] -SkipCaption: using existing captions$label"
    } elseif ($haveCaptions) {
        Write-Host "[+] captions present$label - skipping tagger (delete .txt to re-tag)"
    } else {
        Write-Host "[+] auto-captioning$label (WD14 tagger, trigger '$trigger')"
        Push-Location $sdDir
        & $py 'finetune/tag_images_by_wd14_tagger.py' --onnx `
            --repo_id 'SmilingWolf/wd-v1-4-convnextv2-tagger-v2' --batch_size 4 $folder
        $rc = $LASTEXITCODE
        Pop-Location
        if ($rc -ne 0) { Die "WD14 tagger failed$label (rc=$rc)" }
        # NB: only pass --prune/--outfit/--keep when non-empty -- PowerShell drops an empty-string arg,
        # leaving argparse to see the flag with no value (errors). prep_captions defaults them to "".
        $prepArgs = @($folder, '--trigger', $trigger)
        if ($outfit) { $prepArgs += @('--outfit', $outfit) }   # auto-bakes the outfit (colour variants too)
        if ($Prune)  { $prepArgs += @('--prune', $Prune) }     # identity lock, baked on every frame
        if ($keep)   { $prepArgs += @('--keep', $keep) }       # these garments stay promptable (removable)
        # Abort if the outfit/identity baked NOTHING (a vocab mismatch). -Force skips this.
        if (($outfit -or $Prune) -and -not $Force) { $prepArgs += '--strict' }
        & $py (Join-Path $PSScriptRoot 'prep_captions.py') @prepArgs
        $prc = $LASTEXITCODE
        if ($prc -eq 2) {
            Die ("outfit/prune matched NO caption tags$label -- it did NOT bake into '$trigger', so the LoRA " +
                 "would not carry it. Fix the words in characters.toml to match the WD14 tagger's tags " +
                 "(open a .txt in $folder to see them), or re-run with -Force to train anyway.")
        } elseif ($prc -ne 0) { Die "prep_captions failed$label (rc=$prc)" }
    }
}

if ($DryRun) {
    Write-Host "[-] -DryRun: skipping captioning"
} elseif ($Modular) {
    foreach ($s in $subsets) {
        Invoke-Prep $s.Dir "$Trigger, ${Char}_$($s.Name)" $s.Garments $s.Keep " ['$($s.Name)']"
    }
} else {
    Invoke-Prep $data $Trigger $Outfit $Keep ""
}

# --- write the dataset config (generated; not a hand-copied per-char file) ---
$cacheDir = Join-Path $PSScriptRoot '.cache'
New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
$cfgPath  = Join-Path $cacheDir "$Char.toml"

if ($Modular) {
    # MULTI-SUBSET config (one subset per outfit, keep_tokens=2). Balanced num_repeats + the TOML come
    # from dataset_plan.py (pure, unit-tested) so each outfit contributes an equal step-share regardless
    # of frame count -- no outfit dominates, identity (on every frame) still accumulates the full budget.
    $planArgs = @((Join-Path $PSScriptRoot 'dataset_plan.py'), '--char', $Char,
        '--steps', [int]$cfg.steps, '--epochs', [int]$cfg.epochs, '--batch', [int]$cfg.batch,
        '--resolution', [int]$cfg.resolution, '--bucket-min', [int]$cfg.min_bucket_reso,
        '--bucket-max', [int]$cfg.max_bucket_reso, '--bucket-steps', [int]$cfg.bucket_reso_steps,
        '--keep-tokens', 2)
    foreach ($s in $subsets) {
        $planArgs += @('--subset', ("{0}={1}" -f ($s.Dir -replace '\\','/'), [int]$s.Count))
    }
    $planOut = & $py @planArgs
    if ($LASTEXITCODE -ne 0) { Die "dataset_plan.py failed:`n$planOut" }
    $tomlText = ($planOut -join "`n")
    Write-Host "[+] $imgCount imgs across $($subsets.Count) outfits -> multi-subset config (keep_tokens=2, balanced num_repeats, target $($cfg.steps) steps)"
} else {
    # --- derive num_repeats from the target step count ---
    #   steps ~= images * repeats * epochs / batch  ->  repeats = round(steps * batch / (images * epochs))
    $repeats = [int][math]::Max(1, [math]::Round($cfg.steps * $cfg.batch / ($imgCount * $cfg.epochs)))
    $actual  = [int]($imgCount * $repeats * $cfg.epochs / $cfg.batch)
    Write-Host "[+] $imgCount imgs x $repeats repeats x $($cfg.epochs) epochs / batch $($cfg.batch)  ~= $actual steps (target $($cfg.steps))"
    # warn if the integer num_repeats forces a big drift from the requested step count
    $dev = [math]::Abs($actual - $cfg.steps) / [double]$cfg.steps
    if ($dev -gt 0.25) { Warn ("derived ~{0} steps is {1:P0} off the target {2} (num_repeats rounds to an int on {3} images); adjust -Steps/-Epochs if it matters" -f $actual, $dev, [int]$cfg.steps, $imgCount) }

    $dataFwd  = $data -replace '\\','/'
    $tomlText = @"
# AUTO-GENERATED by train_lora.ps1 for character '$Char' -- do not hand-edit.
[general]
shuffle_caption = true
keep_tokens = 1
caption_extension = ".txt"

[[datasets]]
resolution = $($cfg.resolution)
batch_size = $($cfg.batch)
enable_bucket = true
min_bucket_reso = $($cfg.min_bucket_reso)
max_bucket_reso = $($cfg.max_bucket_reso)
bucket_reso_steps = $($cfg.bucket_reso_steps)

  [[datasets.subsets]]
  image_dir = "$dataFwd"
  num_repeats = $repeats
"@
}
# Write UTF-8 WITHOUT a BOM: WinPS 5.1 `Set-Content -Encoding UTF8` prepends a BOM, which the
# sd-scripts `toml` reader rejects ("invalid character in key name: '#'").
[System.IO.File]::WriteAllText($cfgPath, $tomlText, (New-Object System.Text.UTF8Encoding($false)))

# --- optimizer block (Prodigy is the default; AdamW / AdaFactor are Blackwell-safe alternatives.
# NOT AdamW8bit: bitsandbytes on sm_120 is unverified -- the reason Prodigy was chosen. clip_skip is
# deliberately NOT passed: sdxl_train_network ignores it for SDXL.) LR overrides apply to adamw/adafactor;
# prodigy LR must stay 1.0 (tune it via -DCoef / d_coef). ---
$adLr   = if ($cfg.lr)              { [string]$cfg.lr }              else { '3e-4' }
$adUnet = if ($cfg.unet_lr)         { [string]$cfg.unet_lr }         else { '3e-4' }
$adTe   = if ($cfg.text_encoder_lr) { [string]$cfg.text_encoder_lr } else { '3e-5' }
switch ($cfg.optimizer) {
    'adamw' {
        $optArgs = @('--optimizer_type','AdamW',
                     '--learning_rate',$adLr,'--unet_lr',$adUnet,'--text_encoder_lr',$adTe)
    }
    'adafactor' {
        $optArgs = @('--optimizer_type','Adafactor',
                     '--optimizer_args','relative_step=False','scale_parameter=False','warmup_init=False',
                     '--learning_rate',$adLr,'--unet_lr',$adUnet,'--text_encoder_lr',$adTe)
    }
    default {   # prodigy: auto-tunes LR (lr must be 1.0); d_coef knobs effective LR
        $optArgs = @('--optimizer_type','prodigy',
                     '--learning_rate','1.0','--unet_lr','1.0','--text_encoder_lr','1.0',
                     '--optimizer_args','decouple=True','weight_decay=0.01',"d_coef=$($cfg.d_coef)",'use_bias_correction=True','safeguard_warmup=True')
    }
}

# --- train ---
$trainArgs = @(
    'launch','--num_cpu_threads_per_process',$cfg.num_cpu_threads,'sdxl_train_network.py',
    '--pretrained_model_name_or_path',$Base,
    '--dataset_config',$cfgPath,
    '--output_dir',$outDir,'--output_name',"${Char}_v1",
    '--network_module',$cfg.network_module,'--network_dim',$cfg.dim,'--network_alpha',$cfg.alpha
)
$trainArgs += $optArgs
$trainArgs += @(
    '--lr_scheduler',$cfg.lr_scheduler,
    '--max_train_epochs',$cfg.epochs,'--save_every_n_epochs',$cfg.save_every_n_epochs,
    '--train_batch_size',$cfg.batch,
    '--mixed_precision',$cfg.mixed_precision,'--save_precision',$cfg.save_precision,
    '--min_snr_gamma',$cfg.min_snr_gamma,'--seed',$cfg.seed
)
# Blackwell / 16 GB safety toggles (on by default; train.toml can disable them, with the documented risk).
if ($cfg.gradient_checkpointing) { $trainArgs += '--gradient_checkpointing' }
if ($cfg.cache_latents)          { $trainArgs += '--cache_latents' }
if ($cfg.cache_latents_to_disk)  { $trainArgs += '--cache_latents_to_disk' }
if ($cfg.sdpa)                   { $trainArgs += '--sdpa' }
if ($cfg.no_half_vae)            { $trainArgs += '--no_half_vae' }
if (-not $cfg.train_text_encoder) { $trainArgs += '--network_train_unet_only' }   # 16 GB-safe default

$optDesc = if ($cfg.optimizer -eq 'prodigy') { "prodigy d_coef=$($cfg.d_coef)" } else { $cfg.optimizer }

# --- DryRun: show everything, train nothing ---
if ($DryRun) {
    Write-Host "`n===== DRY RUN ($Char) =====" -ForegroundColor Cyan
    Write-Host "resolved params (train.toml defaults < per-char < profile < CLI):" -ForegroundColor Cyan
    Write-Host $cfgJson
    Write-Host "`ndataset config ($cfgPath):" -ForegroundColor Cyan
    Write-Host $tomlText
    Write-Host "`naccelerate command:" -ForegroundColor Cyan
    Write-Host ("{0} {1}" -f $acc, ($trainArgs -join ' '))
    Write-Host "`n[-] DryRun: nothing trained." -ForegroundColor Yellow
    exit 0
}

# --- guard against silently clobbering a previous run's LoRA + epochs ---
$loraPath = Join-Path $outDir "${Char}_v1.safetensors"
if (Test-Path $loraPath) { Warn "models/loras/${Char}_v1.safetensors exists — this run will overwrite it (and its per-epoch checkpoints)" }

Write-Host "[+] training ${Char}_v1 (dim $($cfg.dim) / alpha $($cfg.alpha), $optDesc) -> models/loras/${Char}_v1.safetensors"
Push-Location $sdDir                         # sd-scripts imports library.* relative to its dir
& $acc @trainArgs
$rc = $LASTEXITCODE
Pop-Location

# --- provenance: record the resolved params + exact command next to the LoRA ---
$argsLog = Join-Path $outDir "${Char}_v1.args.txt"
$logText = "# train_lora.ps1 run for '$Char' on $(Get-Date -Format o)`n# resolved params:`n$cfgJson`n`n# command:`n$acc $($trainArgs -join ' ')`n"
[System.IO.File]::WriteAllText($argsLog, $logText, (New-Object System.Text.UTF8Encoding($false)))

if ($rc -ne 0) { Die "training failed (rc=$rc)" }
Write-Host "[+] done. Provenance: models/loras/${Char}_v1.args.txt"
Write-Host "[+] Pick the best epoch via an XY plot of strength {0.5,0.75,0.9} x seeds in IL_1_Base."
