# Train one character LoRA with kohya sd-scripts on the RTX 5080 (Blackwell).
# Activate the venv first:  .\tools\lora_train\.venv\Scripts\Activate.ps1
# Copy for charB (change the 3 paths/name + the trigger used in captioning).
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"   # sd-scripts prints unicode/emoji progress; cp1252 console would crash it

$COMFY   = "C:/Users/vsiva/dev/ComfyUI"
$CKPT    = "$COMFY/models/checkpoints/oneObsession_v19Atypical.safetensors"
$DATACFG = "$COMFY/tools/lora_train/dataset_charA.toml"
$OUTDIR  = "$COMFY/models/loras"
$NAME    = "charA_aria_v1"

Set-Location "$COMFY/tools/sd-scripts"   # sd-scripts imports library.* relative to its own dir

accelerate launch --num_cpu_threads_per_process 8 sdxl_train_network.py `
  --pretrained_model_name_or_path "$CKPT" `
  --dataset_config "$DATACFG" `
  --output_dir "$OUTDIR" --output_name "$NAME" `
  --network_module networks.lora --network_dim 16 --network_alpha 8 `
  --optimizer_type prodigy `
  --learning_rate 1.0 --unet_lr 1.0 --text_encoder_lr 1.0 `
  --optimizer_args "decouple=True" "weight_decay=0.01" "d_coef=1.0" "use_bias_correction=True" "safeguard_warmup=True" `
  --lr_scheduler cosine `
  --max_train_epochs 10 --save_every_n_epochs 1 `
  --train_batch_size 2 --gradient_checkpointing `
  --mixed_precision bf16 --save_precision bf16 `
  --cache_latents --cache_latents_to_disk `
  --sdpa --no_half_vae `
  --min_snr_gamma 5 --seed 42 `
  --network_train_unet_only      # 16 GB-safe; drop this line to also train the text encoder

# Output: $OUTDIR/charA_aria_v1.safetensors  (+ one per epoch).
# Pick the best epoch via an XY plot of strength {0.5,0.75,0.9} x a few seeds in IL_1_Base.
