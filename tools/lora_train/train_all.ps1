#requires -Version 5.1
<#
.SYNOPSIS
  Train a LoRA for every roster character that has a curated dataset.

.DESCRIPTION
  Reads roster.json (written by build_il_graphs.py from the CHARACTERS roster) and calls
  train_lora.ps1 -Char <name> for each entry whose output/dataset/<name>/ folder has images.
  Characters you haven't generated/curated yet are skipped with a notice. Per-character trigger
  and prune come from the roster. Extra args (e.g. -Steps 2000) pass through to every character.

.EXAMPLE
  .\train_all.ps1
.EXAMPLE
  .\train_all.ps1 -Steps 2000          # override for all characters
#>
[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments)] $PassThrough)
$ErrorActionPreference = 'Continue'

$repo       = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$rosterFile = Join-Path $repo 'tools\lora_train\roster.json'
$trainOne   = Join-Path $PSScriptRoot 'train_lora.ps1'
if (-not (Test-Path $rosterFile)) { Write-Host "[x] no roster.json - run build_il_graphs.py first" -ForegroundColor Red; exit 1 }

# NB: WinPS 5.1 ConvertFrom-Json emits the array as one non-enumerated object — don't @()-wrap; foreach iterates it.
$roster = Get-Content $rosterFile -Raw | ConvertFrom-Json
$total = @($roster).Count
$trained = 0; $skipped = @()
foreach ($c in $roster) {
    $data = Join-Path $repo "output\dataset\$($c.name)"
    $imgs = @(Get-ChildItem $data -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in '.png','.jpg','.jpeg','.webp' })
    if ($imgs.Count -lt 12) {
        Write-Host "[-] skip $($c.name): $($imgs.Count) images in output/dataset/$($c.name)/ (generate + curate first)" -ForegroundColor Yellow
        $skipped += $c.name
        continue
    }
    Write-Host "`n===== training $($c.name) =====" -ForegroundColor Cyan
    & powershell -NoProfile -ExecutionPolicy Bypass -File $trainOne -Char $c.name @PassThrough
    if ($LASTEXITCODE -eq 0) { $trained++ } else { Write-Host "[x] $($c.name) failed" -ForegroundColor Red }
}
Write-Host "`n[+] trained $trained / $total roster characters." -ForegroundColor Green
if ($skipped) { Write-Host "    skipped (no dataset yet): $($skipped -join ', ')" }
