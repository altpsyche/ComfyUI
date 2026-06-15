@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
:: ComfyUI v9 onboarding setup
:: ============================================================================
:: One-shot install for a fresh checkout. Idempotent; safe to re-run.
::
:: Usage:
::   setup.bat                    # NVIDIA autodetect (default)
::   setup.bat --gpu <mode>       # override GPU mode (see below)
::   setup.bat --skip-torch       # skip torch (already installed)
::   setup.bat --with-trainer     # also provision the LoRA-training venv (sd-scripts, Blackwell torch)
::   setup.bat --no-color         # disable ANSI colors
::
:: --gpu modes:
::   nvidia       autodetect via nvidia-smi (default)
::   amd-rdna3    Windows AMD RX 7000 series (gfx110X)
::   amd-rdna35   Windows AMD Strix halo / Ryzen AI Max+ (gfx1151)
::   amd-rdna4    Windows AMD RX 9000 series (gfx120X)
::   intel-xpu    Intel Arc XPU
::   cpu          CPU-only wheel
::
:: Prerequisites (phase 1 verifies):
::   - Python 3.10+
::   - git
::   - SSH key authorized for github.com
::   - (optional) NVIDIA GPU + drivers for CUDA
:: ============================================================================

pushd "%~dp0" >nul
set "ROOT=%CD%"
title ComfyUI v9 setup
if not "%~1"=="--no-color" color 0a

set "SKIP_TORCH=0"
set "GPU_MODE=nvidia"
set "WITH_TRAINER=0"
:parse
if "%~1"=="" goto begin
if /I "%~1"=="--skip-torch" set "SKIP_TORCH=1"
if /I "%~1"=="--with-trainer" set "WITH_TRAINER=1"
if /I "%~1"=="--gpu" (
    set "GPU_MODE=%~2"
    shift
)
shift
goto parse

:begin
echo.
echo ================================================
echo  ComfyUI v9 Setup
echo  Root: %ROOT%
echo ================================================
echo.

:: ----------------------------------------------------------------------------
:: [1/6] Prereqs
:: ----------------------------------------------------------------------------
echo [1/6] Verifying prerequisites...

where python >nul 2>&1
if errorlevel 1 (
    echo   [x] python not found on PATH
    echo       Install Python 3.10+ from https://www.python.org/downloads/
    goto fail
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set "PYVER=%%i"
echo   [+] python %PYVER%

where git >nul 2>&1
if errorlevel 1 (
    echo   [x] git not found on PATH
    goto fail
)
for /f "tokens=3" %%i in ('git --version') do set "GITVER=%%i"
echo   [+] git %GITVER%

echo   ... testing ssh -T git@github.com (may prompt to accept host key on first run)
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=accept-new git@github.com >nul 2>&1
if errorlevel 1 (
    ssh -T git@github.com 2>&1 | findstr /C:"successfully authenticated"
    if errorlevel 1 (
        echo   [x] SSH auth to github.com failed
        echo       Set up an SSH key: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
        goto fail
    )
)
echo   [+] SSH to github.com works

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   [!] nvidia-smi not found - GPU optional but expected for production gen
    goto :after_gpu
)
for /f "delims=" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do (
    echo   [+] GPU: %%i
)
:after_gpu
echo.

:: ----------------------------------------------------------------------------
:: [2/6] Submodules
:: ----------------------------------------------------------------------------
echo [2/6] Initializing submodules (this can take a few minutes on first run)...
git submodule update --init --recursive --jobs 4
if errorlevel 1 (
    echo   [!] Some submodules failed; retrying with --depth 1
    git submodule update --init --recursive --depth 1 --jobs 4
    if errorlevel 1 (
        echo   [x] Submodule init failed
        goto fail
    )
)
for /f %%c in ('git submodule status ^| find /c "custom_nodes"') do set "SUBCOUNT=%%c"
echo   [+] %SUBCOUNT% submodules initialized
echo.

:: ----------------------------------------------------------------------------
:: [3/6] venv
:: ----------------------------------------------------------------------------
echo [3/6] Setting up venv...
if not exist "venv\Scripts\activate.bat" (
    python -m venv venv
    if errorlevel 1 (
        echo   [x] venv creation failed
        goto fail
    )
    echo   [+] Created venv
) else (
    echo   [+] venv already exists
)
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo   [x] Failed to activate venv
    goto fail
)
python -m pip install -U pip wheel setuptools >nul 2>&1
echo   [+] pip/wheel/setuptools upgraded
echo.

:: ----------------------------------------------------------------------------
:: [4/6] Core requirements + torch
:: ----------------------------------------------------------------------------
echo [4/6] Installing ComfyUI core requirements...
if exist requirements.txt (
    rem only-if-needed: upgrade pinned packages (frontend etc.) without
    rem clobbering CUDA torch with the unpinned PyPI CPU wheel
    pip install --upgrade --upgrade-strategy only-if-needed -r requirements.txt
    if errorlevel 1 (
        echo   [x] ComfyUI requirements install failed
        goto fail
    )
    echo   [+] ComfyUI requirements installed/upgraded
) else (
    echo   [!] requirements.txt missing - skipping core install
)

if "%SKIP_TORCH%"=="0" (
    echo   ... ensuring torch + torchvision for GPU mode: %GPU_MODE%
    powershell -ExecutionPolicy Bypass -File "%ROOT%\scripts\install_torch.ps1" -GpuMode "%GPU_MODE%"
    if errorlevel 1 (
        echo   [!] torch install reported issues - check output above
    )
) else (
    echo   [-] Skipping torch install per --skip-torch
)
echo.

:: ----------------------------------------------------------------------------
:: [5/6] Custom node requirements
:: ----------------------------------------------------------------------------
echo [5/6] Installing custom-node requirements (loops over custom_nodes\)...
powershell -ExecutionPolicy Bypass -File "%ROOT%\scripts\install_node_reqs.ps1"
if errorlevel 1 (
    echo   [!] Some custom node pip installs failed - see log above
)
echo.

:: ----------------------------------------------------------------------------
:: [optional] LoRA trainer venv  (--with-trainer)
:: ----------------------------------------------------------------------------
if "%WITH_TRAINER%"=="1" (
    echo [+] Provisioning LoRA trainer venv ^(sd-scripts submodule, Blackwell torch^)...
    powershell -ExecutionPolicy Bypass -File "%ROOT%\scripts\install_trainer.ps1"
    if errorlevel 1 (
        echo   [!] trainer venv install reported issues - see above
    )
    echo.
)

:: ----------------------------------------------------------------------------
:: [6/6] Verify
:: ----------------------------------------------------------------------------
echo [6/6] Verifying install...
powershell -ExecutionPolicy Bypass -File "%ROOT%\scripts\verify.ps1"
set "VERIFY_RC=%ERRORLEVEL%"
echo.

if not "%VERIFY_RC%"=="0" (
    echo ================================================
    echo  Setup completed with WARNINGS - see above
    echo ================================================
    goto eof
)

echo ================================================
echo  Setup complete!
echo ================================================
echo.
echo Next steps:
echo   1. (Workflows + models live in a separate repo - see ONBOARDING.md)
echo   2. run_comfy.bat                - launch ComfyUI
if "%WITH_TRAINER%"=="1" echo   3. tools\lora_train\README.md   - generate data, caption, train a character LoRA
echo.
goto eof

:fail
echo.
echo ================================================
echo  Setup FAILED - fix errors above and re-run
echo ================================================
popd >nul
endlocal
exit /b 1

:eof
popd >nul
endlocal
exit /b 0
