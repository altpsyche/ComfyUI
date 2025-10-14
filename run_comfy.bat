@echo off
setlocal

:: Nice console title and color
title ComfyUI Launcher
color 0a

:: Ensure script runs from this folder (important if launched via shortcut)
pushd "%~dp0" >nul

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
  echo [!] Virtual environment not found: venv\Scripts\activate.bat
  echo     Please run your install/setup script first.
  echo.
  pause
  popd >nul
  endlocal
  exit /b 1
)

echo [√] Activating virtual environment...
call "venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [x] Failed to activate venv.
  echo.
  pause
  popd >nul
  endlocal
  exit /b 1
)
echo.

:: Optional: Silence some warnings / improve stability
set "PYTHONUTF8=1"
set "HF_HUB_DISABLE_TELEMETRY=1"

echo [→] Launching ComfyUI...
echo.

python main.py
set ERR=%ERRORLEVEL%

echo.
if %ERR% NEQ 0 (
  echo [x] ComfyUI exited with error code %ERR%.
  echo     Check the log/console above for details.
) else (
  echo [√] ComfyUI closed normally.
)
echo.
pause

popd >nul
endlocal
