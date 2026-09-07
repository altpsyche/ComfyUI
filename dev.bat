@echo off
rem ComfyUI dev toolkit dispatcher (Windows).
rem
rem The toolkit lives in the private comfy-devkit repo, cloned beside this one. This stub exports
rem COMFY_ROOT (so the devkit knows which checkout to act on) and hands off.
setlocal
set "COMFY_ROOT=%~dp0"
if "%COMFY_ROOT:~-1%"=="\" set "COMFY_ROOT=%COMFY_ROOT:~0,-1%"

if not defined COMFY_DEVKIT (
    for %%I in ("%COMFY_ROOT%\..") do set "COMFY_DEVKIT=%%~fI\comfy-devkit"
)
if not exist "%COMFY_DEVKIT%\devkit\cli.py" (
    echo comfy-devkit not found at %COMFY_DEVKIT%
    echo.
    echo The dev toolkit lives in a separate private repo. Clone it beside this one:
    echo     git clone git@github.com:altpsyche/comfy-devkit.git "%COMFY_DEVKIT%"
    echo.
    echo Or point COMFY_DEVKIT at an existing clone.
    exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
    echo python not found on PATH - install Python 3.10+ from https://www.python.org/downloads/
    exit /b 1
)
rem Deliberately no `pushd`: staying in the caller's directory keeps relative path arguments
rem resolving as the user typed them.
if defined PYTHONPATH (
    set "PYTHONPATH=%COMFY_DEVKIT%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%COMFY_DEVKIT%"
)
python -m devkit %*
exit /b %ERRORLEVEL%
