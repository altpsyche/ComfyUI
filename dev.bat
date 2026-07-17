@echo off
rem ComfyUI dev toolkit dispatcher (Windows).
rem Runs the stdlib-only dispatcher on the system python; commands re-exec into the right venv.
setlocal
cd /d "%~dp0"
where python >nul 2>&1
if errorlevel 1 (
    echo python not found on PATH - install Python 3.10+ from https://www.python.org/downloads/
    exit /b 1
)
python -m devtools %*
exit /b %ERRORLEVEL%
