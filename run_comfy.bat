@echo off
rem Convenience alias for `dev run` (Windows) — launches ComfyUI in the main venv.
rem Double-clickable: pause at the end so the window stays open to read the exit status.
call "%~dp0dev.bat" run %*
set ERR=%ERRORLEVEL%
echo.
if %ERR% NEQ 0 (echo [x] ComfyUI exited with error code %ERR%.) else (echo [+] ComfyUI closed normally.)
pause
exit /b %ERR%
