@echo off
rem Convenience alias for `dev setup` (Windows). See `dev.bat setup --help`.
call "%~dp0dev.bat" setup %*
exit /b %ERRORLEVEL%
