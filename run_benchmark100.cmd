@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
if not exist "SimpleMem\.venv\Scripts\python.exe" (
  echo Python environment missing: SimpleMem\.venv\Scripts\python.exe
  pause
  exit /b 1
)
"SimpleMem\.venv\Scripts\python.exe" "experiments\benchmark100.py" %*
set "benchmark_exit=%ERRORLEVEL%"
echo.
if not "%benchmark_exit%"=="0" echo Run stopped. Read the message above and run this file again to resume.
pause
exit /b %benchmark_exit%
