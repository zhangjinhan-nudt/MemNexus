@echo off
REM Overnight 150-item A/B: default x3, no_grounding x2, no_normalize x2
REM Resumable: finished cells are skipped, so re-running this file is safe.
cd /d "%~dp0"
"SimpleMem\.venv\Scripts\python.exe" experiments\overnight_run.py --deadline 07:00 --budget 28 >> result\overnight_20261021\driver.out 2>> result\overnight_20261021\driver.err
