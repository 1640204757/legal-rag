@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0;%~dp0.local-deps"
set "PYTHONIOENCODING=utf-8"
if defined LOCAL_RAG_PYTHON goto configured
if not exist ".local-python-path.txt" goto fallback
set /p "LOCAL_RAG_PYTHON="<".local-python-path.txt"
:configured
"%LOCAL_RAG_PYTHON%" run_local.py
if errorlevel 1 pause
exit /b
:fallback
python run_local.py
if errorlevel 1 pause
