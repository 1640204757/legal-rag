@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0;%~dp0.local-deps"
set "PYTHONIOENCODING=utf-8"
if defined LOCAL_RAG_PYTHON goto configured
if not exist ".local-python-path.txt" goto fallback
set /p "LOCAL_RAG_PYTHON="<".local-python-path.txt"
:configured
"%LOCAL_RAG_PYTHON%" -m local_rag.index
set "INDEX_EXIT=%errorlevel%"
if not "%INDEX_EXIT%"=="0" pause
exit /b %INDEX_EXIT%
:fallback
python -m local_rag.index
set "INDEX_EXIT=%errorlevel%"
if not "%INDEX_EXIT%"=="0" pause
exit /b %INDEX_EXIT%
