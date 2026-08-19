@echo off
title Retail Intelligence & Forecasting Dashboard
echo ============================================================
echo   Launching Retail Intelligence & Forecasting Dashboard...
echo ============================================================
echo.
.venv\Scripts\python.exe -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Trying fallback using global python...
    python -m streamlit run app.py
)
pause
