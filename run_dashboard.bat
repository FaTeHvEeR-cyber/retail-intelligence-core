@echo off
title Retail Intelligence & Forecasting Dashboard
echo ============================================================
echo   Launching Retail Intelligence & Forecasting Dashboard...
echo ============================================================
echo.
echo Dashboard URL: http://localhost:8501
echo.
.venv\Scripts\python.exe -m streamlit run app.py --server.address localhost --server.port 8501
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Trying fallback using global python...
    python -m streamlit run app.py --server.address localhost --server.port 8501
)
pause
