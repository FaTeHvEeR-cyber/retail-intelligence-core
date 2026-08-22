@echo off
title Retail Intelligence ^& Forecasting Dashboard
echo ============================================================
echo   Launching Retail Intelligence ^& Forecasting Dashboard...
echo ============================================================
echo.
echo Dashboard URL: http://localhost:8501
echo.

:: Clear any lingering process on port 8501
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)

if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -m streamlit run app.py --server.address localhost --server.port 8501
) else (
    python -m streamlit run app.py --server.address localhost --server.port 8501
)

pause
