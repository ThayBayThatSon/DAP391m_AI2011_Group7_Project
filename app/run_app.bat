@echo off
title California AQI Forecasting Dashboard
echo ========================================================
echo   Starting California AQI Forecasting Dashboard...
echo ========================================================
cd /d "%~dp0"
call .\.venv\Scripts\activate.bat
python -m streamlit run app\ui.py --server.port 8501 --server.address 0.0.0.0
pause
