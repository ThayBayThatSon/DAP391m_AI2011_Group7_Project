@echo off
title Update & Run California AQI Forecasting Dashboard
echo ========================================================
echo   1. Updating Code & Data from GitHub...
echo ========================================================
cd /d "%~dp0"
git stash push -m "local-changes"
git pull origin main

echo.
echo ========================================================
echo   2. Installing/Updating Python Dependencies...
echo ========================================================
call .\.venv\Scripts\activate.bat
python -m pip install -r requirements.txt

echo.
echo ========================================================
echo   3. Backfilling Historical Model Predictions...
echo ========================================================
python scr\backfill_historical_model_predictions.py

echo.
echo ========================================================
echo   4. Starting Streamlit Dashboard...
echo ========================================================
python -m streamlit run app\ui.py --server.port 8501 --server.address 0.0.0.0
pause
