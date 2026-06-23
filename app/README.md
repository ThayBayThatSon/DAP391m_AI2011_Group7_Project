# California AQI Forecasting Web App

Lightweight integrated web application for the California US AQI forecasting project.

Stack:

- FastAPI backend: `app/main.py`
- Streamlit dashboard: `app/ui.py`
- SQLite local database: `aqi_data.db` in the project root
- Background Open-Meteo collector: `app/data_collector.py`

The app supports exactly two model horizons:

- `1` hour: short-term nowcasting with local AQI lags from 1-3 hours.
- `24` hours: strict 24-hour forecasting with local AQI hidden from t-1 to t-23.

## 1. Install Dependencies

From the project root:

```powershell
pip install fastapi uvicorn streamlit apscheduler requests pandas numpy lightgbm
```

If you are using the project virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
pip install fastapi uvicorn streamlit apscheduler requests pandas numpy lightgbm
```

## 2. Expected Project Layout

Keep this structure:

```text
Project/
  app/
    data_collector.py
    main.py
    ui.py
    README.md
  data/
    processed/
      california_aqi_model_ready.csv
  models/
    lightgbm_nowcast.txt
    lightgbm_forecast24h.txt
  aqi_data.db
```

Notes:

- `aqi_data.db` is created automatically if it does not exist.
- If trained LightGBM model files are missing, the API uses a deterministic fallback predictor so the demo can still run.
- On FastAPI startup, historical AQI lag rows are bootstrapped from `data/processed/california_aqi_model_ready.csv` when SQLite has no AQI history yet.

## 3. Run Demo With Streamlit Only

For a simple demo on one machine, you only need Streamlit:

```powershell
streamlit run app/ui.py
```

In this mode, Streamlit imports the local prediction engine from `app/main.py` and runs prediction in the same Python process. You do not need to start `uvicorn`.

## 4. Optional: Run the FastAPI Backend Separately

From the project root:

```powershell
uvicorn app.main:app --reload
```

Default API URL:

```text
http://127.0.0.1:8000
```

Useful endpoints:

```text
GET  /health
POST /predict
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Example prediction request:

```powershell
$body = @{
  station_name = "Fresno"
  target_hour_ahead = 1
  temperature_2m = 24.5
  relative_humidity_2m = 42
  wind_speed_10m = 4.8
  wind_direction_10m = 270
  surface_pressure = 1008.2
  rain = 0
  cloud_cover = 15
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -Body $body
```

## 5. Run Streamlit Against a Separate Backend

Open a second terminal from the project root:

```powershell
$env:AQI_API_URL = "http://127.0.0.1:8000"
streamlit run app/ui.py
```

When `AQI_API_URL` is set, the dashboard calls the FastAPI backend at:

```text
http://127.0.0.1:8000
```

To use a different backend URL:

```powershell
$env:AQI_API_URL = "http://127.0.0.1:8000"
streamlit run app/ui.py
```

## 6. Run the Background Data Collector

Open a third terminal from the project root:

```powershell
python app/data_collector.py
```

The worker:

- Runs once immediately.
- Then runs every 60 minutes.
- Fetches hourly Open-Meteo weather for Fresno, Los Angeles, and San Jose.
- Computes VPD, cyclic time features, and wind U/V components.
- Upserts rows into `aqi_data.db`, table `meteorology_history`.

## 7. Recommended Demo Run Order

For demo:

```powershell
# Terminal 1
streamlit run app/ui.py

# Optional Terminal 2
python app/data_collector.py
```

For separated API + UI:

```powershell
# Terminal 1
uvicorn app.main:app --reload

# Terminal 2
streamlit run app/ui.py

# Optional Terminal 3
python app/data_collector.py
```

## 8. Model Files

Place trained LightGBM text models here:

```text
models/lightgbm_nowcast.txt
models/lightgbm_forecast24h.txt
```

The backend loads these paths automatically:

- `target_hour_ahead = 1` uses `models/lightgbm_nowcast.txt`
- `target_hour_ahead = 24` uses `models/lightgbm_forecast24h.txt`

Generate both files from the panel training script:

```powershell
python scr/train_combined_panel_models.py --lightgbm-only
```

This trains only the production LightGBM models and exports:

```text
models/lightgbm_nowcast.txt
models/lightgbm_forecast24h.txt
```

If the model files are absent, the app still runs with a deterministic fallback predictor for demo purposes.

## 9. Leakage-Control Rules

The backend excludes current PM2.5 and PM10 from the prediction matrix.

For `target_hour_ahead = 1`, it uses:

- Local AQI lags: 1h, 2h, 3h
- Cross-city spatial lags: 1h, 2h, 3h

For `target_hour_ahead = 24`, it uses:

- Local AQI lags: 24h, 48h, 72h
- Cross-city spatial lags: 24h, 48h, 72h
- No local AQI from t-1 through t-23

## 10. Troubleshooting

If FastAPI cannot start because dependencies are missing:

```powershell
pip install fastapi uvicorn pydantic pandas numpy lightgbm
```

If Streamlit cannot connect to the API when using `AQI_API_URL`:

- Make sure `uvicorn app.main:app --reload` is still running.
- Check that the API is available at `http://127.0.0.1:8000/health`.
- Set `AQI_API_URL` if your API runs on another host or port.

If the collector cannot fetch weather:

- Check internet access.
- Re-run `python app/data_collector.py`.
- The collector logs failures per station and keeps running on the next scheduled cycle.
