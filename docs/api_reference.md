# API Reference and Deployment

This document describes the FastAPI backend interfaces and the Docker deployment setup.

## 1. FastAPI Endpoints

The backend (`app/main.py`) exposes a REST API built with FastAPI. It relies on Pydantic models for strict input validation.

### `POST /predict`
Generates a single AQI forecast for a specific future horizon.

**Request Body (`PredictionRequest`)**:
```json
{
  "station_name": "Fresno",
  "target_hour_ahead": 24,
  "observed_at": "2026-07-19T12:00:00Z",
  "temperature_2m": 35.5,
  "relative_humidity_2m": 20.0,
  "wind_speed_10m": 5.2,
  "wind_direction_10m": 180,
  "surface_pressure": 1012.5,
  "rain": 0.0,
  "cloud_cover": 10
}
```

**Response (`PredictionResponse`)**:
```json
{
  "model_horizon": "Long-term Forecasting (Lag 24h)",
  "predicted_aqi": 85.4,
  "confidence_interval": {
    "lower": 70.1,
    "upper": 100.7
  },
  "shap_values": [
    {"feature": "target_aqi_lag_24", "value": 15.3},
    {"feature": "temperature_2m", "value": 5.2}
  ]
}
```
*Note: `shap_values` represent the absolute contribution of each feature to the model's base expected value, moving the final prediction up or down.*

### `POST /predict_timeline`
Generates a continuous sequence of predictions for multiple horizons (1h, 6h, 12h, 18h, 24h) to power trend charts.

**Request Body**: Same as `/predict` (the `target_hour_ahead` field is ignored).

**Response (`TimelinePredictionResponse`)**:
```json
{
  "predictions": {
    "1": 45.2,
    "6": 60.1,
    "12": 85.4,
    "18": 75.0,
    "24": 65.2
  }
}
```

## 2. Docker Deployment

The system is fully containerized using Docker and Docker Compose, ensuring identical execution environments across development and production.

### Services (`docker-compose.yml`)
- **`backend`**: Runs the FastAPI server via Uvicorn on port `8000`. It mounts the `./data` and `./models` directories to access the SQLite database and pre-trained LightGBM artifacts.
- **`frontend`**: Runs the Streamlit dashboard on port `8501`. It is configured with `AQI_API_URL=http://backend:8000` to seamlessly route prediction requests to the backend service over the internal Docker network.

### Execution
To build and start the entire stack:
```bash
docker-compose up --build
```
The dashboard will be available at `http://localhost:8501`.
