# Features and Data Engineering

This document details the data sources and the feature engineering pipeline used to model Air Quality Index (AQI).

## 1. Data Sources

### EPA AQS (Air Quality System)
- **Role**: Primary ground-truth source for historical and high-fidelity AQI readings.
- **Coverage**: Historical data spanning from 2018.
- **Limitation**: Data can have latency, and specific stations might occasionally drop offline or lack extreme real-time coverage during disasters.

### Open-Meteo
- **Role**: Primary source for meteorological features and secondary source for AQI when EPA data is unavailable (using European models).
- **Features Extracted**:
  - `temperature_2m` (Air temperature at 2 meters)
  - `relative_humidity_2m`
  - `wind_speed_10m` & `wind_direction_10m`
  - `surface_pressure`
  - `rain`
  - `cloud_cover`

## 2. Feature Engineering

The `train_combined_panel_models.py` script applies extensive transformations to raw data to create a predictive feature set.

### 2.1 Spatiotemporal Features
- **Temporal Cyclical Encoding**: `hour`, `month`, and `dayofweek` are extracted from the timestamp to capture diurnal and seasonal emission cycles (e.g., rush hour traffic, winter inversions).
- **Spatial Embeddings**: `lat`, `lon`, and categorically encoded `station_id` (via target encoding natively supported by LightGBM/CatBoost) allow the model to learn localized geographic dynamics (e.g., valley accumulation in Fresno vs. coastal clearing in LA).

### 2.2 Autoregressive (Lag) Features
Historical AQI readings are fundamental to predicting future AQI. 
- **Direct Lags**: Depending on the forecast horizon (e.g., 24h), the model is provided with the exact AQI value observed 24, 25, and 26 hours prior. 
- **Leakage Prevention**: The pipeline dynamically shifts the `target_aqi` column by the exact `horizon` amount to ensure the model never sees the "present" AQI when predicting the future.

### 2.3 Rolling Statistics
To smooth out transient noise and capture longer-term accumulation or dispersion trends, rolling metrics are computed on the allowed historical window:
- **Rolling Mean**: 24-hour and 72-hour averages of past AQI.
- **Rolling Standard Deviation**: Measures the volatility of recent AQI (e.g., spiking during a sudden wildfire).
- **Rolling Max/Min**: Captures recent extreme events.

### 2.4 Domain-Specific Interactions
- **Vapor Pressure Deficit (VPD)**: A non-linear combination of temperature and relative humidity. High VPD correlates strongly with dry, hot conditions prone to wildfires and subsequent particulate matter (PM2.5) spikes.
- **Air Stagnation Index (Conceptual)**: High humidity, low wind speed, and lack of rain create atmospheric inversions that trap pollutants. Models implicitly learn this interaction, and the UI surfaces an explicit "Air Stagnation Alert" based on these heuristics.
