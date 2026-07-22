# System Architecture

The **California AQI Forecasting** project is a comprehensive machine learning system designed to predict Air Quality Index (AQI) values for multiple cities in California (Fresno, Los Angeles, and San Jose). The architecture is divided into three primary layers: Data & Modeling Pipeline, Backend Services, and Frontend Dashboard.

## 1. High-Level Architecture

```mermaid
graph TD
    subgraph External Sources
        EPA[EPA AQS API]
        OM[Open-Meteo API]
    end

    subgraph Data & Modeling Pipeline
        DB[(SQLite: aqi_data.db)]
        TrainScript[train_combined_panel_models.py]
        Models[(Model Artifacts)]
        EPA --> DB
        OM --> DB
        DB --> TrainScript
        TrainScript --> Models
    end

    subgraph Backend Services
        FastAPI[FastAPI Backend - main.py]
        FastAPI --> Models
        FastAPI --> DB
    end

    subgraph Frontend Dashboard
        Streamlit[Streamlit UI - ui.py]
        Streamlit --> FastAPI
        OM --> Streamlit
    end
```

## 2. Components Description

### 2.1 Data & Modeling Pipeline
- **Database (`aqi_data.db`)**: A SQLite database that stores historical AQI readings and weather data collected from EPA AQS and Open-Meteo.
- **Training Script (`train_combined_panel_models.py`)**: The core machine learning pipeline. It handles data loading, feature engineering (lags, rolling averages, VPD), and training multiple gradient boosting models (LightGBM, XGBoost, CatBoost) and baselines. It manages various forecast horizons (1-3h, 6h, 12h, 18h, 24h).
- **Model Artifacts**: Trained models are saved as text/JSON files in the `models/` directory alongside their metadata.

### 2.2 Backend Services
- **FastAPI (`main.py`)**: A high-performance REST API that serves predictions.
  - Loads the trained LightGBM models into memory upon startup.
  - Exposes endpoints to predict single horizon AQI (`/predict`) and a 24-hour timeline (`/predict_timeline`).
  - Includes a SHAP TreeExplainer for real-time model interpretability.
  - Queries `aqi_data.db` to fetch historical readings required to compute lag and rolling features on the fly.

### 2.3 Frontend Dashboard
- **Streamlit (`ui.py`)**: An interactive web dashboard.
  - Allows users to select stations and forecast horizons.
  - Fetches real-time weather from Open-Meteo and current AQI from the local database.
  - Calls the FastAPI backend to get predictions.
  - Renders UI elements: 3D PyDeck spatial heatmap, timeline line charts (Plotly), and SHAP feature importance charts.
  - Evaluates weather conditions to issue Air Stagnation Alerts.

## 3. Technology Stack
- **Languages**: Python 3.11+
- **Machine Learning**: LightGBM, XGBoost, CatBoost, Scikit-Learn, SHAP
- **Web Frameworks**: FastAPI (Backend), Streamlit (Frontend)
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Plotly, PyDeck
- **Deployment**: Docker, Docker Compose

## 4. Enterprise Architecture Enhancements (Future Roadmap)

To scale this system for high-concurrency production environments and automate the MLOps lifecycle, the following architectural upgrades are recommended:

### 4.1. Centralized API Gateway & Data Fetching
- **Current**: Streamlit fetches data directly from the Open-Meteo API.
- **Upgrade**: Move all external API calls (Open-Meteo, EPA) to the FastAPI backend. Streamlit should only communicate with FastAPI, hiding API logic from the frontend and allowing centralized data aggregation.

### 4.2. Caching Layer (Redis)
- **Upgrade**: Implement Redis between FastAPI and External APIs. Caching weather forecasts and heavy SHAP calculations for 15-30 minutes will drastically reduce latency, improve User Experience, and prevent hitting external API rate limits.

### 4.3. Model Registry (MLflow / Weights & Biases)
- **Current**: Models are saved as static local files (`.txt`).
- **Upgrade**: Integrate MLflow to version-control model artifacts, track hyperparameters, and monitor MAE/RMSE metrics over time. FastAPI can dynamically pull the best-performing "Production" model from the registry.

### 4.4. Automated Training Pipeline (Apache Airflow / Celery)
- **Current**: `train_combined_panel_models.py` is executed manually.
- **Upgrade**: Use a task scheduler (Celery + RabbitMQ or Apache Airflow) to automatically fetch weekly EPA data, retrain models, and deploy them if their accuracy exceeds the current baseline. This creates a fully automated Continuous Training (CT) loop.

### 4.5. Robust Database (PostgreSQL + TimescaleDB)
- **Current**: SQLite is used for storing historical AQI and weather data.
- **Upgrade**: Migrate to PostgreSQL with the TimescaleDB extension for better concurrency when multiple users access the dashboard simultaneously, and for faster time-series data aggregations.
