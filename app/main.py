from __future__ import annotations

import logging
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DB_PATH = PROJECT_ROOT / "aqi_data.db"
PANEL_PATH = PROJECT_ROOT / "data" / "processed" / "california_aqi_model_ready.csv"
MODEL_PATHS = {
    1: PROJECT_ROOT / "models" / "lightgbm_nowcast.txt",
    24: PROJECT_ROOT / "models" / "lightgbm_forecast24h.txt",
}

FORECAST_CONFIGS = {
    1: {
        "label": "Short-Term Nowcasting",
        "model_horizon": "t+1",
        "local_lags": [1, 2, 3],
        "spatial_lags": [1, 2, 3],
        "rolling_shift": 1,
        "rolling_windows": [3],
        "mae": 6.07,
    },
    24: {
        "label": "Long-Term 24-Hour Forecasting",
        "model_horizon": "t+24",
        "local_lags": [24, 48, 72],
        "spatial_lags": [24, 48, 72],
        "rolling_shift": 24,
        "rolling_windows": [24, 72],
        "mae": 13.55,
    },
}

STATIONS: dict[str, dict[str, Any]] = {
    "Fresno": {
        "station_id": "FRES_OPENMETEO",
        "lat": 36.7378,
        "lon": -119.7871,
        "aliases": {"fresno", "fresno - open-meteo", "fresno - garland", "fres"},
        "history_station_ids": ["FRES_OPENMETEO", "FRES"],
    },
    "Los Angeles": {
        "station_id": "LA_OPENMETEO",
        "lat": 34.0522,
        "lon": -118.2437,
        "aliases": {"los angeles", "la", "los angeles - open-meteo", "los angeles - n. main"},
        "history_station_ids": ["LA_OPENMETEO", "LA"],
    },
    "San Jose": {
        "station_id": "SJ_OPENMETEO",
        "lat": 37.3394,
        "lon": -121.8950,
        "aliases": {"san jose", "sj", "downtown san jose", "downtown san jose, california"},
        "history_station_ids": ["SJ_OPENMETEO"],
    },
}

KNOWN_HISTORY_STATION_IDS = ["FRES", "LA", "FRES_OPENMETEO", "LA_OPENMETEO", "SJ_OPENMETEO"]
TARGET = "target_aqi"
POLLUTANT_COLUMNS = {"pm25_ug_m3", "pm10_ug_m3", "pm25", "pm10"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("aqi-fastapi")

app = FastAPI(
    title="California AQI Forecasting API",
    version="1.0.0",
    description="Leakage-controlled LightGBM nowcasting and 24-hour AQI forecasting service.",
)


class PredictionRequest(BaseModel):
    station_name: str = Field(..., examples=["Fresno", "Los Angeles", "San Jose"])
    target_hour_ahead: int = Field(..., description="Supported values are 1 and 24.")
    temperature_2m: float
    relative_humidity_2m: float
    wind_speed_10m: float
    wind_direction_10m: float
    surface_pressure: float
    rain: float
    cloud_cover: float
    observed_at: datetime | None = Field(
        default=None,
        description="UTC timestamp for the current meteorological row. Defaults to the current UTC hour.",
    )


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float
    mae_baseline: float


class PredictionResponse(BaseModel):
    predicted_aqi: float
    model_horizon: str
    confidence_interval: ConfidenceInterval


@contextmanager
def sqlite_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_database() -> None:
    with sqlite_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meteorology_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'openmeteo_realtime',
                station_id TEXT NOT NULL,
                station_name TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                temperature_2m REAL,
                relative_humidity_2m REAL,
                wind_speed_10m REAL,
                wind_direction_10m REAL,
                surface_pressure REAL,
                rain REAL,
                cloud_cover REAL,
                vpd_kpa REAL,
                hour INTEGER,
                dayofweek INTEGER,
                month INTEGER,
                year INTEGER,
                hour_sin REAL,
                hour_cos REAL,
                month_sin REAL,
                month_cos REAL,
                is_weekend INTEGER,
                wind_u REAL,
                wind_v REAL,
                target_aqi REAL,
                pm25_ug_m3 REAL,
                pm10_ug_m3 REAL,
                created_at TEXT NOT NULL,
                UNIQUE(station_id, time)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_meteorology_history_station_time
            ON meteorology_history(station_id, time)
            """
        )


def model_dump_compat(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def vapor_pressure_deficit_kpa(temperature_c: float, relative_humidity_pct: float) -> float:
    saturation_vapor_pressure = 0.6108 * math.exp((17.27 * temperature_c) / (temperature_c + 237.3))
    actual_vapor_pressure = (relative_humidity_pct / 100.0) * saturation_vapor_pressure
    return max(saturation_vapor_pressure - actual_vapor_pressure, 0.0)


def wind_components(speed_mps: float, direction_degrees: float) -> tuple[float, float]:
    radians = math.radians(direction_degrees)
    return speed_mps * math.cos(radians), speed_mps * math.sin(radians)


def safe_station_name(value: str) -> str:
    return (
        str(value)
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "")
        .replace(",", "")
    )


def normalize_reference_time(observed_at: datetime | None) -> datetime:
    if observed_at is None:
        return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).replace(tzinfo=None)
    if observed_at.tzinfo is None:
        return observed_at.replace(minute=0, second=0, microsecond=0)
    return observed_at.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0).replace(tzinfo=None)


def canonical_station(station_name: str) -> tuple[str, dict[str, Any]]:
    normalized = station_name.strip().lower()
    for canonical_name, station in STATIONS.items():
        if normalized == canonical_name.lower() or normalized in station["aliases"]:
            return canonical_name, station
    allowed = ", ".join(STATIONS)
    raise HTTPException(status_code=400, detail=f"Unknown station_name '{station_name}'. Use one of: {allowed}.")


def enrich_current_weather(request: PredictionRequest, station: dict[str, Any], reference_time: datetime) -> dict[str, Any]:
    payload = model_dump_compat(request)
    wind_u, wind_v = wind_components(payload["wind_speed_10m"], payload["wind_direction_10m"])
    hour = reference_time.hour
    month = reference_time.month
    relative_humidity = payload["relative_humidity_2m"]
    temperature = payload["temperature_2m"]
    wind_speed = payload["wind_speed_10m"]
    surface_pressure = payload["surface_pressure"]
    eps = 1e-6

    return {
        "lat": station["lat"],
        "lon": station["lon"],
        "temperature_2m": temperature,
        "relative_humidity_2m": relative_humidity,
        "rain": payload["rain"],
        "cloud_cover": payload["cloud_cover"],
        "wind_speed_10m": wind_speed,
        "wind_direction_10m": payload["wind_direction_10m"],
        "surface_pressure": surface_pressure,
        "hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "month_sin": math.sin(2 * math.pi * month / 12.0),
        "month_cos": math.cos(2 * math.pi * month / 12.0),
        "is_weekend": int(reference_time.weekday() in (5, 6)),
        "wind_u": wind_u,
        "wind_v": wind_v,
        "temp_humidity_ratio": temperature / (relative_humidity + eps),
        "dry_heat_index": temperature * (1 - relative_humidity / 100.0),
        "wind_speed_pressure": wind_speed / (surface_pressure + eps),
        "dry_wind_index": wind_speed * (1 - relative_humidity / 100.0),
        "vpd_kpa": vapor_pressure_deficit_kpa(temperature, relative_humidity),
    }


def bootstrap_history_from_panel_if_needed() -> None:
    with sqlite_connection() as conn:
        existing_rows = conn.execute(
            "SELECT COUNT(*) AS n FROM meteorology_history WHERE target_aqi IS NOT NULL"
        ).fetchone()["n"]
    if existing_rows > 0 or not PANEL_PATH.exists():
        return

    logger.info("Bootstrapping AQI lag history from %s", PANEL_PATH)
    columns = [
        "time",
        "source",
        "station_id",
        "station_name",
        "lat",
        "lon",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "surface_pressure",
        "rain",
        "cloud_cover",
        "target_aqi",
        "pm25_ug_m3",
        "pm10_ug_m3",
    ]
    insert_columns = [
        "time",
        "source",
        "station_id",
        "station_name",
        "lat",
        "lon",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "surface_pressure",
        "rain",
        "cloud_cover",
        "vpd_kpa",
        "hour",
        "dayofweek",
        "month",
        "year",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "wind_u",
        "wind_v",
        "target_aqi",
        "pm25_ug_m3",
        "pm10_ug_m3",
        "created_at",
    ]
    placeholders = ", ".join("?" for _ in insert_columns)
    sql = f"""
        INSERT OR IGNORE INTO meteorology_history ({", ".join(insert_columns)})
        VALUES ({placeholders})
    """

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    total_inserted = 0
    for chunk in pd.read_csv(PANEL_PATH, usecols=lambda column: column in columns, chunksize=10_000):
        chunk["time"] = pd.to_datetime(chunk["time"], errors="coerce")
        chunk = chunk.dropna(subset=["time", "station_id", TARGET])
        chunk["vpd_kpa"] = [
            vapor_pressure_deficit_kpa(float(temp), float(rh))
            for temp, rh in zip(chunk["temperature_2m"], chunk["relative_humidity_2m"], strict=False)
        ]
        radians = np.deg2rad(chunk["wind_direction_10m"].astype(float))
        chunk["wind_u"] = chunk["wind_speed_10m"].astype(float) * np.cos(radians)
        chunk["wind_v"] = chunk["wind_speed_10m"].astype(float) * np.sin(radians)
        chunk["hour"] = chunk["time"].dt.hour
        chunk["dayofweek"] = chunk["time"].dt.dayofweek
        chunk["month"] = chunk["time"].dt.month
        chunk["year"] = chunk["time"].dt.year
        chunk["hour_sin"] = np.sin(2 * np.pi * chunk["hour"] / 24.0)
        chunk["hour_cos"] = np.cos(2 * np.pi * chunk["hour"] / 24.0)
        chunk["month_sin"] = np.sin(2 * np.pi * chunk["month"] / 12.0)
        chunk["month_cos"] = np.cos(2 * np.pi * chunk["month"] / 12.0)
        chunk["is_weekend"] = chunk["dayofweek"].isin([5, 6]).astype(int)
        chunk["created_at"] = created_at
        chunk["time"] = chunk["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

        records_frame = chunk.reindex(columns=insert_columns)
        records = records_frame.where(pd.notna(records_frame), None).values.tolist()
        with sqlite_connection() as conn:
            conn.executemany(sql, records)
        total_inserted += len(records)
    logger.info("Bootstrapped %s candidate history rows", total_inserted)


def read_history(reference_time: datetime, horizon: int) -> pd.DataFrame:
    max_lag = max(FORECAST_CONFIGS[horizon]["local_lags"] + FORECAST_CONFIGS[horizon]["spatial_lags"])
    rolling_context = FORECAST_CONFIGS[horizon]["rolling_shift"] + max(FORECAST_CONFIGS[horizon]["rolling_windows"])
    lookback_hours = max(max_lag, rolling_context, 96) + 24
    start_time = reference_time - timedelta(hours=lookback_hours)
    with sqlite_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM meteorology_history
            WHERE target_aqi IS NOT NULL
              AND time <= ?
              AND time >= ?
            ORDER BY time ASC
            """,
            (
                reference_time.strftime("%Y-%m-%d %H:%M:%S"),
                start_time.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        ).fetchall()
    history = pd.DataFrame([dict(row) for row in rows])
    if history.empty:
        with sqlite_connection() as conn:
            max_time_row = conn.execute(
                "SELECT MAX(time) AS max_time FROM meteorology_history WHERE target_aqi IS NOT NULL"
            ).fetchone()
            max_time_value = max_time_row["max_time"] if max_time_row else None
            if max_time_value:
                max_time = pd.to_datetime(max_time_value).to_pydatetime()
                fallback_start = max_time - timedelta(hours=lookback_hours)
                rows = conn.execute(
                    """
                    SELECT *
                    FROM meteorology_history
                    WHERE target_aqi IS NOT NULL
                      AND time <= ?
                      AND time >= ?
                    ORDER BY time ASC
                    """,
                    (
                        max_time.strftime("%Y-%m-%d %H:%M:%S"),
                        fallback_start.strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                ).fetchall()
                history = pd.DataFrame([dict(row) for row in rows])
    if history.empty:
        return history
    history["time"] = pd.to_datetime(history["time"], errors="coerce")
    history[TARGET] = pd.to_numeric(history[TARGET], errors="coerce")
    return history.dropna(subset=["time", TARGET]).sort_values(["station_id", "time"])


def lag_value(
    history: pd.DataFrame,
    station_ids: list[str],
    reference_time: datetime,
    lag_hours: int,
    strict_not_after: datetime | None = None,
) -> float:
    if history.empty:
        return float("nan")
    target_time = reference_time - timedelta(hours=lag_hours)
    upper_bound = min(target_time, strict_not_after) if strict_not_after else target_time
    subset = history[
        history["station_id"].isin(station_ids)
        & (history["time"] <= pd.Timestamp(upper_bound))
    ].sort_values("time")
    if subset.empty:
        return float("nan")

    exact = subset[subset["time"] == pd.Timestamp(target_time)]
    value = exact.iloc[-1][TARGET] if not exact.empty else subset.iloc[-1][TARGET]
    return float(value)


def rolling_stats(
    history: pd.DataFrame,
    station_ids: list[str],
    reference_time: datetime,
    shift_hours: int,
    window_hours: int,
) -> tuple[float, float]:
    if history.empty:
        return float("nan"), float("nan")
    end_time = reference_time - timedelta(hours=shift_hours)
    start_time = end_time - timedelta(hours=window_hours - 1)
    subset = history[
        history["station_id"].isin(station_ids)
        & (history["time"] >= pd.Timestamp(start_time))
        & (history["time"] <= pd.Timestamp(end_time))
    ][TARGET].astype(float)
    if subset.empty:
        return float("nan"), float("nan")
    return float(subset.mean()), float(subset.std(ddof=1) if len(subset) > 1 else 0.0)


def build_lag_features(
    history: pd.DataFrame,
    station: dict[str, Any],
    reference_time: datetime,
    horizon: int,
) -> dict[str, float]:
    config = FORECAST_CONFIGS[horizon]
    features: dict[str, float] = {}
    strict_not_after = reference_time - timedelta(hours=24) if horizon == 24 else None

    for lag in config["local_lags"]:
        features[f"{TARGET}_lag_{lag}"] = lag_value(
            history,
            station["history_station_ids"],
            reference_time,
            lag,
            strict_not_after=strict_not_after,
        )

    for window in config["rolling_windows"]:
        mean_value, std_value = rolling_stats(
            history,
            station["history_station_ids"],
            reference_time,
            config["rolling_shift"],
            window,
        )
        features[f"{TARGET}_roll_mean_from_lag{config['rolling_shift']}_{window}"] = mean_value
        features[f"{TARGET}_roll_std_from_lag{config['rolling_shift']}_{window}"] = std_value

    for lag in config["spatial_lags"]:
        lag_values = []
        for station_id in KNOWN_HISTORY_STATION_IDS:
            value = lag_value(
                history,
                [station_id],
                reference_time,
                lag,
                strict_not_after=strict_not_after,
            )
            features[f"spatial_{TARGET}_{safe_station_name(station_id)}_lag_{lag}"] = value
            if not np.isnan(value):
                lag_values.append(value)
        spatial_mean = float(np.mean(lag_values)) if lag_values else float("nan")
        for station_id in KNOWN_HISTORY_STATION_IDS:
            feature_name = f"spatial_{TARGET}_{safe_station_name(station_id)}_lag_{lag}"
            if np.isnan(features[feature_name]):
                features[feature_name] = spatial_mean

    return features


def build_feature_vector(request: PredictionRequest, station: dict[str, Any], reference_time: datetime) -> dict[str, Any]:
    history = read_history(reference_time, request.target_hour_ahead)
    current_features = enrich_current_weather(request, station, reference_time)
    lag_features = build_lag_features(history, station, reference_time, request.target_hour_ahead)
    feature_vector = {
        **current_features,
        **lag_features,
        "source": "openmeteo_realtime",
        "station_id": station["station_id"],
    }
    return {key: value for key, value in feature_vector.items() if key not in POLLUTANT_COLUMNS}


class LightGBMModel:
    def __init__(self, horizon: int, model_path: Path):
        self.horizon = horizon
        self.model_path = model_path
        self.booster = None
        self.feature_names: list[str] = []
        self.load()

    def load(self) -> None:
        if not self.model_path.exists():
            logger.warning("Model file not found at %s. Falling back to deterministic baseline.", self.model_path)
            return
        try:
            import lightgbm as lgb

            self.booster = lgb.Booster(model_file=str(self.model_path))
            self.feature_names = list(self.booster.feature_name())
            logger.info("Loaded LightGBM model for horizon %s from %s", self.horizon, self.model_path)
        except Exception as exc:
            logger.exception("Could not load LightGBM model %s: %s", self.model_path, exc)
            self.booster = None
            self.feature_names = []

    def align_features(self, features: dict[str, Any]) -> pd.DataFrame:
        if not self.feature_names:
            numeric = {k: v for k, v in features.items() if k not in {"source", "station_id"}}
            return pd.DataFrame([numeric])

        aligned: dict[str, float] = {}
        source_value = str(features.get("source", ""))
        station_id_value = str(features.get("station_id", ""))
        for name in self.feature_names:
            if name in features:
                aligned[name] = features[name]
            elif name.startswith("source_"):
                aligned[name] = float(name == f"source_{source_value}")
            elif name.startswith("station_id_"):
                aligned[name] = float(name == f"station_id_{station_id_value}")
            else:
                aligned[name] = float("nan")
        return pd.DataFrame([aligned], columns=self.feature_names)

    def fallback_predict(self, features: dict[str, Any]) -> float:
        config = FORECAST_CONFIGS[self.horizon]
        local_values = [
            float(features.get(f"{TARGET}_lag_{lag}", np.nan))
            for lag in config["local_lags"]
            if not np.isnan(float(features.get(f"{TARGET}_lag_{lag}", np.nan)))
        ]
        spatial_values = [
            float(value)
            for key, value in features.items()
            if key.startswith(f"spatial_{TARGET}_") and not np.isnan(float(value))
        ]
        local_baseline = float(np.mean(local_values)) if local_values else np.nan
        spatial_baseline = float(np.mean(spatial_values)) if spatial_values else np.nan

        if np.isnan(local_baseline) and np.isnan(spatial_baseline):
            baseline = 50.0
        elif np.isnan(local_baseline):
            baseline = spatial_baseline
        elif np.isnan(spatial_baseline):
            baseline = local_baseline
        else:
            baseline = 0.75 * local_baseline + 0.25 * spatial_baseline

        rh = float(features["relative_humidity_2m"])
        wind = float(features["wind_speed_10m"])
        rain = float(features["rain"])
        vpd = float(features["vpd_kpa"])
        stagnation_boost = 8.0 if wind < 6.07 and rh < 55.0 else 0.0
        dryness_boost = min(vpd * 2.5, 8.0)
        rain_reduction = min(max(rain, 0.0) * 4.0, 15.0)
        horizon_shrinkage = 0.92 if self.horizon == 1 else 0.78
        prediction = horizon_shrinkage * baseline + stagnation_boost + dryness_boost - rain_reduction
        return float(np.clip(prediction, 0.0, 500.0))

    def predict(self, features: dict[str, Any]) -> float:
        if self.booster is None:
            return self.fallback_predict(features)
        matrix = self.align_features(features)
        prediction = float(self.booster.predict(matrix)[0])
        return float(np.clip(prediction, 0.0, 500.0))


MODELS: dict[int, LightGBMModel] = {}


@app.on_event("startup")
def startup() -> None:
    initialize_database()
    bootstrap_history_from_panel_if_needed()
    for horizon, model_path in MODEL_PATHS.items():
        MODELS[horizon] = LightGBMModel(horizon=horizon, model_path=model_path)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "database": str(DB_PATH),
        "loaded_model_horizons": sorted(MODELS),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if request.target_hour_ahead not in FORECAST_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail="target_hour_ahead must be exactly 1 for nowcasting or 24 for strict 24-hour forecasting.",
        )

    try:
        _, station = canonical_station(request.station_name)
        reference_time = normalize_reference_time(request.observed_at)
        features = build_feature_vector(request, station, reference_time)
        model = MODELS.get(request.target_hour_ahead) or LightGBMModel(
            horizon=request.target_hour_ahead,
            model_path=MODEL_PATHS[request.target_hour_ahead],
        )
        predicted_aqi = round(model.predict(features), 2)
        mae = FORECAST_CONFIGS[request.target_hour_ahead]["mae"]
        margin = 1.96 * mae
        confidence_interval = ConfidenceInterval(
            lower=round(max(predicted_aqi - margin, 0.0), 2),
            upper=round(min(predicted_aqi + margin, 500.0), 2),
            mae_baseline=mae,
        )
        return PredictionResponse(
            predicted_aqi=predicted_aqi,
            model_horizon=f"{FORECAST_CONFIGS[request.target_hour_ahead]['label']} ({FORECAST_CONFIGS[request.target_hour_ahead]['model_horizon']})",
            confidence_interval=confidence_interval,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
