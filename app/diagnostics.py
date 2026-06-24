from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "aqi_data.db"
DEFAULT_PREDICTION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "california_aqi_model_predictions.csv"
)

MODEL_NAMES = (
    "LightGBM",
    "XGBoost",
    "CatBoost",
    "Random Forest",
    "Linear Ridge",
)
SCENARIOS = {
    "Short-term Nowcasting (1h)": {
        "configuration": "Short-term Autoregressive (Lag 1-3h)",
        "horizon_hours": 1,
    },
    "Long-term Forecasting (24h)": {
        "configuration": "Long-term Forecasting (Lag 24h)",
        "horizon_hours": 24,
    },
}
QUICK_RANGES = ("24 Hours", "7 Days", "30 Days", "Full Custom Range")


@dataclass(frozen=True)
class AQICategory:
    name: str
    color: str


AQI_CATEGORIES = (
    (50.0, AQICategory("Good", "#16a34a")),
    (100.0, AQICategory("Moderate", "#eab308")),
    (150.0, AQICategory("Unhealthy for Sensitive Groups", "#f97316")),
    (200.0, AQICategory("Unhealthy", "#dc2626")),
    (300.0, AQICategory("Very Unhealthy", "#7e22ce")),
    (math.inf, AQICategory("Hazardous", "#7f1d1d")),
)


def classify_aqi(value: float) -> AQICategory:
    normalized = max(float(value), 0.0)
    for upper_bound, category in AQI_CATEGORIES:
        if normalized <= upper_bound:
            return category
    raise RuntimeError("AQI category lookup failed.")


def resolve_historical_window(
    start_date: date,
    end_date: date,
    quick_range: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if quick_range not in QUICK_RANGES:
        raise ValueError(f"Unsupported historical range: {quick_range}")

    selected_start = pd.Timestamp(start_date).normalize()
    selected_end = (
        pd.Timestamp(end_date).normalize()
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )
    if selected_start > selected_end:
        raise ValueError("Historical start date must not be after end date.")

    if quick_range == "24 Hours":
        return selected_end - pd.Timedelta(hours=23, minutes=59, seconds=59), selected_end
    if quick_range == "7 Days":
        return selected_end - pd.Timedelta(days=7) + pd.Timedelta(seconds=1), selected_end
    if quick_range == "30 Days":
        return selected_end - pd.Timedelta(days=30) + pd.Timedelta(seconds=1), selected_end
    return selected_start, selected_end


def calculate_model_metrics(aligned: pd.DataFrame) -> pd.DataFrame:
    required = {"model_name", "actual_aqi", "predicted_aqi"}
    missing = required.difference(aligned.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")

    rows: list[dict[str, float | int | str]] = []
    valid = aligned.dropna(subset=["model_name", "actual_aqi", "predicted_aqi"])
    for model_name, model_rows in valid.groupby("model_name", sort=False):
        actual = model_rows["actual_aqi"].astype(float).to_numpy()
        predicted = model_rows["predicted_aqi"].astype(float).to_numpy()
        denominator = float(np.abs(actual).sum())
        wmape = (
            float(np.abs(actual - predicted).sum() / denominator)
            if denominator
            else np.nan
        )
        rows.append(
            {
                "model_name": model_name,
                "n": len(model_rows),
                "mae": float(mean_absolute_error(actual, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                "r2": float(r2_score(actual, predicted))
                if len(actual) >= 2
                else np.nan,
                "relative_accuracy": max(0.0, 1.0 - wmape) * 100.0
                if not np.isnan(wmape)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
