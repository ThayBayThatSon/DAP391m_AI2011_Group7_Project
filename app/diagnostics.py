from __future__ import annotations

import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
CONFIGURATION_TO_SCENARIO = {
    details["configuration"]: (label, details["horizon_hours"])
    for label, details in SCENARIOS.items()
}


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
MODEL_COLORS = {
    "LightGBM": "#2563eb",
    "XGBoost": "#f97316",
    "Linear Ridge": "#16a34a",
    "CatBoost": "#7e22ce",
    "Random Forest": "#8b5e3c",
}


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


@contextmanager
def sqlite_connection(
    db_path: Path = DEFAULT_DB_PATH,
) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(Path(db_path))
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_prediction_table(db_path: Path = DEFAULT_DB_PATH) -> None:
    with sqlite_connection(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT NOT NULL,
                station_id TEXT NOT NULL,
                station_name TEXT NOT NULL,
                city_name TEXT NOT NULL,
                scenario TEXT NOT NULL,
                horizon_hours INTEGER NOT NULL CHECK (horizon_hours IN (1, 24)),
                model_name TEXT NOT NULL,
                predicted_aqi REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(time, station_id, scenario, model_name)
            );
            CREATE INDEX IF NOT EXISTS idx_predictions_city_scenario_time
                ON model_predictions(city_name, scenario, time);
            CREATE INDEX IF NOT EXISTS idx_predictions_model_scenario_time
                ON model_predictions(model_name, scenario, time);
            """
        )


def sync_prediction_csv(
    prediction_path: Path = DEFAULT_PREDICTION_PATH,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    frame = pd.read_csv(prediction_path)
    required = {
        "Configuration",
        "Model",
        "time",
        "station_id",
        "station_name",
        "city_name",
        "Predicted_AQI",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction report is missing columns: {sorted(missing)}")
    if not set(frame["Model"]).issubset(MODEL_NAMES):
        raise ValueError("Prediction report contains unsupported model names.")
    if not set(frame["Configuration"]).issubset(CONFIGURATION_TO_SCENARIO):
        raise ValueError("Prediction report contains unsupported configurations.")

    records: list[tuple[object, ...]] = []
    for row in frame.itertuples(index=False):
        scenario, horizon = CONFIGURATION_TO_SCENARIO[row.Configuration]
        records.append(
            (
                pd.Timestamp(row.time).strftime("%Y-%m-%d %H:%M:%S"),
                str(row.station_id),
                str(row.station_name),
                str(row.city_name),
                scenario,
                horizon,
                str(row.Model),
                float(row.Predicted_AQI),
            )
        )

    initialize_prediction_table(db_path)
    with sqlite_connection(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO model_predictions (
                time, station_id, station_name, city_name, scenario,
                horizon_hours, model_name, predicted_aqi
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(time, station_id, scenario, model_name) DO UPDATE SET
                station_name=excluded.station_name,
                city_name=excluded.city_name,
                horizon_hours=excluded.horizon_hours,
                predicted_aqi=excluded.predicted_aqi,
                created_at=CURRENT_TIMESTAMP
            """,
            records,
        )
    return len(records)


def _history_station_ids(city_name: str) -> tuple[str, ...]:
    mapping = {
        "Fresno": ("FRES_OPENMETEO",),
        "Los Angeles": ("LA_OPENMETEO",),
        "San Jose": ("SJ_OPENMETEO",),
    }
    try:
        return mapping[city_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported city: {city_name}") from exc


def load_validation_data(
    city_name: str,
    scenario: str,
    start_at: pd.Timestamp,
    end_at: pd.Timestamp,
    model_names: Iterable[str],
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unsupported scenario: {scenario}")

    selected_models = tuple(model_names)
    unsupported = set(selected_models).difference(MODEL_NAMES)
    if unsupported:
        raise ValueError(f"Unsupported models: {sorted(unsupported)}")

    station_ids = _history_station_ids(city_name)
    station_placeholders = ",".join("?" for _ in station_ids)
    start_text = pd.Timestamp(start_at).strftime("%Y-%m-%d %H:%M:%S")
    end_text = pd.Timestamp(end_at).strftime("%Y-%m-%d %H:%M:%S")

    with sqlite_connection(db_path) as connection:
        actual = pd.read_sql_query(
            f"""
            SELECT time, station_id, target_aqi AS actual_aqi
            FROM meteorology_history
            WHERE station_id IN ({station_placeholders})
              AND time BETWEEN ? AND ?
              AND target_aqi IS NOT NULL
            ORDER BY time, station_id
            """,
            connection,
            params=(*station_ids, start_text, end_text),
            parse_dates=["time"],
        )
        if actual.empty or not selected_models:
            actual["model_name"] = pd.NA
            actual["predicted_aqi"] = np.nan
            return actual

        model_placeholders = ",".join("?" for _ in selected_models)
        predictions = pd.read_sql_query(
            f"""
            SELECT time, station_id, model_name, predicted_aqi
            FROM model_predictions
            WHERE city_name = ?
              AND scenario = ?
              AND model_name IN ({model_placeholders})
              AND time BETWEEN ? AND ?
            ORDER BY time, model_name
            """,
            connection,
            params=(
                city_name,
                scenario,
                *selected_models,
                start_text,
                end_text,
            ),
            parse_dates=["time"],
        )
    return actual.merge(
        predictions,
        on=["time", "station_id"],
        how="left",
    ).sort_values(["time", "model_name"], na_position="last")


def _actual_history(aligned: pd.DataFrame) -> pd.DataFrame:
    return (
        aligned[["time", "station_id", "actual_aqi"]]
        .dropna(subset=["actual_aqi"])
        .drop_duplicates(["time", "station_id"])
        .sort_values(["time", "station_id"])
    )


def build_alignment_figure(
    aligned: pd.DataFrame,
    selected_models: Iterable[str],
) -> go.Figure:
    actual = _actual_history(aligned).copy()
    actual["category"] = actual["actual_aqi"].map(
        lambda value: classify_aqi(value).name
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=actual["time"],
            y=actual["actual_aqi"],
            mode="lines",
            name="Actual AQI",
            customdata=actual[["category"]],
            line={"color": "black", "width": 3, "dash": "dash"},
            hovertemplate=(
                "Actual AQI: %{y:.2f}<br>"
                "Category: %{customdata[0]}<extra></extra>"
            ),
        )
    )

    category_labels = {
        "Good": "Good (0-50)",
        "Moderate": "Moderate (51-100)",
        "Unhealthy for Sensitive Groups": (
            "Unhealthy for Sensitive Groups (101-150)"
        ),
        "Unhealthy": "Unhealthy (151-200)",
        "Very Unhealthy": "Very Unhealthy (201-300)",
        "Hazardous": "Hazardous (301+)",
    }
    for _, category in AQI_CATEGORIES:
        category_mask = actual["category"].eq(category.name)
        if not category_mask.any():
            continue
        figure.add_trace(
            go.Scatter(
                x=actual["time"],
                y=actual["actual_aqi"].where(category_mask, np.nan),
                mode="lines+markers",
                name=category_labels[category.name],
                legendgroup="aqi-category",
                line={"color": category.color, "width": 4},
                marker={"color": category.color, "size": 7},
                hoverinfo="skip",
                connectgaps=False,
            )
        )

    for model_name in tuple(selected_models):
        if model_name not in MODEL_COLORS:
            raise ValueError(f"Unsupported model: {model_name}")
        model_rows = aligned[aligned["model_name"].eq(model_name)].sort_values(
            "time"
        )
        if model_rows.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=model_rows["time"],
                y=model_rows["predicted_aqi"],
                mode="lines",
                name=model_name,
                line={"color": MODEL_COLORS[model_name], "width": 2},
                hovertemplate=f"{model_name}: %{{y:.2f}}<extra></extra>",
            )
        )

    figure.update_layout(
        hovermode="x unified",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#111827"},
        height=460,
        margin={"l": 55, "r": 25, "t": 35, "b": 55},
        legend={
            "orientation": "h",
            "y": 1.16,
            "bgcolor": "rgba(255,255,255,0.92)",
            "font": {"color": "#111827"},
        },
        xaxis_title="Timestamp",
        yaxis_title="AQI",
    )
    figure.update_xaxes(
        showline=True,
        linecolor="#cbd5e1",
        gridcolor="whitesmoke",
        tickfont={"color": "#111827"},
        title_font={"color": "#111827"},
    )
    figure.update_yaxes(
        showline=True,
        linecolor="#cbd5e1",
        gridcolor="whitesmoke",
        tickfont={"color": "#111827"},
        title_font={"color": "#111827"},
    )
    return figure
