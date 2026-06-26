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
DEFAULT_WILDFIRE_EVENT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "california_wildfire_events.csv"
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


@dataclass(frozen=True)
class WildfireEvent:
    event_name: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    affected_cities: tuple[str, ...]
    severity: str
    context: str


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


def _parse_affected_cities(value: object) -> tuple[str, ...]:
    if pd.isna(value):
        return ()
    return tuple(
        city.strip()
        for city in str(value).replace(",", ";").split(";")
        if city.strip()
    )


def load_wildfire_events(
    event_path: Path = DEFAULT_WILDFIRE_EVENT_PATH,
    *,
    city_name: str,
    start_at: pd.Timestamp,
    end_at: pd.Timestamp,
) -> list[WildfireEvent]:
    path = Path(event_path)
    if not path.exists():
        return []

    frame = pd.read_csv(path)
    required = {
        "event_name",
        "start_time",
        "end_time",
        "affected_cities",
        "severity",
        "context",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Wildfire event file is missing columns: {sorted(missing)}")

    events: list[WildfireEvent] = []
    selected_start = pd.Timestamp(start_at)
    selected_end = pd.Timestamp(end_at)
    for row in frame.itertuples(index=False):
        event_start = pd.Timestamp(row.start_time)
        event_end = (
            pd.Timestamp(row.end_time).normalize()
            + pd.Timedelta(days=1)
            - pd.Timedelta(seconds=1)
        )
        affected_cities = _parse_affected_cities(row.affected_cities)
        if city_name not in affected_cities and "All" not in affected_cities:
            continue
        if event_end < selected_start or event_start > selected_end:
            continue
        events.append(
            WildfireEvent(
                event_name=str(row.event_name),
                start_time=max(event_start, selected_start),
                end_time=min(event_end, selected_end),
                affected_cities=affected_cities,
                severity=str(row.severity),
                context=str(row.context),
            )
        )
    return events


def list_wildfire_events(
    event_path: Path = DEFAULT_WILDFIRE_EVENT_PATH,
    *,
    city_name: str,
) -> list[WildfireEvent]:
    return load_wildfire_events(
        event_path,
        city_name=city_name,
        start_at=pd.Timestamp.min,
        end_at=pd.Timestamp.max,
    )


def list_available_wildfire_events(
    *,
    city_name: str,
    db_path: Path = DEFAULT_DB_PATH,
    event_path: Path = DEFAULT_WILDFIRE_EVENT_PATH,
) -> list[WildfireEvent]:
    events = list_wildfire_events(event_path, city_name=city_name)
    if not events:
        return []

    station_ids = _history_station_ids(city_name)
    station_placeholders = ",".join("?" for _ in station_ids)
    available: list[WildfireEvent] = []
    with sqlite_connection(db_path) as connection:
        for event in events:
            count = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM meteorology_history
                WHERE station_id IN ({station_placeholders})
                  AND target_aqi IS NOT NULL
                  AND time BETWEEN ? AND ?
                """,
                (
                    *station_ids,
                    event.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    event.end_time.strftime("%Y-%m-%d %H:%M:%S"),
                ),
            ).fetchone()[0]
            if count:
                available.append(event)
    return available


def detect_aqi_peak_episodes(
    actual: pd.DataFrame,
    *,
    minimum_aqi: float = 100.0,
    percentile: float = 0.95,
    max_episodes: int = 5,
) -> list[dict[str, object]]:
    if actual.empty:
        return []
    required = {"time", "station_id", "actual_aqi"}
    missing = required.difference(actual.columns)
    if missing:
        raise ValueError(f"Missing AQI peak columns: {sorted(missing)}")

    history = (
        actual[list(required)]
        .dropna(subset=["time", "station_id", "actual_aqi"])
        .copy()
    )
    if history.empty:
        return []

    history["time"] = pd.to_datetime(history["time"])
    history["actual_aqi"] = history["actual_aqi"].astype(float)
    quantile_threshold = float(history["actual_aqi"].quantile(percentile))
    threshold = max(float(minimum_aqi), quantile_threshold)
    candidates = history[history["actual_aqi"].ge(threshold)].sort_values(
        ["station_id", "time"]
    )
    if candidates.empty:
        return []

    episodes: list[dict[str, object]] = []
    for station_id, station_rows in candidates.groupby("station_id", sort=False):
        ordered = station_rows.sort_values("time").reset_index(drop=True)
        groups = ordered["time"].diff().gt(pd.Timedelta(hours=1.5)).cumsum()
        for _, episode_rows in ordered.groupby(groups, sort=False):
            peak_index = episode_rows["actual_aqi"].idxmax()
            peak = episode_rows.loc[peak_index]
            episodes.append(
                {
                    "station_id": station_id,
                    "start_time": episode_rows["time"].min(),
                    "end_time": episode_rows["time"].max(),
                    "peak_time": pd.Timestamp(peak["time"]),
                    "peak_aqi": float(peak["actual_aqi"]),
                    "threshold": threshold,
                }
            )

    return sorted(
        episodes,
        key=lambda episode: (
            -float(episode["peak_aqi"]),
            pd.Timestamp(episode["peak_time"]),
        ),
    )[:max_episodes]


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


def ensure_prediction_data(
    prediction_path: Path = DEFAULT_PREDICTION_PATH,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    initialize_prediction_table(db_path)
    with sqlite_connection(db_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM model_predictions"
        ).fetchone()[0]
    if count:
        return 0
    return sync_prediction_csv(prediction_path, db_path)


def _history_station_ids(city_name: str) -> tuple[str, ...]:
    mapping = {
        "Fresno": ("FRES_OPENMETEO", "FRES"),
        "Los Angeles": ("LA_OPENMETEO", "LA"),
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


def _colored_actual_segments(
    actual: pd.DataFrame,
) -> dict[str, dict[str, list[object]]]:
    segments = {
        category.name: {"x": [], "y": []}
        for _, category in AQI_CATEGORIES
    }
    thresholds = (50.0, 100.0, 150.0, 200.0, 300.0)

    for _, station_rows in actual.groupby("station_id", sort=False):
        ordered = station_rows.sort_values("time").reset_index(drop=True)
        for index in range(len(ordered) - 1):
            start = ordered.iloc[index]
            end = ordered.iloc[index + 1]
            start_time = pd.Timestamp(start["time"])
            end_time = pd.Timestamp(end["time"])
            start_aqi = float(start["actual_aqi"])
            end_aqi = float(end["actual_aqi"])

            fractions = [0.0, 1.0]
            if end_aqi != start_aqi:
                lower, upper = sorted((start_aqi, end_aqi))
                for threshold in thresholds:
                    if lower < threshold < upper:
                        fractions.append(
                            (threshold - start_aqi) / (end_aqi - start_aqi)
                        )
            fractions = sorted(fractions)

            for left, right in zip(fractions, fractions[1:]):
                midpoint = (left + right) / 2.0
                category = classify_aqi(
                    start_aqi + (end_aqi - start_aqi) * midpoint
                )
                x_values = segments[category.name]["x"]
                y_values = segments[category.name]["y"]
                x_values.extend(
                    [
                        start_time + (end_time - start_time) * left,
                        start_time + (end_time - start_time) * right,
                        None,
                    ]
                )
                y_values.extend(
                    [
                        start_aqi + (end_aqi - start_aqi) * left,
                        start_aqi + (end_aqi - start_aqi) * right,
                        None,
                    ]
                )
    return segments


def build_alignment_figure(
    aligned: pd.DataFrame,
    selected_models: Iterable[str],
    *,
    city_name: str | None = None,
    show_wildfire_events: bool = False,
    show_detected_peaks: bool = False,
    wildfire_event_path: Path = DEFAULT_WILDFIRE_EVENT_PATH,
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
            mode="markers",
            name="Actual AQI",
            showlegend=False,
            customdata=actual[["category"]],
            marker={"color": "rgba(0,0,0,0)", "size": 10},
            hovertemplate=(
                "Actual AQI: %{y:.2f}<br>"
                "Category: %{customdata[0]}<extra></extra>"
            ),
        )
    )

    band_bounds = (
        (0, 50, "#16a34a"),
        (50, 100, "#eab308"),
        (100, 150, "#f97316"),
        (150, 200, "#dc2626"),
        (200, 300, "#7e22ce"),
        (300, 500, "#7f1d1d"),
    )
    for lower, upper, color in band_bounds:
        figure.add_hrect(
            y0=lower,
            y1=upper,
            fillcolor=color,
            opacity=0.035,
            line_width=0,
            layer="below",
        )

    visible_max = (
        float(actual["actual_aqi"].max())
        if not actual.empty
        else 0.0
    )
    if show_wildfire_events and city_name and not actual.empty:
        events = load_wildfire_events(
            wildfire_event_path,
            city_name=city_name,
            start_at=pd.Timestamp(actual["time"].min()),
            end_at=pd.Timestamp(actual["time"].max()),
        )
        for event in events:
            figure.add_vrect(
                x0=event.start_time,
                x1=event.end_time,
                fillcolor="#ef4444",
                opacity=0.085,
                line_width=0,
                layer="below",
                annotation_text=event.event_name,
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="#991b1b",
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
    colored_segments = _colored_actual_segments(actual)
    for _, category in AQI_CATEGORIES:
        segment = colored_segments[category.name]
        if not segment["x"]:
            continue
        figure.add_trace(
            go.Scatter(
                x=segment["x"],
                y=segment["y"],
                mode="lines",
                name=category_labels[category.name],
                legendgroup="aqi-category",
                showlegend=False,
                line={"color": category.color, "width": 4},
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
        visible_max = max(
            visible_max,
            float(model_rows["predicted_aqi"].dropna().max()),
        )
        figure.add_trace(
            go.Scatter(
                x=model_rows["time"],
                y=model_rows["predicted_aqi"],
                mode="lines",
                name=model_name,
                line={
                    "color": MODEL_COLORS[model_name],
                    "width": 1.25,
                    "dash": "dot",
                },
                opacity=0.58,
                hovertemplate=f"{model_name}: %{{y:.2f}}<extra></extra>",
            )
        )

    if show_detected_peaks and not actual.empty:
        peak_episodes = detect_aqi_peak_episodes(actual)
        if peak_episodes:
            figure.add_trace(
                go.Scatter(
                    x=[episode["peak_time"] for episode in peak_episodes],
                    y=[episode["peak_aqi"] for episode in peak_episodes],
                    mode="markers+text",
                    name="Detected AQI Peak",
                    marker={
                        "color": "#f97316",
                        "size": 10,
                        "symbol": "diamond",
                        "line": {"color": "white", "width": 1},
                    },
                    text=[
                        f'{float(episode["peak_aqi"]):.0f}'
                        for episode in peak_episodes
                    ],
                    textposition="top center",
                    textfont={"color": "#9a3412", "size": 10},
                    hovertemplate=(
                        "Detected peak AQI: %{y:.2f}<br>"
                        "Peak time: %{x}<extra></extra>"
                    ),
                )
            )

    y_axis_max = min(
        500,
        max(80, math.ceil((visible_max * 1.15) / 10.0) * 10),
    )
    figure.update_layout(
        hovermode="x unified",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#111827"},
        height=400,
        margin={"l": 55, "r": 25, "t": 20, "b": 50},
        legend={
            "orientation": "h",
            "y": 1.08,
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
        range=[0, y_axis_max],
    )
    return figure
