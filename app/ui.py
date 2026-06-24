from __future__ import annotations

import math
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.diagnostics import (
    DEFAULT_DB_PATH,
    MODEL_NAMES,
    QUICK_RANGES,
    SCENARIOS,
    build_alignment_figure,
    calculate_model_metrics,
    initialize_prediction_table,
    load_validation_data,
    resolve_historical_window,
)

API_BASE_URL = os.getenv("AQI_API_URL", "").strip()
USE_REMOTE_API = bool(API_BASE_URL)
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 30
PREDICTION_RETRY_ATTEMPTS = 4
PREDICTION_RETRY_STATUS_CODES = {429, 502, 503, 504}
PREDICTION_API_SESSION = requests.Session()
PREDICTION_API_SESSION.trust_env = False

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "rain",
    "cloud_cover",
]

STATIONS: dict[str, dict[str, float]] = {
    "Fresno": {"lat": 36.7378, "lon": -119.7871},
    "Los Angeles": {"lat": 34.0522, "lon": -118.2437},
    "San Jose": {"lat": 37.3394, "lon": -121.8950},
}
EMPTY_HISTORY_WARNING = (
    "No historical records were found for the selected period. "
    "Please update the filters."
)


@st.cache_resource(show_spinner=False)
def load_local_prediction_engine():
    from app import main as prediction_engine

    prediction_engine.startup()
    return prediction_engine


PREDICTION_ENGINE = None if USE_REMOTE_API else load_local_prediction_engine()


def parse_open_meteo_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def select_latest_hourly_record(payload: dict[str, Any]) -> tuple[datetime, dict[str, float]]:
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        raise ValueError("Open-Meteo response did not contain hourly timestamps.")
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    parsed_times = [parse_open_meteo_time(value) for value in times]
    eligible_indices = [idx for idx, ts in enumerate(parsed_times) if ts <= now_utc]
    selected_index = eligible_indices[-1] if eligible_indices else 0

    values: dict[str, float] = {}
    for variable in WEATHER_VARIABLES:
        series = hourly.get(variable)
        if series is None or selected_index >= len(series):
            raise ValueError(f"Open-Meteo response is missing hourly variable: {variable}")
        values[variable] = float(series[selected_index])
    return parsed_times[selected_index], values


@st.cache_data(ttl=900, show_spinner=False)
def fetch_realtime_weather(station_name: str) -> tuple[datetime, dict[str, float]]:
    station = STATIONS[station_name]
    params = {
        "latitude": station["lat"],
        "longitude": station["lon"],
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "GMT",
        "past_days": 1,
        "forecast_days": 1,
        "wind_speed_unit": "ms",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return select_latest_hourly_record(response.json())


def call_prediction_api(station_name: str, horizon: int, observed_at: datetime, weather: dict[str, float]) -> dict[str, Any]:
    payload = {
        "station_name": station_name,
        "target_hour_ahead": horizon,
        "observed_at": observed_at.isoformat(),
        **weather,
    }

    if not USE_REMOTE_API:
        if PREDICTION_ENGINE is None:
            raise RuntimeError("Local prediction engine is not initialized.")
        request = PREDICTION_ENGINE.PredictionRequest(**payload)
        response = PREDICTION_ENGINE.predict(request)
        if hasattr(response, "model_dump"):
            return response.model_dump()
        return response.dict()

    last_error: requests.HTTPError | None = None
    for attempt in range(1, PREDICTION_RETRY_ATTEMPTS + 1):
        response = PREDICTION_API_SESSION.post(
            f"{API_BASE_URL.rstrip('/')}/predict",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        try:
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            last_error = exc
            if response.status_code not in PREDICTION_RETRY_STATUS_CODES or attempt == PREDICTION_RETRY_ATTEMPTS:
                raise
            time.sleep(min(2 ** (attempt - 1), 4))

    if last_error is not None:
        raise last_error
    raise RuntimeError("Prediction API request failed before receiving a response.")


def vpd_kpa(temperature_c: float, relative_humidity_pct: float) -> float:
    saturation_vapor_pressure = 0.6108 * math.exp((17.27 * temperature_c) / (temperature_c + 237.3))
    actual_vapor_pressure = (relative_humidity_pct / 100.0) * saturation_vapor_pressure
    return max(saturation_vapor_pressure - actual_vapor_pressure, 0.0)


def aqi_category(aqi: float) -> tuple[str, str]:
    if aqi <= 50:
        return "Good", "#2ca25f"
    if aqi <= 100:
        return "Moderate", "#f2c94c"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#f2994a"
    if aqi <= 200:
        return "Unhealthy", "#eb5757"
    if aqi <= 300:
        return "Very Unhealthy", "#9b51e0"
    return "Hazardous", "#7f1d1d"


def timeline_frame(prediction: dict[str, Any], horizon: int) -> pd.DataFrame:
    confidence = prediction["confidence_interval"]
    predicted_aqi = float(prediction["predicted_aqi"])
    hours = list(range(0, horizon + 1))
    if horizon == 0:
        horizon = 1
    return pd.DataFrame(
        {
            "Forecast Hour": hours,
            "Predicted AQI": [predicted_aqi for _ in hours],
            "Lower Bound": [
                max(predicted_aqi - (predicted_aqi - confidence["lower"]) * (hour / max(horizon, 1)), 0)
                for hour in hours
            ],
            "Upper Bound": [
                min(predicted_aqi + (confidence["upper"] - predicted_aqi) * (hour / max(horizon, 1)), 500)
                for hour in hours
            ],
        }
    )


def render_timeline_chart(frame: pd.DataFrame, color: str) -> None:
    band = (
        alt.Chart(frame)
        .mark_area(opacity=0.22, color=color)
        .encode(
            x=alt.X("Forecast Hour:Q", title="Hours Ahead", scale=alt.Scale(domain=[0, max(frame["Forecast Hour"].max(), 1)])),
            y=alt.Y("Lower Bound:Q", title="AQI", scale=alt.Scale(domain=[0, max(frame["Upper Bound"].max() + 20, 80)])),
            y2="Upper Bound:Q",
        )
    )
    line = (
        alt.Chart(frame)
        .mark_line(color=color, strokeWidth=3)
        .encode(x="Forecast Hour:Q", y="Predicted AQI:Q")
    )
    points = (
        alt.Chart(frame.tail(1))
        .mark_circle(color=color, size=90)
        .encode(x="Forecast Hour:Q", y="Predicted AQI:Q")
    )
    st.altair_chart((band + line + points).properties(height=360), width="stretch")


def render_live_forecast_content(
    observed_at: datetime,
    weather: dict[str, float],
    prediction: dict[str, Any],
    horizon: int,
    vpd: float,
) -> None:
    if weather["wind_speed_10m"] < 6.07 and weather["relative_humidity_2m"] < 55.0:
        st.markdown(
            """
            <div class="critical-alert">
            CRITICAL AIR STAGNATION WARNING: Meteorological conditions match the
            historical 99th percentile extreme-pollution window
            (Winter Inversion/Wildfire context). Expect rapid ground-level
            PM2.5 accumulation.
            </div>
            """,
            unsafe_allow_html=True,
        )

    category, category_color = aqi_category(float(prediction["predicted_aqi"]))
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Predicted AQI",
        f"{prediction['predicted_aqi']:.2f}",
        category,
    )
    metric_columns[1].metric(
        "Temperature",
        f"{weather['temperature_2m']:.1f} C",
    )
    metric_columns[2].metric(
        "Humidity",
        f"{weather['relative_humidity_2m']:.0f}%",
    )
    metric_columns[3].metric(
        "Wind",
        f"{weather['wind_speed_10m']:.2f} m/s",
    )

    interval = prediction["confidence_interval"]
    st.caption(
        f"{prediction['model_horizon']} | observed "
        f"{observed_at.strftime('%Y-%m-%d %H:%M UTC')} | "
        f"CI {interval['lower']:.2f}-{interval['upper']:.2f} AQI | "
        f"VPD {vpd:.3f} kPa"
    )
    render_timeline_chart(
        timeline_frame(prediction, horizon),
        category_color,
    )


@st.cache_data(ttl=300, show_spinner=False)
def cached_validation_data(
    city_name: str,
    scenario: str,
    start_at: pd.Timestamp,
    end_at: pd.Timestamp,
    model_names: tuple[str, ...],
) -> pd.DataFrame:
    return load_validation_data(
        city_name=city_name,
        scenario=scenario,
        start_at=start_at,
        end_at=end_at,
        model_names=model_names,
        db_path=DEFAULT_DB_PATH,
    )


def render_metric_cards(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        st.info("Select at least one model to display model metrics.")
        return

    ranked = metrics.sort_values(
        "relative_accuracy",
        ascending=False,
        na_position="last",
    )
    columns = st.columns(min(len(ranked), 5))
    for column, row in zip(columns, ranked.itertuples(index=False)):
        accuracy = (
            "N/A"
            if pd.isna(row.relative_accuracy)
            else f"{row.relative_accuracy:.2f}%"
        )
        r2_text = "R² N/A" if pd.isna(row.r2) else f"R² {row.r2:.4f}"
        column.metric(row.model_name, accuracy, r2_text)


def render_diagnostics_tab() -> None:
    with st.expander("Validation Controls", expanded=True):
        city_column, scenario_column = st.columns(2)
        city_name = city_column.selectbox(
            "City",
            list(STATIONS),
            key="validation_city",
        )
        scenario = scenario_column.radio(
            "Forecast Scenario",
            list(SCENARIOS),
            horizontal=True,
            key="validation_scenario",
        )
        date_range = st.date_input(
            "Historical Window",
            value=(date(2025, 11, 1), date(2025, 12, 31)),
            key="validation_dates",
        )
        quick_range = st.segmented_control(
            "Quick Historical Range",
            QUICK_RANGES,
            default="30 Days",
            key="validation_quick_range",
        )
        selected_models = st.multiselect(
            "Visible Models",
            MODEL_NAMES,
            default=["LightGBM", "Linear Ridge"],
            key="validation_models",
        )

    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.warning("Select both a start date and an end date.")
        return

    start_at, end_at = resolve_historical_window(
        date_range[0],
        date_range[1],
        quick_range,
    )
    try:
        initialize_prediction_table(DEFAULT_DB_PATH)
        aligned = cached_validation_data(
            city_name,
            scenario,
            start_at,
            end_at,
            tuple(selected_models),
        )
    except (OSError, sqlite3.DatabaseError, ValueError) as exc:
        st.error(f"Historical validation is unavailable: {exc}")
        return

    if aligned.empty:
        st.warning(EMPTY_HISTORY_WARNING)
        return

    st.subheader("Historical AQI and Forecast Alignment")
    st.caption(
        "Actual AQI is a thick solid line colored by US AQI health category. "
        "Hover over a timestamp to compare every active model."
    )
    st.plotly_chart(
        build_alignment_figure(aligned, selected_models),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "US AQI zones: Good 0-50 | Moderate 51-100 | "
        "Unhealthy for Sensitive Groups 101-150 | Unhealthy 151-200 | "
        "Very Unhealthy 201-300 | Hazardous 301+"
    )

    st.subheader("Relative Prediction Accuracy")
    metrics = calculate_model_metrics(aligned)
    render_metric_cards(metrics)
    st.info(
        "Relative Prediction Accuracy = max(0, 100% - WMAPE), where WMAPE is "
        "the sum of absolute prediction errors divided by the sum of Actual AQI. "
        "A score of 100% means an exact match. R² measures explained AQI "
        "variance and is not an accuracy percentage."
    )
    if metrics.empty:
        return

    leaderboard = metrics.rename(
        columns={
            "model_name": "Model",
            "relative_accuracy": "Relative Accuracy",
            "mae": "MAE (Mean Absolute Error)",
            "rmse": "RMSE (Root Mean Squared Error)",
            "r2": "R² Score (Explained Variance)",
        }
    )[
        [
            "Model",
            "Relative Accuracy",
            "MAE (Mean Absolute Error)",
            "RMSE (Root Mean Squared Error)",
            "R² Score (Explained Variance)",
        ]
    ]
    st.subheader("Live Metrics Leaderboard")
    st.dataframe(
        leaderboard.style.format(
            {
                "Relative Accuracy": "{:.4f}%",
                "MAE (Mean Absolute Error)": "{:.4f}",
                "RMSE (Root Mean Squared Error)": "{:.4f}",
                "R² Score (Explained Variance)": "{:.4f}",
            },
            na_rep="N/A",
        ),
        width="stretch",
        hide_index=True,
    )


st.set_page_config(page_title="California AQI Forecasting", page_icon="CA", layout="wide")
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1180px;
    }
    .aqi-title {
        font-size: 2.05rem;
        font-weight: 760;
        letter-spacing: 0;
        margin-bottom: 0.15rem;
    }
    .aqi-subtitle {
        color: #536471;
        font-size: 1.02rem;
        margin-bottom: 1rem;
    }
    .metric-strip {
        border: 1px solid #e6edf3;
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        background: #ffffff;
    }
    .critical-alert {
        border-left: 8px solid #b91c1c;
        background: #fff1f2;
        color: #7f1d1d;
        padding: 1rem 1.1rem;
        border-radius: 8px;
        font-weight: 740;
        line-height: 1.45;
        margin: 0.5rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="aqi-title">California AQI Forecasting Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="aqi-subtitle">Fresno, Los Angeles, and San Jose hourly AQI nowcasting with leakage-controlled 24-hour forecasting.</div>',
    unsafe_allow_html=True,
)

live_tab, diagnostics_tab = st.tabs(
    ["Live Forecast", "📊 Live Validation & Model Diagnostics"]
)

with live_tab:
    left_panel, right_panel = st.columns([0.42, 0.58], gap="large")

    with left_panel:
        station_name = st.selectbox("Station", list(STATIONS.keys()), index=0)
        horizon = st.segmented_control(
            "Forecast Horizon",
            options=[1, 24],
            default=24,
            format_func=lambda value: "1 Hour" if value == 1 else "24 Hours",
            key="live_forecast_horizon",
        )
        station_frame = pd.DataFrame(
            [
                {
                    "lat": STATIONS[station_name]["lat"],
                    "lon": STATIONS[station_name]["lon"],
                }
            ]
        )
        st.map(
            station_frame,
            latitude="lat",
            longitude="lon",
            zoom=8,
            width="stretch",
        )

    try:
        with right_panel:
            with st.spinner("Updating forecast..."):
                observed_at, weather = fetch_realtime_weather(station_name)
                vpd = vpd_kpa(
                    weather["temperature_2m"],
                    weather["relative_humidity_2m"],
                )
                prediction = call_prediction_api(
                    station_name,
                    horizon,
                    observed_at,
                    weather,
                )
            render_live_forecast_content(
                observed_at,
                weather,
                prediction,
                horizon,
                vpd,
            )
    except requests.exceptions.ConnectionError:
        with right_panel:
            st.error(
                f"FastAPI backend is not reachable at {API_BASE_URL}. "
                "Start it with: uvicorn app.main:app --reload"
            )
    except Exception as exc:
        with right_panel:
            st.error(f"Dashboard update failed: {exc}")

with diagnostics_tab:
    render_diagnostics_tab()
