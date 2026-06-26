from __future__ import annotations

import html
import math
import os
import sqlite3
import sys
import time
from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.diagnostics import (
    DEFAULT_DB_PATH,
    DEFAULT_PREDICTION_PATH,
    MODEL_NAMES,
    QUICK_RANGES,
    SCENARIOS,
    build_alignment_figure,
    calculate_model_metrics,
    ensure_prediction_data,
    initialize_prediction_table,
    list_available_wildfire_events,
    load_validation_data,
    resolve_historical_window,
)
from app.alerts import evaluate_air_stagnation_alert
from app.current_aqi import AQIReading, resolve_current_aqi
from app.forecast_panel import build_forecast_range_panel

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

STATIONS: dict[str, dict[str, Any]] = {
    "Fresno": {
        "lat": 36.7378,
        "lon": -119.7871,
        "history_station_ids": ("FRES_OPENMETEO", "FRES"),
    },
    "Los Angeles": {
        "lat": 34.0522,
        "lon": -118.2437,
        "history_station_ids": ("LA_OPENMETEO", "LA"),
    },
    "San Jose": {
        "lat": 37.3394,
        "lon": -121.8950,
        "history_station_ids": ("SJ_OPENMETEO",),
    },
}
EMPTY_HISTORY_WARNING = (
    "No historical records were found for the selected period. "
    "Please update the filters."
)
MODEL_CARD_ACCENTS = {
    "LightGBM": "#3b82f6",
    "XGBoost": "#f59e0b",
    "CatBoost": "#a855f7",
    "Random Forest": "#a16207",
    "Linear Ridge": "#14b8a6",
}


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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_current_aqi(station_name: str) -> AQIReading:
    station = STATIONS[station_name]
    return resolve_current_aqi(
        latitude=station["lat"],
        longitude=station["lon"],
        station_ids=station["history_station_ids"],
        db_path=DEFAULT_DB_PATH,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )


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


def metric_card_html(
    label: str,
    value: str,
    detail: str,
    accent: str,
    meta: str = "",
    variant: str = "model",
) -> str:
    meta_markup = (
        f'<div class="metric-meta">{html.escape(meta)}</div>'
        if meta
        else ""
    )
    return (
        f'<div class="metric-card metric-card-{html.escape(variant)}" '
        f'style="--metric-accent:{html.escape(accent)}">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-detail">{html.escape(detail)}</div>'
        f"{meta_markup}"
        "</div>"
    )


def metric_cards_grid_html(
    cards: Sequence[tuple[str, str, str, str, str]],
    variant: str,
) -> str:
    card_markup = "".join(
        metric_card_html(label, value, detail, accent, meta, variant)
        for label, value, detail, accent, meta in cards
    )
    safe_variant = html.escape(variant)
    return (
        f'<div class="metric-card-grid metric-card-grid-{safe_variant}">'
        f"{card_markup}"
        "</div>"
    )


def render_live_forecast_content(
    observed_at: datetime,
    weather: dict[str, float],
    prediction: dict[str, Any],
    current_aqi: AQIReading,
    horizon: int,
    vpd: float,
) -> None:
    alert = evaluate_air_stagnation_alert(
        wind_speed_mps=weather["wind_speed_10m"],
        relative_humidity_pct=weather["relative_humidity_2m"],
        rain_mm=weather["rain"],
        predicted_aqi=float(prediction["predicted_aqi"]),
    )
    if alert is not None:
        st.markdown(
            f"""
            <div class="air-alert air-alert-{alert.level}">
            <strong>{alert.title}</strong><br>
            {alert.message}
            </div>
            """,
            unsafe_allow_html=True,
        )

    category, category_color = aqi_category(float(prediction["predicted_aqi"]))
    if current_aqi.value is None:
        current_value = "N/A"
        current_category = "Live reading unavailable"
        current_color = "#64748b"
    else:
        current_value = f"{current_aqi.value:.0f}"
        current_category, current_color = aqi_category(current_aqi.value)

    current_timestamp = (
        current_aqi.observed_at.strftime("%Y-%m-%d %H:%M UTC")
        if current_aqi.observed_at is not None
        else "No timestamp"
    )
    current_meta = f"{current_aqi.source} | {current_timestamp}"

    predicted_detail = category
    if current_aqi.is_current and current_aqi.value is not None:
        difference = float(prediction["predicted_aqi"]) - current_aqi.value
        predicted_detail = f"{category} | {difference:+.1f} vs current"

    live_metrics = (
        (
            current_aqi.label,
            current_value,
            current_category,
            current_color,
            current_meta,
        ),
        (
            "Predicted AQI",
            f"{prediction['predicted_aqi']:.2f}",
            predicted_detail,
            category_color,
            prediction["model_horizon"],
        ),
        (
            "Temperature",
            f"{weather['temperature_2m']:.1f} C",
            "Air temperature",
            "#f97316",
            "Open-Meteo weather",
        ),
        (
            "Humidity",
            f"{weather['relative_humidity_2m']:.0f}%",
            "Relative humidity",
            "#38bdf8",
            "Open-Meteo weather",
        ),
        (
            "Wind",
            f"{weather['wind_speed_10m']:.2f} m/s",
            "10 m wind speed",
            "#14b8a6",
            "Open-Meteo weather",
        ),
    )
    st.markdown(
        metric_cards_grid_html(live_metrics, "live"),
        unsafe_allow_html=True,
    )

    interval = prediction["confidence_interval"]
    st.caption(
        f"{prediction['model_horizon']} | observed "
        f"{observed_at.strftime('%Y-%m-%d %H:%M UTC')} | "
        f"VPD {vpd:.3f} kPa"
    )
    figure, summary = build_forecast_range_panel(
        predicted_aqi=float(prediction["predicted_aqi"]),
        confidence_lower=float(interval["lower"]),
        confidence_upper=float(interval["upper"]),
        horizon=horizon,
        current_aqi=current_aqi.value if current_aqi.is_current else None,
        current_is_live=current_aqi.is_current,
    )
    st.markdown(
        (
            '<div class="forecast-summary">'
            '<span class="forecast-summary-label">Forecast comparison</span>'
            f"<span>{html.escape(summary)}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        figure,
        width="stretch",
        config={"displaylogo": False, "displayModeBar": False},
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
    cards: list[tuple[str, str, str, str, str]] = []
    for row in ranked.itertuples(index=False):
        accuracy = (
            "N/A"
            if pd.isna(row.relative_accuracy)
            else f"{row.relative_accuracy:.2f}%"
        )
        r2_text = "R² N/A" if pd.isna(row.r2) else f"R² {row.r2:.4f}"
        cards.append(
            (
                row.model_name,
                accuracy,
                r2_text,
                MODEL_CARD_ACCENTS.get(row.model_name, "#64748b"),
                "",
            )
        )
    st.markdown(
        metric_cards_grid_html(cards, "model"),
        unsafe_allow_html=True,
    )


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
        st.markdown("**Models shown on chart**")
        model_columns = st.columns(len(MODEL_NAMES))
        selected_models = [
            model_name
            for column, model_name in zip(model_columns, MODEL_NAMES)
            if column.checkbox(
                model_name,
                value=model_name in {"LightGBM", "Linear Ridge"},
                key=f"validation_model_{model_name}",
            )
        ]
        marker_column, peak_column = st.columns(2)
        show_wildfire_events = marker_column.checkbox(
            "Show wildfire / smoke event markers",
            value=True,
            key="validation_show_wildfire_events",
        )
        show_detected_peaks = peak_column.checkbox(
            "Show detected AQI peak episodes",
            value=True,
            key="validation_show_detected_peaks",
        )
        wildfire_focus_options = ["Selected date window"]
        wildfire_focus_events = {}
        if show_wildfire_events:
            for event in list_available_wildfire_events(city_name=city_name):
                label = (
                    f"{event.event_name} "
                    f"({event.start_time:%Y-%m-%d} to {event.end_time:%Y-%m-%d})"
                )
                wildfire_focus_options.append(label)
                wildfire_focus_events[label] = event
        wildfire_focus = st.selectbox(
            "Wildfire event focus",
            wildfire_focus_options,
            key="validation_wildfire_focus",
        )

    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.warning("Select both a start date and an end date.")
        return

    start_at, end_at = resolve_historical_window(
        date_range[0],
        date_range[1],
        quick_range,
    )
    selected_wildfire_event = wildfire_focus_events.get(wildfire_focus)
    if selected_wildfire_event is not None:
        start_at = selected_wildfire_event.start_time - pd.Timedelta(days=3)
        end_at = selected_wildfire_event.end_time + pd.Timedelta(days=3)

    try:
        ensure_prediction_data(
            DEFAULT_PREDICTION_PATH,
            DEFAULT_DB_PATH,
        )
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
        build_alignment_figure(
            aligned,
            selected_models,
            city_name=city_name,
            show_wildfire_events=show_wildfire_events,
            show_detected_peaks=show_detected_peaks,
        ),
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


def dashboard_styles() -> str:
    return """
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1180px;
    }
    .aqi-title {
        color: var(--text-color);
        font-size: 2.05rem;
        font-weight: 760;
        letter-spacing: 0;
        margin-bottom: 0.15rem;
    }
    .aqi-subtitle {
        color: var(--text-color);
        font-size: 1.02rem;
        margin-bottom: 0.75rem;
        opacity: 0.68;
    }
    .aqi-accent-strip {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        height: 4px;
        margin: 0 0 1.15rem;
        overflow: hidden;
        border: 1px solid rgba(128, 140, 155, 0.32);
        border-radius: 4px;
    }
    .aqi-accent-good { background: #16a34a; }
    .aqi-accent-moderate { background: #eab308; }
    .aqi-accent-sensitive { background: #f97316; }
    .aqi-accent-unhealthy { background: #dc2626; }
    .aqi-accent-very-unhealthy { background: #7e22ce; }
    .aqi-accent-hazardous { background: #7f1d1d; }
    .metric-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 0.75rem;
        margin: 0.35rem 0 0.9rem;
    }
    .metric-card {
        min-height: 126px;
        padding: 0.85rem 0.9rem;
        border: 1px solid rgba(128, 140, 155, 0.32);
        border-top: 3px solid var(--metric-accent);
        border-radius: 8px;
        background: var(--secondary-background-color);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.12);
    }
    .metric-card-live {
        min-height: 116px;
    }
    .metric-label {
        color: var(--text-color);
        font-size: 0.82rem;
        font-weight: 650;
        line-height: 1.25;
        opacity: 0.72;
    }
    .metric-value {
        color: var(--text-color);
        font-size: 1.82rem;
        font-weight: 720;
        line-height: 1.15;
        margin-top: 0.42rem;
        overflow-wrap: anywhere;
    }
    .metric-detail {
        color: var(--metric-accent);
        font-size: 0.78rem;
        font-weight: 650;
        line-height: 1.3;
        margin-top: 0.5rem;
    }
    .metric-meta {
        color: var(--text-color);
        font-size: 0.68rem;
        line-height: 1.35;
        margin-top: 0.55rem;
        opacity: 0.58;
        overflow-wrap: anywhere;
    }
    .forecast-summary {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem 0.7rem;
        align-items: baseline;
        padding: 0.7rem 0.85rem;
        margin: 0.45rem 0 0.25rem;
        border: 1px solid rgba(128, 140, 155, 0.32);
        border-radius: 8px;
        background: var(--secondary-background-color);
        color: var(--text-color);
        font-size: 0.82rem;
        line-height: 1.4;
    }
    .forecast-summary-label {
        color: #38bdf8;
        font-weight: 700;
    }
    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"],
    [data-testid="stDeckGlJsonChart"],
    [data-testid="stExpander"] {
        border: 1px solid rgba(128, 140, 155, 0.32);
        border-radius: 8px;
        overflow: hidden;
        background: var(--background-color);
    }
    [data-testid="stPlotlyChart"] {
        padding: 0.35rem;
    }
    [data-testid="stDataFrame"] {
        padding: 0.2rem;
    }
    [data-testid="stExpander"] {
        padding: 0.15rem 0.45rem;
    }
    div[data-baseweb="select"] > div,
    [data-testid="stDateInput"] input,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        border-color: rgba(128, 140, 155, 0.42);
        background: var(--secondary-background-color);
    }
    div[data-baseweb="select"] > div:hover,
    [data-testid="stDateInput"] input:hover,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div:hover {
        border-color: #64748b;
    }
    button[data-baseweb="tab"] {
        border-bottom: 2px solid transparent;
        color: var(--text-color);
        opacity: 0.7;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom-color: #38bdf8;
        color: var(--text-color);
        opacity: 1;
    }
    [data-testid="stSegmentedControl"] {
        padding: 0.2rem;
        border: 1px solid rgba(128, 140, 155, 0.32);
        border-radius: 8px;
        background: var(--secondary-background-color);
    }
    .air-alert {
        border: 1px solid;
        border-left-width: 6px;
        padding: 1rem 1.1rem;
        border-radius: 8px;
        line-height: 1.45;
        margin: 0.5rem 0 1rem 0;
    }
    .air-alert-advisory {
        border-color: #0ea5e9;
        background: #f0f9ff;
        color: #0c4a6e;
    }
    .air-alert-warning {
        border-color: #f59e0b;
        background: #fffbeb;
        color: #78350f;
    }
    .air-alert-critical {
        border-color: #dc2626;
        background: #fff1f2;
        color: #7f1d1d;
    }
    @media (max-width: 760px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .aqi-title {
            font-size: 1.7rem;
        }
        .metric-card,
        .metric-card-live {
            min-height: 108px;
        }
        .metric-value {
            font-size: 1.5rem;
        }
    }
    """


st.set_page_config(page_title="California AQI Forecasting", page_icon="CA", layout="wide")
st.markdown(
    f"""
    <style>
    {dashboard_styles()}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="aqi-title">California AQI Forecasting Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="aqi-subtitle">Fresno, Los Angeles, and San Jose hourly AQI nowcasting with leakage-controlled 24-hour forecasting.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="aqi-accent-strip" aria-hidden="true">
        <span class="aqi-accent-good"></span>
        <span class="aqi-accent-moderate"></span>
        <span class="aqi-accent-sensitive"></span>
        <span class="aqi-accent-unhealthy"></span>
        <span class="aqi-accent-very-unhealthy"></span>
        <span class="aqi-accent-hazardous"></span>
    </div>
    """,
    unsafe_allow_html=True,
)

live_tab, diagnostics_tab = st.tabs(
    ["AQI Forecast", "Model Validation"]
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
                current_aqi = fetch_current_aqi(station_name)
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
                current_aqi,
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
