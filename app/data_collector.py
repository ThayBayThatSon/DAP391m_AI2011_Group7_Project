from __future__ import annotations

import logging
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from apscheduler.schedulers.blocking import BlockingScheduler


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DB_PATH = PROJECT_ROOT / "aqi_data.db"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 30

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
        "station_id": "FRES_OPENMETEO",
        "station_name": "Fresno",
        "latitude": 36.7378,
        "longitude": -119.7871,
    },
    "Los Angeles": {
        "station_id": "LA_OPENMETEO",
        "station_name": "Los Angeles",
        "latitude": 34.0522,
        "longitude": -118.2437,
    },
    "San Jose": {
        "station_id": "SJ_OPENMETEO",
        "station_name": "San Jose",
        "latitude": 37.3394,
        "longitude": -121.8950,
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("aqi-data-collector")


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


def vapor_pressure_deficit_kpa(temperature_c: float, relative_humidity_pct: float) -> float:
    saturation_vapor_pressure = 0.6108 * math.exp((17.27 * temperature_c) / (temperature_c + 237.3))
    actual_vapor_pressure = (relative_humidity_pct / 100.0) * saturation_vapor_pressure
    return max(saturation_vapor_pressure - actual_vapor_pressure, 0.0)


def cyclic_time_features(timestamp_utc: datetime) -> dict[str, float | int]:
    hour = timestamp_utc.hour
    month = timestamp_utc.month
    return {
        "hour": hour,
        "dayofweek": timestamp_utc.weekday(),
        "month": month,
        "year": timestamp_utc.year,
        "hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "month_sin": math.sin(2 * math.pi * month / 12.0),
        "month_cos": math.cos(2 * math.pi * month / 12.0),
        "is_weekend": int(timestamp_utc.weekday() in (5, 6)),
    }


def wind_components(speed_mps: float, direction_degrees: float) -> tuple[float, float]:
    radians = math.radians(direction_degrees)
    return speed_mps * math.cos(radians), speed_mps * math.sin(radians)


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
    selected_time = parsed_times[selected_index].replace(minute=0, second=0, microsecond=0)

    values: dict[str, float] = {}
    for variable in WEATHER_VARIABLES:
        series = hourly.get(variable)
        if series is None or selected_index >= len(series):
            raise ValueError(f"Open-Meteo response is missing hourly variable: {variable}")
        values[variable] = float(series[selected_index])
    return selected_time, values


def fetch_station_weather(station: dict[str, Any]) -> dict[str, Any]:
    params = {
        "latitude": station["latitude"],
        "longitude": station["longitude"],
        "hourly": ",".join(WEATHER_VARIABLES),
        "timezone": "GMT",
        "past_days": 1,
        "forecast_days": 1,
        "wind_speed_unit": "ms",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    selected_time, weather_values = select_latest_hourly_record(response.json())
    vpd = vapor_pressure_deficit_kpa(
        weather_values["temperature_2m"],
        weather_values["relative_humidity_2m"],
    )
    wind_u, wind_v = wind_components(
        weather_values["wind_speed_10m"],
        weather_values["wind_direction_10m"],
    )

    return {
        "time": selected_time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "openmeteo_realtime",
        "station_id": station["station_id"],
        "station_name": station["station_name"],
        "lat": float(station["latitude"]),
        "lon": float(station["longitude"]),
        **weather_values,
        "vpd_kpa": vpd,
        **cyclic_time_features(selected_time),
        "wind_u": wind_u,
        "wind_v": wind_v,
        "target_aqi": None,
        "pm25_ug_m3": None,
        "pm10_ug_m3": None,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


def upsert_meteorology_row(row: dict[str, Any]) -> None:
    columns = list(row.keys())
    placeholders = ", ".join("?" for _ in columns)
    update_columns = [column for column in columns if column not in {"station_id", "time"}]
    update_clause = ", ".join(f"{column}=excluded.{column}" for column in update_columns)

    sql = f"""
        INSERT INTO meteorology_history ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(station_id, time) DO UPDATE SET {update_clause}
    """
    with sqlite_connection() as conn:
        conn.execute(sql, [row[column] for column in columns])


def collect_all_stations() -> None:
    initialize_database()
    for station_name, station in STATIONS.items():
        try:
            row = fetch_station_weather(station)
            upsert_meteorology_row(row)
            logger.info(
                "Stored %s weather at %s | temp=%.2f C rh=%.2f%% wind=%.2f m/s vpd=%.3f kPa",
                station_name,
                row["time"],
                row["temperature_2m"],
                row["relative_humidity_2m"],
                row["wind_speed_10m"],
                row["vpd_kpa"],
            )
        except Exception as exc:
            logger.exception("Failed to collect %s meteorology: %s", station_name, exc)


def main() -> None:
    initialize_database()
    collect_all_stations()

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        collect_all_stations,
        trigger="interval",
        minutes=60,
        id="hourly_openmeteo_collection",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )

    logger.info("AQI meteorology collector started. SQLite database: %s", DB_PATH)
    scheduler.start()


if __name__ == "__main__":
    main()
