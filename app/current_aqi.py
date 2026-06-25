from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


OPEN_METEO_AIR_QUALITY_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)


@dataclass(frozen=True)
class AQIReading:
    value: float | None
    observed_at: datetime | None
    label: str
    source: str
    is_current: bool


def parse_utc_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_current_payload(payload: dict[str, Any]) -> AQIReading:
    current = payload.get("current")
    if not isinstance(current, dict):
        raise ValueError("Air quality response is missing current conditions.")

    value = float(current["us_aqi"])
    if not math.isfinite(value) or value < 0:
        raise ValueError("Air quality response contains an invalid US AQI value.")

    observed_at = parse_utc_timestamp(str(current["time"]))
    return AQIReading(
        value=value,
        observed_at=observed_at,
        label="Current AQI",
        source="Open-Meteo Air Quality",
        is_current=True,
    )


def load_latest_recorded_aqi(
    db_path: Path,
    station_ids: Sequence[str],
) -> AQIReading | None:
    identifiers = tuple(station_ids)
    if not identifiers:
        return None

    placeholders = ", ".join("?" for _ in identifiers)
    query = f"""
        SELECT target_aqi, time
        FROM meteorology_history
        WHERE station_id IN ({placeholders})
          AND target_aqi IS NOT NULL
        ORDER BY datetime(time) DESC
        LIMIT 1
    """
    with closing(sqlite3.connect(db_path)) as connection:
        row = connection.execute(query, identifiers).fetchone()

    if row is None:
        return None

    value = float(row[0])
    if not math.isfinite(value) or value < 0:
        return None
    return AQIReading(
        value=value,
        observed_at=parse_utc_timestamp(str(row[1])),
        label="Last Recorded AQI",
        source="SQLite history",
        is_current=False,
    )


def resolve_current_aqi(
    *,
    latitude: float,
    longitude: float,
    station_ids: Sequence[str],
    db_path: Path,
    timeout_seconds: int = 30,
    http_get: Callable[..., Any] = requests.get,
) -> AQIReading:
    try:
        response = http_get(
            OPEN_METEO_AIR_QUALITY_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "us_aqi",
                "timezone": "GMT",
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return parse_current_payload(response.json())
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        requests.RequestException,
    ):
        pass

    try:
        fallback = load_latest_recorded_aqi(db_path, station_ids)
    except (OSError, sqlite3.DatabaseError, TypeError, ValueError, OverflowError):
        fallback = None

    if fallback is not None:
        return fallback
    return AQIReading(
        value=None,
        observed_at=None,
        label="Current AQI",
        source="Unavailable",
        is_current=False,
    )
