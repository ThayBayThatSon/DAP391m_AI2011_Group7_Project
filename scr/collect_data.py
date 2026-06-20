import csv
import glob
import os
import time
from datetime import date, timedelta
from functools import lru_cache
from typing import Iterable

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------
# Targets: California locations used for the AQI panel benchmark.
API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LOCATIONS = [
    {
        "station_id": "SJ_OPENMETEO",
        "station_name": "Downtown San Jose - Open-Meteo",
        "slug": "san_jose_ca",
        "latitude": 37.3394,
        "longitude": -121.895,
    },
    {
        "station_id": "FRES_OPENMETEO",
        "station_name": "Fresno - Open-Meteo",
        "slug": "fresno_ca",
        "latitude": 36.7378,
        "longitude": -119.7871,
    },
    {
        "station_id": "LA_OPENMETEO",
        "station_name": "Los Angeles - Open-Meteo",
        "slug": "los_angeles_ca",
        "latitude": 34.0522,
        "longitude": -118.2437,
    },
]

# Use GMT/UTC timestamps for data collection. With timezone="auto", California
# local time contains duplicated 01:00 rows during daylight-saving fall-back
# transitions, which breaks one-to-one AQI/weather merging.
TIMEZONE = "GMT"

# Air-quality archive coverage is much more stable from 2023 onward. Earlier
# years can still be requested by changing START_YEAR, but they often contain
# substantial missing AQI/pollutant values.
START_YEAR = 2023
END_YEAR = 2025

OUTPUT_DIR = os.path.join("data", "raw")
OUTPUT_FILE = os.path.join("data", "processed", "openmeteo_california_hourly_raw.csv")
CACHE_NAME = ".cache"
RAW_FILE_PREFIX = "openmeteo"
FAILED_DOWNLOAD_LOG = os.path.join(OUTPUT_DIR, "failed_downloads.txt")

# Outer retry protects long historical runs from transient Windows socket,
# firewall, VPN, or API connection interruptions.
DOWNLOAD_RETRIES = 3
RETRY_SLEEP_SECONDS = 20
MONTH_SLEEP_SECONDS = 1

AIR_QUALITY_VARIABLES = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "carbon_dioxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "us_aqi",
]

AIR_QUALITY_COLUMNS = [
    "pm10_ug_m3",
    "pm2_5_ug_m3",
    "carbon_monoxide_ug_m3",
    "carbon_dioxide_ppm",
    "nitrogen_dioxide_ug_m3",
    "sulphur_dioxide_ug_m3",
    "ozone_ug_m3",
    "us_aqi",
]

WEATHER_VARIABLES = [
    "temperature_2m",
    "rain",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
    "surface_pressure",
]

WEATHER_COLUMNS = WEATHER_VARIABLES

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def _parse_date_string(date_string: str) -> date:
    return date.fromisoformat(date_string)


def _month_end(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def _build_month_dates(year: int, month: int) -> tuple[str, str]:
    start_date = date(year, month, 1)
    end_date = _month_end(year, month)
    return start_date.isoformat(), end_date.isoformat()


def _iter_month_ranges(start_date: str, end_date: str) -> Iterable[tuple[str, str]]:
    current = _parse_date_string(start_date)
    last = _parse_date_string(end_date)

    while current <= last:
        month_end = _month_end(current.year, current.month)
        if month_end > last:
            month_end = last

        yield current.isoformat(), month_end.isoformat()

        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


# ---------------------------------------------------------------------------
# Open-Meteo client and parsing helpers
# ---------------------------------------------------------------------------
def _normalize_timezone(value):
    if isinstance(value, bytes):
        return value.decode()
    return value


@lru_cache(maxsize=1)
def _make_client() -> openmeteo_requests.Client:
    cache_session = requests_cache.CachedSession(CACHE_NAME, expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def _response_to_dataframe(response, columns: list[str]) -> pd.DataFrame:
    timezone = _normalize_timezone(response.Timezone())
    hourly = response.Hourly()

    timestamps = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ).tz_convert(timezone)
    timestamps = timestamps.tz_localize(None)

    data = {"time": timestamps.strftime("%Y-%m-%dT%H:%M")}
    for i, column in enumerate(columns):
        data[column] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data)
    if df["time"].duplicated().any():
        duplicated = df.loc[df["time"].duplicated(), "time"].head(5).tolist()
        raise ValueError(
            "Open-Meteo returned duplicated timestamps after timezone conversion. "
            f"Use TIMEZONE='GMT' or another non-DST timezone. Examples: {duplicated}"
        )
    return df


def _extract_metadata(response) -> dict[str, str]:
    return {
        "latitude": response.Latitude(),
        "longitude": response.Longitude(),
        "elevation": response.Elevation(),
        "utc_offset_seconds": response.UtcOffsetSeconds(),
        "timezone": _normalize_timezone(response.Timezone()),
        "timezone_abbreviation": _normalize_timezone(response.TimezoneAbbreviation()),
    }


def _save_raw_csv(df: pd.DataFrame, metadata: dict[str, str], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(metadata.keys())
        writer.writerow(metadata.values())
        df.to_csv(f, index=False)


def _build_raw_filepath(location: dict, start_date: str, end_date: str) -> str:
    filename = f"{RAW_FILE_PREFIX}_{location['slug']}_utc_{start_date}_to_{end_date}.csv"
    return os.path.join(OUTPUT_DIR, filename)


# ---------------------------------------------------------------------------
# Download functions
# ---------------------------------------------------------------------------
def _download_period_combined(location: dict, start_date: str, end_date: str, overwrite: bool = False) -> str:
    """Download one monthly/period chunk and merge AQI + weather by timestamp."""
    filepath = _build_raw_filepath(location, start_date, end_date)

    if os.path.exists(filepath) and not overwrite:
        print(f"Skipping download, file already exists: {filepath}")
        return filepath

    client = _make_client()

    common_params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "timezone": TIMEZONE,
        "start_date": start_date,
        "end_date": end_date,
    }
    aq_params = {**common_params, "hourly": AIR_QUALITY_VARIABLES}
    weather_params = {**common_params, "hourly": WEATHER_VARIABLES}

    aq_response = client.weather_api(API_URL, params=aq_params)[0]
    weather_response = client.weather_api(ARCHIVE_URL, params=weather_params)[0]

    aq_df = _response_to_dataframe(aq_response, AIR_QUALITY_COLUMNS)
    weather_df = _response_to_dataframe(weather_response, WEATHER_COLUMNS)
    df = aq_df.merge(weather_df, on="time", how="left", validate="one_to_one")

    metadata = _extract_metadata(aq_response)
    metadata.update(
        {
            "station_id": location["station_id"],
            "station_name": location["station_name"],
            "location_slug": location["slug"],
            "source_air_quality": API_URL,
            "source_weather": ARCHIVE_URL,
            "period_start": start_date,
            "period_end": end_date,
        }
    )

    _save_raw_csv(df, metadata, filepath)
    print(f"Downloaded {filepath} ({len(df)} rows)")
    return filepath


def _download_period_with_retry(location: dict, start_date: str, end_date: str, overwrite: bool = False) -> str:
    last_error = None

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            return _download_period_combined(location, start_date, end_date, overwrite=overwrite)
        except Exception as exc:
            last_error = exc
            print(
                f"Download failed for {location['station_id']} {start_date} to {end_date} "
                f"(attempt {attempt}/{DOWNLOAD_RETRIES}): {exc}"
            )
            _make_client.cache_clear()
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(
        f"Failed after {DOWNLOAD_RETRIES} attempts: {location['station_id']} {start_date} to {end_date}"
    ) from last_error


def _write_failed_downloads(failed_periods: list[tuple[str, str, str, str]]) -> None:
    if not failed_periods:
        if os.path.exists(FAILED_DOWNLOAD_LOG):
            os.remove(FAILED_DOWNLOAD_LOG)
        return

    with open(FAILED_DOWNLOAD_LOG, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["station_id", "start_date", "end_date", "error"])
        writer.writerows(failed_periods)

    print(f"\nFailed periods saved to: {FAILED_DOWNLOAD_LOG}")


def download_range(
    location: dict,
    start_date: str,
    end_date: str,
    overwrite: bool = False,
    failed_periods: list[tuple[str, str, str, str]] | None = None,
) -> list[str]:
    files = []
    for month_start, month_end in _iter_month_ranges(start_date, end_date):
        try:
            files.append(_download_period_with_retry(location, month_start, month_end, overwrite=overwrite))
        except Exception as exc:
            if failed_periods is not None:
                failed_periods.append((location["station_id"], month_start, month_end, str(exc)))
            print(f"Skipping failed period for now: {location['station_id']} {month_start} to {month_end}")
        time.sleep(MONTH_SLEEP_SECONDS)
    return files


def download_year(
    location: dict,
    year: int,
    overwrite: bool = False,
    failed_periods: list[tuple[str, str, str, str]] | None = None,
) -> list[str]:
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    return download_range(location, start_date, end_date, overwrite=overwrite, failed_periods=failed_periods)


def download_years(start_year: int = START_YEAR, end_year: int = END_YEAR, overwrite: bool = False) -> list[str]:
    files = []
    failed_periods: list[tuple[str, str, str, str]] = []
    for location in LOCATIONS:
        print(f"\n=== Downloading {location['station_name']} ===")
        for year in range(start_year, end_year + 1):
            print(f"\n--- {location['station_id']} {year} ---")
            files.extend(download_year(location, year, overwrite=overwrite, failed_periods=failed_periods))

    _write_failed_downloads(failed_periods)
    if failed_periods:
        failed_summary = ", ".join(f"{station} {start} to {end}" for station, start, end, _ in failed_periods)
        raise RuntimeError(
            "Some periods failed to download. Re-run the script later; existing files will be skipped. "
            f"Failed periods: {failed_summary}"
        )

    return files


# ---------------------------------------------------------------------------
# Merge and final export
# ---------------------------------------------------------------------------
def _read_raw_openmeteo(file_path: str) -> pd.DataFrame:
    with open(file_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        values = next(reader)
        metadata = dict(zip(headers, values))

    df = pd.read_csv(file_path, skiprows=2, encoding="utf-8")
    df["time"] = pd.to_datetime(df["time"])

    # Older exported files may have been saved with a GMT+8 offset bug.
    # Keep this compatibility guard so legacy raw files do not silently shift.
    timezone_abbr = metadata.get("timezone_abbreviation", "")
    if metadata.get("utc_offset_seconds") == "28800" or timezone_abbr.startswith("GMT+8"):
        df["time"] = df["time"] - pd.Timedelta(hours=1)

    df.insert(1, "source", "openmeteo")
    df.insert(2, "station_id", metadata.get("station_id", "SJ_OPENMETEO"))
    df.insert(3, "station_name", metadata.get("station_name", metadata.get("location_name", "Open-Meteo Station")))
    df.insert(4, "lat", float(metadata.get("latitude")))
    df.insert(5, "lon", float(metadata.get("longitude")))

    return df


def _files_for_years(years: list[int]) -> list[str]:
    selected = []
    for year in years:
        pattern = os.path.join(OUTPUT_DIR, f"{RAW_FILE_PREFIX}_*_utc_{year}-*.csv")
        selected.extend(glob.glob(pattern))
    return sorted(set(selected))


def merge_years(years: list[int]) -> pd.DataFrame:
    raw_files = _files_for_years(years)

    if not raw_files:
        raise FileNotFoundError(
            f"No Open-Meteo raw files found for years {years}. "
            f"Expected pattern: {RAW_FILE_PREFIX}_LOCATION_utc_YYYY-MM-DD_to_YYYY-MM-DD.csv"
        )

    frames = [_read_raw_openmeteo(file_path) for file_path in raw_files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["station_id", "time"]).sort_values(["time", "station_id"]).reset_index(drop=True)
    return combined


def save_combined(df: pd.DataFrame, output_file: str = OUTPUT_FILE) -> None:
    df.to_csv(output_file, index=False)
    print(f"Saved merged file: {output_file} ({len(df)} rows)")


if __name__ == "__main__":
    target_years = list(range(START_YEAR, END_YEAR + 1))

    download_years(START_YEAR, END_YEAR, overwrite=False)
    combined_df = merge_years(target_years)
    save_combined(combined_df)
