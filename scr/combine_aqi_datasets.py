from pathlib import Path

import pandas as pd


CA_INPUT = Path("data/processed/epa_aqs_station_hourly_clean.csv")
OPENMETEO_INPUT = Path("data/processed/openmeteo_california_hourly_clean.csv")

COMBINED_OUTPUT = Path("data/processed/california_aqi_merged_panel.csv")
MODEL_READY_OUTPUT = Path("data/processed/california_aqi_model_ready.csv")
REPORT_OUTPUT = Path("data/processed/california_aqi_merged_source_report.csv")

COMMON_MODEL_COLUMNS = [
    "time",
    "source",
    "station_id",
    "station_name",
    "lat",
    "lon",
    "pm25_ug_m3",
    "pm10_ug_m3",
    "target_aqi",
    "target_type",
    "us_aqi_official",
    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "hour",
    "dayofweek",
    "month",
    "year",
]

MODEL_READY_COLUMNS = [col for col in COMMON_MODEL_COLUMNS if col != "us_aqi_official"]


def pm25_to_us_aqi(pm25: pd.Series) -> pd.Series:
    """Compute PM2.5-based US AQI proxy with EPA-style breakpoints."""
    breakpoints = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500),
    ]

    aqi = pd.Series(pd.NA, index=pm25.index, dtype="Float64")
    for c_low, c_high, i_low, i_high in breakpoints:
        mask = pm25.between(c_low, c_high, inclusive="both")
        aqi.loc[mask] = ((i_high - i_low) / (c_high - c_low)) * (pm25.loc[mask] - c_low) + i_low
    return aqi.astype(float)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["year"] = df["time"].dt.year
    return df


def normalize_time_to_utc_naive(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert("UTC").dt.tz_localize(None)


def load_ca_pipeline(path: Path = CA_INPUT) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = normalize_time_to_utc_naive(df["time"])

    normalized = pd.DataFrame(
        {
            "time": df["time"],
            "source": "epa_aqs_station",
            "station_id": df["station_id"],
            "station_name": df["station_name"],
            "lat": df["lat"],
            "lon": df["lon"],
            "pm25_ug_m3": df["pm25"],
            "pm10_ug_m3": df["pm10"],
            "target_aqi": df["us_aqi_pm25_proxy"],
            "target_type": "pm25_proxy",
            "us_aqi_official": pd.NA,
            "temperature_2m": df["temperature_2m"],
            "relative_humidity_2m": df["relative_humidity_2m"],
            "rain": df["rain"],
            "cloud_cover": df["cloud_cover"],
            "wind_speed_10m": df["wind_speed_10m"],
            "wind_direction_10m": df["wind_direction_10m"],
            "surface_pressure": df["surface_pressure"],
        }
    )
    return add_time_features(normalized)


def load_openmeteo(path: Path = OPENMETEO_INPUT) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = normalize_time_to_utc_naive(df["time"])

    pm25_proxy = pm25_to_us_aqi(df["pm2_5_ug_m3"])
    station_id = df["station_id"] if "station_id" in df.columns else "SJ_OPENMETEO"
    station_name = (
        df["station_name"]
        if "station_name" in df.columns
        else "Downtown San Jose - Open-Meteo"
    )
    lat = df["lat"] if "lat" in df.columns else 37.3394
    lon = df["lon"] if "lon" in df.columns else -121.895
    source = df["source"] if "source" in df.columns else "openmeteo"

    normalized = pd.DataFrame(
        {
            "time": df["time"],
            "source": source,
            "station_id": station_id,
            "station_name": station_name,
            "lat": lat,
            "lon": lon,
            "pm25_ug_m3": df["pm2_5_ug_m3"],
            "pm10_ug_m3": df["pm10_ug_m3"],
            # Use PM2.5-AQI proxy as the harmonized target. Keep official
            # Open-Meteo US AQI separately to avoid mixing target definitions.
            "target_aqi": pm25_proxy,
            "target_type": "pm25_proxy",
            "us_aqi_official": df["us_aqi"],
            "temperature_2m": df["temperature_2m"],
            "relative_humidity_2m": df["relative_humidity_2m"],
            "rain": df["rain"],
            "cloud_cover": df["cloud_cover"],
            "wind_speed_10m": df["wind_speed_10m"],
            "wind_direction_10m": df["wind_direction_10m"],
            "surface_pressure": df["surface_pressure"],
        }
    )
    return add_time_features(normalized)


def build_report(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in combined.groupby(["source", "station_id", "station_name"], dropna=False):
        source, station_id, station_name = keys
        rows.append(
            {
                "source": source,
                "station_id": station_id,
                "station_name": station_name,
                "rows": len(group),
                "time_min": group["time"].min(),
                "time_max": group["time"].max(),
                "target_missing_pct": round(group["target_aqi"].isna().mean() * 100, 2),
                "pm25_missing_pct": round(group["pm25_ug_m3"].isna().mean() * 100, 2),
                "pm10_missing_pct": round(group["pm10_ug_m3"].isna().mean() * 100, 2),
                "weather_missing_pct": round(
                    group[
                        [
                            "temperature_2m",
                            "relative_humidity_2m",
                            "rain",
                            "cloud_cover",
                            "wind_speed_10m",
                            "wind_direction_10m",
                            "surface_pressure",
                        ]
                    ].isna().mean().mean()
                    * 100,
                    2,
                ),
            }
        )
    return pd.DataFrame(rows)


def combine_datasets() -> pd.DataFrame:
    ca_df = load_ca_pipeline(CA_INPUT)
    openmeteo_df = load_openmeteo(OPENMETEO_INPUT)

    combined = pd.concat([ca_df, openmeteo_df], ignore_index=True)
    combined = combined[COMMON_MODEL_COLUMNS]
    combined = combined.drop_duplicates(subset=["source", "station_id", "time"])
    combined = combined.sort_values(["time", "source", "station_id"]).reset_index(drop=True)

    # Model-ready file uses the harmonized PM2.5-AQI target and common features.
    model_ready = combined.dropna(
        subset=[
            "time",
            "station_id",
            "pm25_ug_m3",
            "pm10_ug_m3",
            "target_aqi",
            "temperature_2m",
            "relative_humidity_2m",
            "rain",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure",
        ]
    ).copy()
    model_ready = model_ready[MODEL_READY_COLUMNS]

    report = build_report(combined)

    combined.to_csv(COMBINED_OUTPUT, index=False)
    model_ready.to_csv(MODEL_READY_OUTPUT, index=False)
    report.to_csv(REPORT_OUTPUT, index=False)

    print(f"CA station rows: {len(ca_df):,}")
    print(f"Open-Meteo rows: {len(openmeteo_df):,}")
    print(f"Combined rows: {len(combined):,}")
    print(f"Model-ready rows: {len(model_ready):,}")
    print(f"Combined time range: {combined['time'].min()} to {combined['time'].max()}")
    print(f"Saved combined panel: {COMBINED_OUTPUT}")
    print(f"Saved model-ready file: {MODEL_READY_OUTPUT}")
    print(f"Saved report: {REPORT_OUTPUT}")
    print("\nRows by source/station:")
    print(combined.groupby(["source", "station_id"]).size().to_string())

    return combined


if __name__ == "__main__":
    combine_datasets()
