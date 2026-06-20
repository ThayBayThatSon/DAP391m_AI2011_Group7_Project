from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/processed/epa_aqs_station_hourly_raw.csv")
OUTPUT_FILE = Path("data/processed/epa_aqs_station_hourly_clean.csv")
REPORT_FILE = Path("data/processed/epa_aqs_station_missingness_report.csv")

# FRES and LA provide the most complete pollutant histories in the current file.
DEFAULT_STATIONS = ["FRES", "LA"]

# These columns are too sparse for a robust benchmark in this dataset.
DROP_COLUMNS = ["no", "nox"]

# Keep short interpolation conservative. Long gaps should remain missing and be
# removed for rows where those variables are required.
MAX_INTERPOLATE_HOURS = 6

POLLUTANT_COLUMNS = ["co", "no2", "o3", "pm10", "pm25", "so2"]
METEO_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
]


def pm25_to_us_aqi(pm25: pd.Series) -> pd.Series:
    """Compute US AQI from PM2.5 concentration using EPA breakpoints."""
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


def build_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    overall = pd.DataFrame(
        {
            "station_id": "ALL",
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100).round(2),
        }
    )

    station_reports = []
    for station, group in df.groupby("station_id"):
        station_reports.append(
            pd.DataFrame(
                {
                    "station_id": station,
                    "column": group.columns,
                    "missing_count": group.isna().sum().values,
                    "missing_pct": (group.isna().mean().values * 100).round(2),
                }
            )
        )

    return pd.concat([overall, *station_reports], ignore_index=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["year"] = df["time"].dt.year
    return df


def clean_station_group(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("time").copy()
    numeric_cols = [col for col in POLLUTANT_COLUMNS + METEO_COLUMNS if col in group.columns]

    # Short-gap interpolation only within each station. This keeps sensor outages
    # from being artificially filled across long missing intervals.
    group[numeric_cols] = group[numeric_cols].interpolate(
        method="linear",
        limit=MAX_INTERPOLATE_HOURS,
        limit_direction="both",
    )
    return group


def prepare_ca_pipeline_input(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
    report_file: Path = REPORT_FILE,
    stations: list[str] | None = None,
) -> pd.DataFrame:
    stations = stations or DEFAULT_STATIONS

    df = pd.read_csv(input_file, parse_dates=["time"])
    df = df.sort_values(["station_id", "time"]).drop_duplicates(["station_id", "time"])

    report = build_missingness_report(df)
    report.to_csv(report_file, index=False)

    clean = df[df["station_id"].isin(stations)].copy()
    clean = clean.drop(columns=DROP_COLUMNS, errors="ignore")
    clean = pd.concat(
        [clean_station_group(group) for _, group in clean.groupby("station_id")],
        ignore_index=True,
    )

    # The source file does not contain official US AQI. For modeling continuity,
    # create a PM2.5-derived AQI proxy and keep PM2.5 as the scientifically direct
    # target option. Do not impute PM2.5 if it is still missing after short-gap fill.
    clean = clean.dropna(subset=["pm25"])
    clean["us_aqi_pm25_proxy"] = pm25_to_us_aqi(clean["pm25"])
    clean = clean.dropna(subset=["us_aqi_pm25_proxy"])

    clean = add_time_features(clean)
    clean = clean.sort_values(["time", "station_id"]).reset_index(drop=True)
    clean.to_csv(output_file, index=False)

    print(f"Input rows: {len(df):,}")
    print(f"Selected stations: {stations}")
    print(f"Clean rows: {len(clean):,}")
    print(f"Time range: {clean['time'].min()} to {clean['time'].max()}")
    print(f"Saved clean dataset: {output_file}")
    print(f"Saved missingness report: {report_file}")

    remaining_missing = clean.isna().mean().mul(100).sort_values(ascending=False)
    print("\nRemaining missing percentage:")
    print(remaining_missing[remaining_missing > 0].round(2).to_string())

    return clean


if __name__ == "__main__":
    prepare_ca_pipeline_input()
