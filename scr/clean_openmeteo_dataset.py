from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/processed/openmeteo_california_hourly_raw.csv")
CLEAN_FILE = Path("data/processed/openmeteo_california_hourly_clean.csv")
REPORT_FILE = Path("data/processed/openmeteo_california_missingness_report.csv")

TARGET_COL = "us_aqi"
CORE_AQ_COLS = [
    "us_aqi",
    "pm10_ug_m3",
    "pm2_5_ug_m3",
    "carbon_monoxide_ug_m3",
    "nitrogen_dioxide_ug_m3",
    "sulphur_dioxide_ug_m3",
    "ozone_ug_m3",
]

# Use this if you want the longest valid AQI archive.
MIN_CORE_VALID_START = "2022-08-05"

# Use this if you want a cleaner full-year benchmark window.
RECOMMENDED_STABLE_START = "2023-01-01"

DROP_MISSING_PCT_THRESHOLD = 80.0


def build_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100).round(2),
        }
    )
    return report.sort_values(["missing_pct", "missing_count"], ascending=False)


def clean_openmeteo_dataset(
    input_file: Path = INPUT_FILE,
    clean_file: Path = CLEAN_FILE,
    report_file: Path = REPORT_FILE,
    start_date: str = RECOMMENDED_STABLE_START,
) -> pd.DataFrame:
    df = pd.read_csv(input_file, parse_dates=["time"])
    duplicate_keys = ["time"]
    if "station_id" in df.columns:
        duplicate_keys = ["station_id", "time"]
    df = df.sort_values(duplicate_keys).drop_duplicates(subset=duplicate_keys).reset_index(drop=True)

    report = build_missingness_report(df)
    report.to_csv(report_file, index=False)

    high_missing_cols = report.loc[
        report["missing_pct"] >= DROP_MISSING_PCT_THRESHOLD, "column"
    ].tolist()
    high_missing_cols = [col for col in high_missing_cols if col not in {"time", TARGET_COL}]

    clean = df[df["time"] >= pd.Timestamp(start_date)].copy()
    clean = clean.drop(columns=high_missing_cols, errors="ignore")

    # Target and pollutant columns should not be imputed for the research benchmark.
    available_core_cols = [col for col in CORE_AQ_COLS if col in clean.columns]
    clean = clean.dropna(subset=available_core_cols)

    # Meteorological gaps, if any appear later, can be time-interpolated because
    # they are continuous exogenous measurements rather than the target itself.
    meteo_cols = [
        col
        for col in clean.columns
        if col not in {"time", *available_core_cols}
        and pd.api.types.is_numeric_dtype(clean[col])
    ]
    if meteo_cols:
        if "station_id" in clean.columns:
            clean[meteo_cols] = clean.groupby("station_id", group_keys=False)[meteo_cols].apply(
                lambda group: group.interpolate(method="linear", limit_direction="both")
            )
        else:
            clean[meteo_cols] = clean[meteo_cols].interpolate(method="linear", limit_direction="both")

    clean.to_csv(clean_file, index=False)

    print(f"Input rows: {len(df):,}")
    print(f"Clean rows: {len(clean):,}")
    print(f"Start date used: {start_date}")
    print(f"Dropped high-missing columns: {high_missing_cols}")
    print(f"Saved clean dataset: {clean_file}")
    print(f"Saved missingness report: {report_file}")

    return clean


if __name__ == "__main__":
    clean_openmeteo_dataset()
