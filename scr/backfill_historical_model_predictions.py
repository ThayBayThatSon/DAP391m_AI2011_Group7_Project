from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.diagnostics import (  # noqa: E402
    CONFIGURATION_TO_SCENARIO,
    DEFAULT_DB_PATH,
    DEFAULT_WILDFIRE_EVENT_PATH,
    initialize_prediction_table,
    sqlite_connection,
)
from app.model_loader import load_lightgbm_booster  # noqa: E402
from scr.train_combined_panel_models import (  # noqa: E402
    ARTIFACT_DIR_NAMES,
    CAT_FEATURES,
    CITY_BY_STATION_ID,
    CONFIGURATIONS,
    DATA_PATH,
    MODEL_DIR,
    add_features,
    build_feature_matrix,
)

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - dependency is declared for the app
    CatBoostRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - dependency is declared for the app
    XGBRegressor = None


ARTIFACT_FILENAMES = {
    "LightGBM": "lightgbm.txt",
    "XGBoost": "xgboost.json",
    "CatBoost": "catboost.cbm",
    "Random Forest": "random_forest.joblib",
    "Linear Ridge": "linear_ridge.joblib",
}
DEFAULT_MODELS = tuple(ARTIFACT_FILENAMES)


@dataclass(frozen=True)
class EventWindow:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    cities: tuple[str, ...]


def load_event_windows(event_path: Path = DEFAULT_WILDFIRE_EVENT_PATH) -> list[EventWindow]:
    frame = pd.read_csv(event_path)
    windows: list[EventWindow] = []
    for row in frame.itertuples(index=False):
        cities = tuple(
            city.strip()
            for city in str(row.affected_cities).replace(",", ";").split(";")
            if city.strip()
        )
        windows.append(
            EventWindow(
                name=str(row.event_name),
                start=pd.Timestamp(row.start_time).normalize(),
                end=(
                    pd.Timestamp(row.end_time).normalize()
                    + pd.Timedelta(days=1)
                    - pd.Timedelta(seconds=1)
                ),
                cities=cities,
            )
        )
    return windows


def wildfire_window_mask(
    times: pd.Series,
    station_ids: pd.Series,
    windows: Iterable[EventWindow],
) -> pd.Series:
    normalized_times = pd.to_datetime(times)
    cities = station_ids.map(CITY_BY_STATION_ID)
    mask = pd.Series(False, index=times.index)
    for window in windows:
        city_match = cities.isin(window.cities) | ("All" in window.cities)
        time_match = normalized_times.between(window.start, window.end)
        mask |= city_match & time_match
    return mask.fillna(False)


def load_metadata(configuration: str) -> dict:
    metadata_path = Path(MODEL_DIR) / ARTIFACT_DIR_NAMES[configuration] / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def general_matrix(X: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    general = metadata["preprocessing"]["general"]
    numeric_features = general["numeric_features"]
    mean = np.asarray(general["scaler_mean"], dtype=float)
    scale = np.asarray(general["scaler_scale"], dtype=float)
    numeric = pd.DataFrame(
        (X[numeric_features].astype(float).to_numpy() - mean) / scale,
        columns=numeric_features,
        index=X.index,
    )
    cat = X[CAT_FEATURES].astype(str)
    one_hot = pd.get_dummies(cat, prefix=CAT_FEATURES, dtype=float).reindex(
        columns=general["one_hot_columns"],
        fill_value=0.0,
    )
    return pd.concat([numeric, one_hot], axis=1).reindex(
        columns=general["feature_order"],
        fill_value=0.0,
    )


def catboost_matrix(X: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    general = metadata["preprocessing"]["general"]
    numeric_features = general["numeric_features"]
    mean = np.asarray(general["scaler_mean"], dtype=float)
    scale = np.asarray(general["scaler_scale"], dtype=float)
    numeric = pd.DataFrame(
        (X[numeric_features].astype(float).to_numpy() - mean) / scale,
        columns=numeric_features,
        index=X.index,
    )
    cat = X[CAT_FEATURES].astype(str)
    return pd.concat([numeric, cat], axis=1).reindex(
        columns=metadata["preprocessing"]["catboost"]["feature_order"]
    )


def lightgbm_matrix(X: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    numeric_features = metadata["preprocessing"]["general"]["numeric_features"]
    cat = X[CAT_FEATURES].astype(str)
    one_hot = pd.get_dummies(cat, prefix=CAT_FEATURES, dtype=float)
    matrix = pd.concat([X[numeric_features].astype(float), one_hot], axis=1)
    return matrix.reindex(
        columns=metadata["preprocessing"]["lightgbm"]["feature_order"],
        fill_value=0.0,
    )


def load_model(configuration: str, model_name: str):
    model_path = Path(MODEL_DIR) / ARTIFACT_DIR_NAMES[configuration] / ARTIFACT_FILENAMES[model_name]
    if model_name == "LightGBM":
        return load_lightgbm_booster(model_path)
    if model_name == "XGBoost":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not installed.")
        model = XGBRegressor()
        model.load_model(str(model_path))
        return model
    if model_name == "CatBoost":
        if CatBoostRegressor is None:
            raise RuntimeError("catboost is not installed.")
        model = CatBoostRegressor()
        model.load_model(str(model_path))
        return model
    return joblib.load(model_path)


def predict_model(model, model_name: str, X: pd.DataFrame, metadata: dict) -> np.ndarray:
    if model_name == "LightGBM":
        return np.asarray(model.predict(lightgbm_matrix(X, metadata)), dtype=float)
    if model_name == "CatBoost":
        return np.asarray(model.predict(catboost_matrix(X, metadata)), dtype=float)
    return np.asarray(model.predict(general_matrix(X, metadata)), dtype=float)


def build_records_for_configuration(
    configuration: str,
    *,
    models: Iterable[str] = DEFAULT_MODELS,
    data_path: Path = Path(DATA_PATH),
    event_path: Path = DEFAULT_WILDFIRE_EVENT_PATH,
) -> list[tuple[object, ...]]:
    raw = pd.read_csv(data_path)
    engineered = add_features(raw, configuration=configuration)
    X, y, y_time = build_feature_matrix(engineered, configuration=configuration)
    station_ids = engineered.loc[X.index, "station_id"].reset_index(drop=True)
    station_names = engineered.loc[X.index, "station_name"].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    y_time = pd.to_datetime(y_time).reset_index(drop=True)

    mask = wildfire_window_mask(
        y_time,
        station_ids,
        load_event_windows(event_path),
    )
    if not mask.any():
        return []

    scenario, horizon = CONFIGURATION_TO_SCENARIO[configuration]
    metadata = load_metadata(configuration)
    X_target = X.loc[mask].reset_index(drop=True)
    target_times = y_time.loc[mask].reset_index(drop=True)
    target_station_ids = station_ids.loc[mask].reset_index(drop=True)
    target_station_names = station_names.loc[mask].reset_index(drop=True)

    records: list[tuple[object, ...]] = []
    for model_name in models:
        predictions = predict_model(
            load_model(configuration, model_name),
            model_name,
            X_target,
            metadata,
        )
        for time_value, station_id, station_name, predicted_aqi in zip(
            target_times,
            target_station_ids,
            target_station_names,
            predictions,
            strict=False,
        ):
            city_name = CITY_BY_STATION_ID[str(station_id)]
            records.append(
                (
                    pd.Timestamp(time_value).strftime("%Y-%m-%d %H:%M:%S"),
                    str(station_id),
                    str(station_name),
                    city_name,
                    scenario,
                    horizon,
                    model_name,
                    float(predicted_aqi),
                )
            )
    return records


def upsert_prediction_records(records: list[tuple[object, ...]], db_path: Path) -> int:
    if not records:
        return 0
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


def backfill(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    models: Iterable[str] = DEFAULT_MODELS,
) -> int:
    total = 0
    for configuration in CONFIGURATIONS:
        records = build_records_for_configuration(configuration, models=models)
        inserted = upsert_prediction_records(records, db_path)
        print(
            f"{datetime.now(timezone.utc).isoformat()} | "
            f"{configuration}: upserted {inserted:,} wildfire-window predictions",
            flush=True,
        )
        total += inserted
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill model_predictions for historical wildfire windows."
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=list(DEFAULT_MODELS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = backfill(db_path=args.db_path, models=tuple(args.models))
    print(f"Total upserted prediction rows: {total:,}", flush=True)


if __name__ == "__main__":
    main()
