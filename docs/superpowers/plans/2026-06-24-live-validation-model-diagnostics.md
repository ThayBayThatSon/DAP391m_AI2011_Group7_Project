# Live Validation and Model Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an English Streamlit diagnostics tab that compares five historical AQI model forecasts with SQLite ground truth, supports model-by-model toggling and category-colored Actual AQI history, and packages all ten trained model artifacts.

**Architecture:** Put database access, alignment, metrics, AQI categorization, and Plotly construction in a new import-safe `app/diagnostics.py` module. Keep `app/ui.py` focused on Streamlit controls and rendering, extend the training pipeline to preserve station identity and serialize all model families, and use an idempotent CLI to synchronize the station-aware prediction report into SQLite.

**Tech Stack:** Python 3.11, Streamlit, Plotly Graph Objects, pandas, NumPy, scikit-learn, SQLite, LightGBM, XGBoost, CatBoost, joblib, unittest.

---

## File Map

- Create `app/diagnostics.py`: validation constants, SQLite schema/import/query, historical window resolution, metrics, AQI categories, and Plotly figure construction.
- Create `scr/sync_model_predictions.py`: command-line entry point for idempotently importing the training prediction report into `aqi_data.db`.
- Modify `scr/train_combined_panel_models.py`: preserve station identity, serialize five model families for both horizons, write preprocessing metadata, and retain LightGBM compatibility paths.
- Modify `app/ui.py`: introduce `Live Forecast` and `Live Validation & Model Diagnostics` tabs and render the approved English diagnostics layout.
- Modify `app/README.md`: document full training, prediction synchronization, artifact layout, and diagnostics usage.
- Modify `requirements.txt`: declare `joblib` explicitly.
- Create `tests/test_diagnostics.py`: unit tests for categories, date windows, metrics, SQLite alignment, model selection, and Plotly traces.
- Create `tests/test_prediction_sync.py`: CLI/import tests for prediction CSV validation and idempotent SQLite upserts.
- Modify `tests/test_model_export.py`: test station-aware reports, ten artifact paths, metadata, and LightGBM-only report protection.
- Modify `tests/test_ui_retry.py`: extend the Streamlit fake for tabs and diagnostics controls while preserving local/remote prediction tests.

### Task 1: Diagnostics Metrics, Categories, and Historical Windows

**Files:**
- Create: `app/diagnostics.py`
- Create: `tests/test_diagnostics.py`

- [ ] **Step 1: Write failing tests for category thresholds, quick ranges, and metrics**

```python
# tests/test_diagnostics.py
from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from app.diagnostics import (
    calculate_model_metrics,
    classify_aqi,
    resolve_historical_window,
)


class DiagnosticsMathTest(unittest.TestCase):
    def test_aqi_categories_match_us_thresholds(self):
        self.assertEqual(classify_aqi(50.0).name, "Good")
        self.assertEqual(classify_aqi(51.0).name, "Moderate")
        self.assertEqual(classify_aqi(101.0).name, "Unhealthy for Sensitive Groups")
        self.assertEqual(classify_aqi(151.0).name, "Unhealthy")
        self.assertEqual(classify_aqi(201.0).name, "Very Unhealthy")
        self.assertEqual(classify_aqi(301.0).name, "Hazardous")

    def test_quick_ranges_end_at_selected_day_end(self):
        start, end = resolve_historical_window(
            date(2025, 11, 1),
            date(2025, 12, 31),
            "24 Hours",
        )
        self.assertEqual(end, pd.Timestamp("2025-12-31 23:59:59"))
        self.assertEqual(start, pd.Timestamp("2025-12-31 00:00:00"))

        start, end = resolve_historical_window(
            date(2025, 11, 1),
            date(2025, 12, 31),
            "Full Custom Range",
        )
        self.assertEqual(start, pd.Timestamp("2025-11-01 00:00:00"))
        self.assertEqual(end, pd.Timestamp("2025-12-31 23:59:59"))

    def test_metrics_include_relative_accuracy_from_wmape(self):
        frame = pd.DataFrame(
            {
                "model_name": ["LightGBM"] * 3,
                "actual_aqi": [50.0, 100.0, 150.0],
                "predicted_aqi": [45.0, 110.0, 135.0],
            }
        )
        metrics = calculate_model_metrics(frame).iloc[0]
        self.assertAlmostEqual(metrics["mae"], 10.0)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(350.0 / 3.0))
        self.assertAlmostEqual(metrics["relative_accuracy"], 90.0)

    def test_zero_actual_denominator_returns_nan_accuracy(self):
        frame = pd.DataFrame(
            {
                "model_name": ["Linear Ridge", "Linear Ridge"],
                "actual_aqi": [0.0, 0.0],
                "predicted_aqi": [2.0, 3.0],
            }
        )
        metrics = calculate_model_metrics(frame).iloc[0]
        self.assertTrue(np.isnan(metrics["relative_accuracy"]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_diagnostics.DiagnosticsMathTest -v
```

Expected: import failure because `app.diagnostics` does not exist.

- [ ] **Step 3: Implement category, window, and metric helpers**

```python
# app/diagnostics.py
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


@dataclass(frozen=True)
class AQICategory:
    name: str
    color: str


AQI_CATEGORIES = (
    (50.0, AQICategory("Good", "#16a34a")),
    (100.0, AQICategory("Moderate", "#eab308")),
    (150.0, AQICategory("Unhealthy for Sensitive Groups", "#f97316")),
    (200.0, AQICategory("Unhealthy", "#dc2626")),
    (300.0, AQICategory("Very Unhealthy", "#7e22ce")),
    (math.inf, AQICategory("Hazardous", "#7f1d1d")),
)


def classify_aqi(value: float) -> AQICategory:
    normalized = max(float(value), 0.0)
    for upper_bound, category in AQI_CATEGORIES:
        if normalized <= upper_bound:
            return category
    raise RuntimeError("AQI category lookup failed.")


def resolve_historical_window(
    start_date: date,
    end_date: date,
    quick_range: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if quick_range not in QUICK_RANGES:
        raise ValueError(f"Unsupported historical range: {quick_range}")
    selected_start = pd.Timestamp(start_date).normalize()
    selected_end = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
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
    for model_name, model_rows in aligned.dropna(
        subset=["actual_aqi", "predicted_aqi"]
    ).groupby("model_name", sort=False):
        actual = model_rows["actual_aqi"].astype(float).to_numpy()
        predicted = model_rows["predicted_aqi"].astype(float).to_numpy()
        denominator = float(np.abs(actual).sum())
        wmape = float(np.abs(actual - predicted).sum() / denominator) if denominator else np.nan
        rows.append(
            {
                "model_name": model_name,
                "n": len(model_rows),
                "mae": float(mean_absolute_error(actual, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else np.nan,
                "relative_accuracy": max(0.0, 1.0 - wmape) * 100.0
                if not np.isnan(wmape)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_diagnostics.DiagnosticsMathTest -v
```

Expected: four tests pass.

- [ ] **Step 5: Commit the diagnostics math foundation**

```powershell
git add app/diagnostics.py tests/test_diagnostics.py
git commit -m "Add AQI diagnostics metrics and ranges"
```

### Task 2: SQLite Prediction Schema, Import, and Ground-Truth Alignment

**Files:**
- Modify: `app/diagnostics.py`
- Create: `tests/test_prediction_sync.py`
- Create: `scr/sync_model_predictions.py`

- [ ] **Step 1: Write failing SQLite import and alignment tests**

```python
# tests/test_prediction_sync.py
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.diagnostics import (
    initialize_prediction_table,
    load_validation_data,
    sync_prediction_csv,
)


class PredictionSyncTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "aqi_data.db"
        self.csv_path = Path(self.temp_dir.name) / "predictions.csv"
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE meteorology_history (
                    time TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    station_name TEXT NOT NULL,
                    target_aqi REAL
                )
                """
            )
            connection.executemany(
                "INSERT INTO meteorology_history VALUES (?, ?, ?, ?)",
                [
                    ("2025-11-01 00:00:00", "FRES_OPENMETEO", "Fresno - Open-Meteo", 50.0),
                    ("2025-11-01 01:00:00", "FRES_OPENMETEO", "Fresno - Open-Meteo", 75.0),
                ],
            )
        pd.DataFrame(
            {
                "Configuration": ["Short-term Autoregressive (Lag 1-3h)"] * 2,
                "Model": ["LightGBM", "Linear Ridge"],
                "time": ["2025-11-01 00:00:00", "2025-11-01 01:00:00"],
                "station_id": ["FRES_OPENMETEO", "FRES_OPENMETEO"],
                "station_name": ["Fresno - Open-Meteo", "Fresno - Open-Meteo"],
                "city_name": ["Fresno", "Fresno"],
                "Predicted_AQI": [48.0, 80.0],
            }
        ).to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sync_is_idempotent_and_query_uses_sqlite_actual_aqi(self):
        initialize_prediction_table(self.db_path)
        self.assertEqual(sync_prediction_csv(self.csv_path, self.db_path), 2)
        self.assertEqual(sync_prediction_csv(self.csv_path, self.db_path), 2)
        with sqlite3.connect(self.db_path) as connection:
            count = connection.execute("SELECT COUNT(*) FROM model_predictions").fetchone()[0]
        self.assertEqual(count, 2)

        aligned = load_validation_data(
            city_name="Fresno",
            scenario="Short-term Nowcasting (1h)",
            start_at=pd.Timestamp("2025-11-01 00:00:00"),
            end_at=pd.Timestamp("2025-11-01 23:59:59"),
            model_names=["LightGBM", "Linear Ridge"],
            db_path=self.db_path,
        )
        self.assertEqual(aligned["actual_aqi"].tolist(), [50.0, 75.0])
        self.assertEqual(aligned["predicted_aqi"].tolist(), [48.0, 80.0])

    def test_empty_model_selection_still_returns_actual_history(self):
        initialize_prediction_table(self.db_path)
        actual = load_validation_data(
            city_name="Fresno",
            scenario="Short-term Nowcasting (1h)",
            start_at=pd.Timestamp("2025-11-01 00:00:00"),
            end_at=pd.Timestamp("2025-11-01 23:59:59"),
            model_names=[],
            db_path=self.db_path,
        )
        self.assertEqual(actual["actual_aqi"].tolist(), [50.0, 75.0])
        self.assertTrue(actual["model_name"].isna().all())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the sync tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_prediction_sync -v
```

Expected: import failure for the missing SQLite functions.

- [ ] **Step 3: Implement schema, validation, upsert, and aligned query**

Append to `app/diagnostics.py`:

```python
CONFIGURATION_TO_SCENARIO = {
    details["configuration"]: (label, details["horizon_hours"])
    for label, details in SCENARIOS.items()
}


@contextmanager
def sqlite_connection(db_path: Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
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


def _history_station_ids(city_name: str) -> tuple[str, ...]:
    mapping = {
        "Fresno": ("FRES_OPENMETEO",),
        "Los Angeles": ("LA_OPENMETEO",),
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
    return predictions.merge(actual, on=["time", "station_id"], how="inner")
```

- [ ] **Step 4: Add the synchronization CLI**

```python
# scr/sync_model_predictions.py
from __future__ import annotations

import argparse
from pathlib import Path

from app.diagnostics import DEFAULT_DB_PATH, DEFAULT_PREDICTION_PATH, sync_prediction_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize historical model predictions into SQLite."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTION_PATH)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    count = sync_prediction_csv(args.predictions, args.database)
    print(f"Synchronized {count:,} prediction rows into {args.database}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the sync tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_prediction_sync -v
```

Expected: two tests pass.

- [ ] **Step 6: Commit SQLite diagnostics storage**

```powershell
git add app/diagnostics.py scr/sync_model_predictions.py tests/test_prediction_sync.py
git commit -m "Add SQLite model prediction diagnostics"
```

### Task 3: Station-Aware Prediction Reports

**Files:**
- Modify: `scr/train_combined_panel_models.py`
- Modify: `tests/test_model_export.py`

- [ ] **Step 1: Write a failing test for station-aware report rows**

Add to `tests/test_model_export.py`:

```python
    def test_prediction_report_preserves_station_identity_and_city(self):
        training = load_training_module()
        y_true = pd.Series(
            [50.0, 75.0],
            index=pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 01:00:00"]),
        )
        predictions = {"LightGBM": (y_true, pd.Series([48.0, 78.0], index=y_true.index))}
        report = training.build_prediction_report(
            "Short-term Autoregressive (Lag 1-3h)",
            predictions,
            pd.Series(["FRES_OPENMETEO", "SJ_OPENMETEO"]),
            {
                "FRES_OPENMETEO": "Fresno - Open-Meteo",
                "SJ_OPENMETEO": "Downtown San Jose, California",
            },
        )
        self.assertEqual(report["station_id"].tolist(), ["FRES_OPENMETEO", "SJ_OPENMETEO"])
        self.assertEqual(report["city_name"].tolist(), ["Fresno", "San Jose"])
        self.assertEqual(report["Model"].tolist(), ["LightGBM", "LightGBM"])
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_model_export.ModelExportTest.test_prediction_report_preserves_station_identity_and_city -v
```

Expected: failure because `build_prediction_report` is undefined.

- [ ] **Step 3: Implement city mapping and report construction**

Add to `scr/train_combined_panel_models.py`:

```python
CITY_BY_STATION_ID = {
    "FRES": "Fresno",
    "FRES_OPENMETEO": "Fresno",
    "LA": "Los Angeles",
    "LA_OPENMETEO": "Los Angeles",
    "SJ_OPENMETEO": "San Jose",
}


def build_prediction_report(
    configuration: str,
    predictions: dict[str, tuple[pd.Series, pd.Series]],
    test_station_ids: pd.Series,
    station_name_lookup: dict[str, str],
) -> pd.DataFrame:
    station_ids = test_station_ids.reset_index(drop=True)
    rows = []
    for model_name, (model_y_true, predicted) in predictions.items():
        if len(station_ids) != len(model_y_true):
            raise ValueError("Station identity count does not match prediction rows.")
        frame = pd.DataFrame(
            {
                "Configuration": configuration,
                "Model": model_name,
                "time": model_y_true.index,
                "station_id": station_ids.to_numpy(),
                "Actual_AQI": model_y_true.to_numpy(),
                "Predicted_AQI": predicted.to_numpy(),
            }
        )
        frame["station_name"] = frame["station_id"].map(station_name_lookup)
        frame["city_name"] = frame["station_id"].map(CITY_BY_STATION_ID)
        if frame[["station_name", "city_name"]].isna().any().any():
            raise ValueError("Prediction report contains an unmapped station.")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)
```

In `run_configuration`, create the lookup immediately after feature engineering:

```python
    station_name_lookup = (
        engineered[["station_id", "station_name"]]
        .drop_duplicates("station_id")
        .set_index("station_id")["station_name"]
        .to_dict()
    )
```

Replace the inline `prediction_rows` block with:

```python
    prediction_report = build_prediction_report(
        configuration,
        predictions,
        test_station_ids,
        station_name_lookup,
    )
```

- [ ] **Step 4: Run model export tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_model_export -v
```

Expected: all model export tests pass.

- [ ] **Step 5: Commit station-aware reports**

```powershell
git add scr/train_combined_panel_models.py tests/test_model_export.py
git commit -m "Preserve station identity in model reports"
```

### Task 4: Package Five Models for Both Horizons

**Files:**
- Modify: `scr/train_combined_panel_models.py`
- Modify: `tests/test_model_export.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing tests for artifact paths and metadata**

Add to `tests/test_model_export.py`:

```python
    def test_all_model_artifact_paths_exist_for_each_horizon(self):
        training = load_training_module()
        expected_files = {
            "LightGBM": "lightgbm.txt",
            "XGBoost": "xgboost.json",
            "CatBoost": "catboost.cbm",
            "Random Forest": "random_forest.joblib",
            "Linear Ridge": "linear_ridge.joblib",
        }
        for configuration in training.CONFIGURATIONS:
            paths = training.model_artifact_paths(configuration)
            self.assertEqual({name: path.name for name, path in paths.items()}, expected_files)
            self.assertEqual(paths["LightGBM"].parent.name, training.ARTIFACT_DIR_NAMES[configuration])

    def test_metadata_payload_contains_preprocessing_and_metrics(self):
        training = load_training_module()
        payload = training.build_artifact_metadata(
            configuration="Short-term Autoregressive (Lag 1-3h)",
            feature_metadata={
                "general": {
                    "numeric_features": ["temperature_2m"],
                    "scaler_mean": [10.0],
                    "scaler_scale": [2.0],
                    "one_hot_columns": ["station_id_FRES_OPENMETEO"],
                    "feature_order": ["temperature_2m", "station_id_FRES_OPENMETEO"],
                },
                "catboost": {
                    "feature_order": ["temperature_2m", "source", "station_id"],
                    "categorical_features": ["source", "station_id"],
                },
                "lightgbm": {
                    "feature_order": ["temperature_2m", "station_id_FRES_OPENMETEO"],
                },
            },
            leaderboard=pd.DataFrame(
                [{"Model": "LightGBM", "MAE": 6.0, "RMSE": 9.0, "R2_Score": 0.87}]
            ),
        )
        self.assertEqual(payload["horizon_hours"], 1)
        self.assertEqual(payload["preprocessing"]["general"]["scaler_mean"], [10.0])
        self.assertEqual(payload["metrics"]["LightGBM"]["mae"], 6.0)
```

- [ ] **Step 2: Run the focused artifact tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_model_export.ModelExportTest.test_all_model_artifact_paths_exist_for_each_horizon tests.test_model_export.ModelExportTest.test_metadata_payload_contains_preprocessing_and_metrics -v
```

Expected: failures because artifact path and metadata functions do not exist.

- [ ] **Step 3: Return preprocessing metadata from matrix preparation**

Change `prepare_model_matrices` to return an eighth value:

```python
    preprocessing_metadata = {
        "general": {
            "numeric_features": numeric_features,
            "scaler_mean": scaler.mean_.astype(float).tolist(),
            "scaler_scale": scaler.scale_.astype(float).tolist(),
            "one_hot_columns": ohe_columns.astype(str).tolist(),
            "feature_order": X_train_general.columns.astype(str).tolist(),
        },
        "catboost": {
            "feature_order": X_train_catboost.columns.astype(str).tolist(),
            "categorical_features": CAT_FEATURES.copy(),
        },
    }
    return (
        X_train_general,
        X_val_general,
        X_test_general,
        X_train_catboost,
        X_val_catboost,
        X_test_catboost,
        CAT_FEATURES.copy(),
        preprocessing_metadata,
    )
```

After preparing LightGBM matrices in `run_configuration`, add:

```python
    preprocessing_metadata["lightgbm"] = {
        "feature_order": X_train_lightgbm.columns.astype(str).tolist(),
    }
```

Update the existing matrix test to unpack the eighth return value and assert:

```python
        (
            train_matrix,
            val_matrix,
            test_matrix,
            _,
            _,
            _,
            _,
            metadata,
        ) = training.prepare_model_matrices(x_train, x_val, x_test)
        self.assertEqual(metadata["general"]["feature_order"], train_matrix.columns.tolist())
```

- [ ] **Step 4: Implement artifact paths, model serialization, and metadata**

Add imports and constants to `scr/train_combined_panel_models.py`:

```python
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib

ARTIFACT_DIR_NAMES = {
    "Short-term Autoregressive (Lag 1-3h)": "nowcast_1h",
    "Long-term Forecasting (Lag 24h)": "forecast_24h",
}
ARTIFACT_FILENAMES = {
    "LightGBM": "lightgbm.txt",
    "XGBoost": "xgboost.json",
    "CatBoost": "catboost.cbm",
    "Random Forest": "random_forest.joblib",
    "Linear Ridge": "linear_ridge.joblib",
}
```

Add serialization helpers:

```python
def model_artifact_paths(configuration: str) -> dict[str, Path]:
    directory = Path(MODEL_DIR) / ARTIFACT_DIR_NAMES[configuration]
    return {name: directory / filename for name, filename in ARTIFACT_FILENAMES.items()}


def save_model_artifact(model, model_name: str, configuration: str) -> Path:
    output_path = model_artifact_paths(configuration)[model_name]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if model_name == "LightGBM":
        best_iteration = getattr(model, "best_iteration_", None)
        model.booster_.save_model(
            str(output_path),
            num_iteration=best_iteration if best_iteration else -1,
        )
        compatibility_path = Path(MODEL_OUTPUT_PATHS[configuration])
        compatibility_path.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(
            str(compatibility_path),
            num_iteration=best_iteration if best_iteration else -1,
        )
    elif model_name == "XGBoost":
        model.save_model(str(output_path))
    elif model_name == "CatBoost":
        model.save_model(str(output_path), format="cbm")
    else:
        joblib.dump(model, output_path)
    return output_path


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def build_artifact_metadata(
    configuration: str,
    feature_metadata: dict,
    leaderboard: pd.DataFrame,
) -> dict:
    metrics = {
        row.Model: {
            "mae": float(row.MAE),
            "rmse": float(row.RMSE),
            "r2": float(row.R2_Score),
            "n": int(row.N),
        }
        for row in leaderboard.itertuples(index=False)
    }
    return {
        "configuration": configuration,
        "horizon_hours": CONFIGURATIONS[configuration]["horizon"],
        "train_years": TRAIN_YEARS,
        "validation_years": VALIDATION_YEARS,
        "test_years": TEST_YEARS,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing": feature_metadata,
        "metrics": metrics,
        "library_versions": {
            "python": os.sys.version.split()[0],
            "lightgbm": _package_version("lightgbm"),
            "xgboost": _package_version("xgboost"),
            "catboost": _package_version("catboost"),
            "scikit-learn": _package_version("scikit-learn"),
            "joblib": _package_version("joblib"),
        },
    }


def save_artifact_metadata(
    configuration: str,
    feature_metadata: dict,
    leaderboard: pd.DataFrame,
) -> Path:
    output_path = Path(MODEL_DIR) / ARTIFACT_DIR_NAMES[configuration] / "metadata.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            build_artifact_metadata(configuration, feature_metadata, leaderboard),
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path
```

Replace the LightGBM-only save call inside the training loop with:

```python
        artifact_path = save_model_artifact(model, name, configuration)
        print(f"Saved {name} model: {artifact_path}", flush=True)
```

After creating `leaderboard`, write metadata:

```python
    metadata_path = save_artifact_metadata(
        configuration,
        preprocessing_metadata,
        leaderboard,
    )
    print(f"Saved model metadata: {metadata_path}", flush=True)
```

- [ ] **Step 5: Declare joblib explicitly**

Add to `requirements.txt`:

```text
joblib
```

- [ ] **Step 6: Run model export tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_model_export -v
```

Expected: all tests pass, including existing LightGBM compatibility path assertions.

- [ ] **Step 7: Commit complete model packaging**

```powershell
git add scr/train_combined_panel_models.py tests/test_model_export.py requirements.txt
git commit -m "Package all trained AQI models"
```

### Task 5: Plotly Historical Alignment Figure

**Files:**
- Modify: `app/diagnostics.py`
- Modify: `tests/test_diagnostics.py`

- [ ] **Step 1: Write failing figure tests**

Add to `tests/test_diagnostics.py`:

```python
from app.diagnostics import build_alignment_figure


class DiagnosticsFigureTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    ["2025-11-01 00:00:00", "2025-11-01 01:00:00"] * 2
                ),
                "station_id": ["FRES_OPENMETEO"] * 4,
                "model_name": ["LightGBM", "LightGBM", "Linear Ridge", "Linear Ridge"],
                "actual_aqi": [45.0, 155.0, 45.0, 155.0],
                "predicted_aqi": [47.0, 150.0, 50.0, 145.0],
            }
        )

    def test_actual_baseline_is_black_dashed_and_always_visible(self):
        figure = build_alignment_figure(self.frame, [])
        actual = next(trace for trace in figure.data if trace.name == "Actual AQI")
        self.assertEqual(actual.line.color, "black")
        self.assertEqual(actual.line.width, 3)
        self.assertEqual(actual.line.dash, "dash")
        self.assertFalse(any(trace.name == "LightGBM" for trace in figure.data))

    def test_only_selected_model_curves_are_added(self):
        figure = build_alignment_figure(self.frame, ["LightGBM"])
        names = [trace.name for trace in figure.data]
        self.assertIn("Actual AQI", names)
        self.assertIn("LightGBM", names)
        self.assertNotIn("Linear Ridge", names)
        self.assertEqual(figure.layout.hovermode, "x unified")

    def test_actual_history_contains_category_coloring(self):
        figure = build_alignment_figure(self.frame, ["LightGBM"])
        category_names = {trace.name for trace in figure.data if trace.legendgroup == "aqi-category"}
        self.assertIn("Good (0-50)", category_names)
        self.assertIn("Unhealthy (151-200)", category_names)
```

- [ ] **Step 2: Run figure tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_diagnostics.DiagnosticsFigureTest -v
```

Expected: failure because `build_alignment_figure` is undefined.

- [ ] **Step 3: Implement model palette and Plotly traces**

Append to `app/diagnostics.py`:

```python
MODEL_COLORS = {
    "LightGBM": "#2563eb",
    "XGBoost": "#f97316",
    "Linear Ridge": "#16a34a",
    "CatBoost": "#7e22ce",
    "Random Forest": "#8b5e3c",
}


def _actual_history(aligned: pd.DataFrame) -> pd.DataFrame:
    return (
        aligned[["time", "station_id", "actual_aqi"]]
        .dropna(subset=["actual_aqi"])
        .drop_duplicates(["time", "station_id"])
        .sort_values(["time", "station_id"])
    )


def build_alignment_figure(
    aligned: pd.DataFrame,
    selected_models: Iterable[str],
) -> go.Figure:
    actual = _actual_history(aligned).copy()
    actual["category"] = actual["actual_aqi"].map(lambda value: classify_aqi(value).name)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=actual["time"],
            y=actual["actual_aqi"],
            mode="lines",
            name="Actual AQI",
            customdata=actual[["category"]],
            line={"color": "black", "width": 3, "dash": "dash"},
            hovertemplate="Actual AQI: %{y:.2f}<br>Category: %{customdata[0]}<extra></extra>",
        )
    )
    category_labels = {
        "Good": "Good (0-50)",
        "Moderate": "Moderate (51-100)",
        "Unhealthy for Sensitive Groups": "Unhealthy for Sensitive Groups (101-150)",
        "Unhealthy": "Unhealthy (151-200)",
        "Very Unhealthy": "Very Unhealthy (201-300)",
        "Hazardous": "Hazardous (301+)",
    }
    for _, category in AQI_CATEGORIES:
        category_mask = actual["category"].eq(category.name)
        if not category_mask.any():
            continue
        figure.add_trace(
            go.Scatter(
                x=actual["time"],
                y=actual["actual_aqi"].where(category_mask, np.nan),
                mode="lines+markers",
                name=category_labels[category.name],
                legendgroup="aqi-category",
                line={"color": category.color, "width": 4},
                marker={"color": category.color, "size": 7},
                hoverinfo="skip",
                connectgaps=False,
            )
        )
    selected = tuple(selected_models)
    for model_name in selected:
        model_rows = aligned[aligned["model_name"].eq(model_name)].sort_values("time")
        if model_rows.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=model_rows["time"],
                y=model_rows["predicted_aqi"],
                mode="lines",
                name=model_name,
                line={"color": MODEL_COLORS[model_name], "width": 2},
                hovertemplate=f"{model_name}: %{{y:.2f}}<extra></extra>",
            )
        )
    figure.update_layout(
        hovermode="x unified",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=460,
        margin={"l": 55, "r": 25, "t": 35, "b": 55},
        legend={"orientation": "h", "y": 1.12},
        xaxis_title="Timestamp",
        yaxis_title="AQI",
    )
    figure.update_xaxes(showline=True, linecolor="#cbd5e1", gridcolor="whitesmoke")
    figure.update_yaxes(showline=True, linecolor="#cbd5e1", gridcolor="whitesmoke")
    return figure
```

- [ ] **Step 4: Run all diagnostics tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_diagnostics tests.test_prediction_sync -v
```

Expected: all diagnostics and sync tests pass.

- [ ] **Step 5: Commit Plotly diagnostics chart**

```powershell
git add app/diagnostics.py tests/test_diagnostics.py
git commit -m "Add historical AQI diagnostics chart"
```

### Task 6: Streamlit Diagnostics Tab

**Files:**
- Modify: `app/ui.py`
- Modify: `tests/test_ui_retry.py`

- [ ] **Step 1: Extend the Streamlit fake and write a failing tab behavior assertion**

Add these methods and fields to `FakeStreamlit` in `tests/test_ui_retry.py`:

```python
        self.tab_labels: list[str] = []
        self.warning_messages: list[str] = []

    def tabs(self, labels):
        self.tab_labels = list(labels)
        return [FakePanel() for _ in labels]

    def radio(self, label, options, **kwargs):
        return options[0]

    def date_input(self, *args, **kwargs):
        return kwargs["value"]

    def multiselect(self, *args, **kwargs):
        return kwargs.get("default", [])

    def segmented_control(self, label, options, **kwargs):
        return kwargs.get("default", options[0])

    def expander(self, *args, **kwargs):
        return FakePanel()

    def plotly_chart(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def warning(self, message):
        self.warning_messages.append(str(message))

    def info(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None
```

Add no-op methods to `FakePanel`:

```python
    def selectbox(self, label, options, **kwargs):
        return options[0]

    def radio(self, label, options, **kwargs):
        return options[0]

    def warning(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None
```

Create a fake diagnostics module inside the existing dashboard test so importing
`ui.py` cannot touch the real SQLite database:

```python
        fake_diagnostics = types.ModuleType("app.diagnostics")
        fake_diagnostics.DEFAULT_DB_PATH = PROJECT_ROOT / "test-aqi-data.db"
        fake_diagnostics.MODEL_NAMES = (
            "LightGBM",
            "XGBoost",
            "CatBoost",
            "Random Forest",
            "Linear Ridge",
        )
        fake_diagnostics.QUICK_RANGES = (
            "24 Hours",
            "7 Days",
            "30 Days",
            "Full Custom Range",
        )
        fake_diagnostics.SCENARIOS = {
            "Short-term Nowcasting (1h)": {},
            "Long-term Forecasting (24h)": {},
        }
        fake_diagnostics.initialize_prediction_table = lambda db_path: None
        fake_diagnostics.resolve_historical_window = (
            lambda start_date, end_date, quick_range: (
                pd.Timestamp(start_date),
                pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
            )
        )
        fake_diagnostics.load_validation_data = lambda *args, **kwargs: pd.DataFrame()
        fake_diagnostics.calculate_model_metrics = lambda frame: pd.DataFrame()
        fake_diagnostics.build_alignment_figure = lambda frame, models: object()
```

Add `import pandas as pd` to `tests/test_ui_retry.py`, then include
`"app.diagnostics": fake_diagnostics` in the existing `patch.dict(sys.modules, ...)`.

Inside the existing dashboard test, assert:

```python
        self.assertEqual(
            fake_streamlit.tab_labels,
            ["Live Forecast", "📊 Live Validation & Model Diagnostics"],
        )
```

- [ ] **Step 2: Run the UI test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_ui_retry -v
```

Expected: failure because `ui.py` does not create tabs.

- [ ] **Step 3: Add diagnostics imports and render helpers to `app/ui.py`**

Add imports:

```python
from datetime import date
from pathlib import Path

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
```

Add helpers before page rendering:

```python
EMPTY_HISTORY_WARNING = (
    "No historical records were found for the selected period. "
    "Please update the filters."
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
        city_name,
        scenario,
        start_at,
        end_at,
        model_names,
        DEFAULT_DB_PATH,
    )


def render_metric_cards(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        st.info("Select at least one model to display model metrics.")
        return
    ranked = metrics.sort_values("relative_accuracy", ascending=False, na_position="last")
    columns = st.columns(min(len(ranked), 5))
    for column, row in zip(columns, ranked.itertuples(index=False)):
        accuracy = "N/A" if pd.isna(row.relative_accuracy) else f"{row.relative_accuracy:.2f}%"
        r2_text = "N/A" if pd.isna(row.r2) else f"R² {row.r2:.4f}"
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
            default="Full Custom Range",
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
        "Actual AQI is always visible and colored by US AQI health category. "
        "Hover over a timestamp to compare every active model."
    )
    st.plotly_chart(
        build_alignment_figure(aligned, selected_models),
        width="stretch",
        config={"displaylogo": False},
    )

    st.subheader("Relative Prediction Accuracy")
    metrics = calculate_model_metrics(aligned)
    render_metric_cards(metrics)
    st.info(
        "Relative Prediction Accuracy = max(0, 100% − WMAPE), where WMAPE is "
        "the sum of absolute prediction errors divided by the sum of Actual AQI. "
        "A score of 100% means an exact match. R² measures explained AQI variance "
        "and is not an accuracy percentage."
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
```

Also add `import sqlite3` to `app/ui.py`.

- [ ] **Step 4: Move the current dashboard into the Live Forecast tab**

Replace top-level rendering after the title/subtitle with:

```python
live_tab, diagnostics_tab = st.tabs(
    ["Live Forecast", "📊 Live Validation & Model Diagnostics"]
)

with live_tab:
    left_panel, right_panel = st.columns([0.42, 0.58], gap="large")
    with left_panel:
        station_name = st.selectbox("Station", list(STATIONS.keys()), index=0)
        horizon = st.slider(
            "Forecast Horizon",
            min_value=1,
            max_value=24,
            value=24,
            step=23,
            format="%d h",
        )
        station_frame = pd.DataFrame(
            [{"lat": STATIONS[station_name]["lat"], "lon": STATIONS[station_name]["lon"]}]
        )
        st.map(station_frame, latitude="lat", longitude="lon", zoom=8, width="stretch")

    try:
        with right_panel:
            with st.spinner("Updating forecast..."):
                observed_at, weather = fetch_realtime_weather(station_name)
                vpd = vpd_kpa(weather["temperature_2m"], weather["relative_humidity_2m"])
                prediction = call_prediction_api(station_name, horizon, observed_at, weather)
            render_live_forecast_content(observed_at, weather, prediction, horizon, vpd)
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
```

Extract the existing warning, metrics, caption, and Altair chart body into:

```python
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
    metric_columns[0].metric("Predicted AQI", f"{prediction['predicted_aqi']:.2f}", category)
    metric_columns[1].metric("Temperature", f"{weather['temperature_2m']:.1f} C")
    metric_columns[2].metric("Humidity", f"{weather['relative_humidity_2m']:.0f}%")
    metric_columns[3].metric("Wind", f"{weather['wind_speed_10m']:.2f} m/s")
    interval = prediction["confidence_interval"]
    st.caption(
        f"{prediction['model_horizon']} | observed "
        f"{observed_at.strftime('%Y-%m-%d %H:%M UTC')} | "
        f"CI {interval['lower']:.2f}-{interval['upper']:.2f} AQI | "
        f"VPD {vpd:.3f} kPa"
    )
    render_timeline_chart(timeline_frame(prediction, horizon), category_color)
```

- [ ] **Step 5: Run UI and diagnostics tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_ui_retry tests.test_diagnostics tests.test_prediction_sync -v
```

Expected: all tests pass with no Streamlit fake errors.

- [ ] **Step 6: Commit the Streamlit diagnostics tab**

```powershell
git add app/ui.py tests/test_ui_retry.py
git commit -m "Add live validation diagnostics tab"
```

### Task 7: Documentation and End-to-End Data Generation

**Files:**
- Modify: `app/README.md`
- Generate: `models/nowcast_1h/*`
- Generate: `models/forecast_24h/*`
- Generate: `data/processed/california_aqi_model_leaderboard.csv`
- Generate: `data/processed/california_aqi_scenario_evaluation.csv`
- Generate: `data/processed/california_aqi_model_predictions.csv`
- Modify local runtime database: `aqi_data.db` (gitignored)

- [ ] **Step 1: Update README commands and artifact documentation**

Add these sections to `app/README.md`:

```markdown
## Train and Package All Five Models

Run the full benchmark training pipeline:

```powershell
.\.venv\Scripts\python.exe scr/train_combined_panel_models.py
```

This trains LightGBM, XGBoost, CatBoost, Random Forest, and Linear Ridge for
both the 1-hour and 24-hour configurations. It writes ten model artifacts under:

```text
models/nowcast_1h/
models/forecast_24h/
```

Each directory also contains `metadata.json` with feature order, preprocessing
statistics, library versions, data split years, and test metrics. The FastAPI
compatibility files remain at:

```text
models/lightgbm_nowcast.txt
models/lightgbm_forecast24h.txt
```

## Synchronize Historical Predictions

After full training, import the station-aware 2025 backtest predictions:

```powershell
.\.venv\Scripts\python.exe scr/sync_model_predictions.py
```

The command creates or updates the SQLite `model_predictions` table. It is safe
to run repeatedly.

## Live Validation and Model Diagnostics

Start Streamlit:

```powershell
.\.venv\Scripts\streamlit.exe run app/ui.py
```

Open the `Live Validation & Model Diagnostics` tab to:

- Filter Fresno, Los Angeles, or San Jose.
- Switch between 1-hour nowcasting and 24-hour forecasting.
- Select 24-hour, 7-day, 30-day, or custom historical windows.
- Add or remove any trained model independently.
- Compare predictions with category-colored Actual AQI history.
- Recalculate MAE, RMSE, R², and Relative Prediction Accuracy.
```

- [ ] **Step 2: Run the full unit test suite before expensive training**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 3: Run full five-model training and package ten artifacts**

Run:

```powershell
.\.venv\Scripts\python.exe scr/train_combined_panel_models.py
```

Expected:

- Both configurations finish.
- The final leaderboard contains ten rows.
- `data/processed/california_aqi_model_predictions.csv` contains all five model names, both configurations, and station columns.
- All ten model files and two metadata files exist.
- The two root LightGBM compatibility files still exist.

- [ ] **Step 4: Synchronize predictions into SQLite**

Run:

```powershell
.\.venv\Scripts\python.exe scr/sync_model_predictions.py
```

Expected: the command reports the synchronized row count without errors.

- [ ] **Step 5: Verify generated report and database coverage**

Run:

```powershell
@'
import sqlite3
from pathlib import Path

import pandas as pd

predictions = pd.read_csv("data/processed/california_aqi_model_predictions.csv")
assert set(predictions["Model"]) == {
    "LightGBM", "XGBoost", "CatBoost", "Random Forest", "Linear Ridge"
}
assert predictions["Configuration"].nunique() == 2
assert predictions["city_name"].nunique() == 3

with sqlite3.connect("aqi_data.db") as connection:
    coverage = pd.read_sql_query(
        """
        SELECT city_name, scenario, model_name, COUNT(*) AS rows
        FROM model_predictions
        GROUP BY city_name, scenario, model_name
        ORDER BY city_name, scenario, model_name
        """,
        connection,
    )
assert len(coverage) == 30
assert (coverage["rows"] > 0).all()

required = [
    Path("models/nowcast_1h/lightgbm.txt"),
    Path("models/nowcast_1h/xgboost.json"),
    Path("models/nowcast_1h/catboost.cbm"),
    Path("models/nowcast_1h/random_forest.joblib"),
    Path("models/nowcast_1h/linear_ridge.joblib"),
    Path("models/nowcast_1h/metadata.json"),
    Path("models/forecast_24h/lightgbm.txt"),
    Path("models/forecast_24h/xgboost.json"),
    Path("models/forecast_24h/catboost.cbm"),
    Path("models/forecast_24h/random_forest.joblib"),
    Path("models/forecast_24h/linear_ridge.joblib"),
    Path("models/forecast_24h/metadata.json"),
]
assert all(path.exists() and path.stat().st_size > 0 for path in required)
print(coverage.to_string(index=False))
'@ | .\.venv\Scripts\python.exe -
```

Expected: 30 city/scenario/model groups and no assertion failures.

- [ ] **Step 6: Start Streamlit and perform browser smoke testing**

Run:

```powershell
.\.venv\Scripts\streamlit.exe run app/ui.py
```

Expected: Streamlit starts on `http://localhost:8501`. Verify:

1. Both tabs render.
2. Default diagnostics filters show Fresno, Short-term Nowcasting, November 1 through December 31, 2025, LightGBM, and Linear Ridge.
3. Adding and removing each model updates the chart and leaderboard.
4. Removing all models leaves Actual AQI visible and shows the model-selection guidance.
5. Actual AQI markers use the six US AQI category colors.
6. Unified hover shows timestamp, Actual AQI, category, and active model predictions.
7. Relative Accuracy explanation states `100% − WMAPE` and distinguishes R².
8. The empty-range warning is shown for a date range without records.

- [ ] **Step 7: Run final verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
git status --short
```

Expected: all tests pass, `git diff --check` produces no errors, and status lists only intended source, report, and model artifact changes.

- [ ] **Step 8: Commit docs and generated reproducibility artifacts**

Do not add `aqi_data.db`, because it is a local runtime database. Stage only intended files:

```powershell
git add app/README.md `
  data/processed/california_aqi_model_leaderboard.csv `
  data/processed/california_aqi_scenario_evaluation.csv `
  data/processed/california_aqi_model_predictions.csv `
  models/nowcast_1h `
  models/forecast_24h `
  models/lightgbm_nowcast.txt `
  models/lightgbm_forecast24h.txt
git commit -m "Generate full AQI validation artifacts"
```
