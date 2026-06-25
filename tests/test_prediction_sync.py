from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
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
        with closing(sqlite3.connect(self.db_path)) as connection:
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
                    (
                        "2025-11-01 00:00:00",
                        "FRES_OPENMETEO",
                        "Fresno - Open-Meteo",
                        50.0,
                    ),
                    (
                        "2025-11-01 01:00:00",
                        "FRES_OPENMETEO",
                        "Fresno - Open-Meteo",
                        75.0,
                    ),
                ],
            )
            connection.commit()
        pd.DataFrame(
            {
                "Configuration": [
                    "Short-term Autoregressive (Lag 1-3h)",
                    "Short-term Autoregressive (Lag 1-3h)",
                ],
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

        with closing(sqlite3.connect(self.db_path)) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM model_predictions"
            ).fetchone()[0]
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

    def test_import_rejects_reports_without_station_columns(self):
        pd.DataFrame(
            {
                "Configuration": ["Short-term Autoregressive (Lag 1-3h)"],
                "Model": ["LightGBM"],
                "time": ["2025-11-01 00:00:00"],
                "Predicted_AQI": [48.0],
            }
        ).to_csv(self.csv_path, index=False)
        with self.assertRaisesRegex(ValueError, "station_id"):
            sync_prediction_csv(self.csv_path, self.db_path)

    def test_actual_history_is_preserved_when_predictions_have_gaps(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                "INSERT INTO meteorology_history VALUES (?, ?, ?, ?)",
                (
                    "2025-11-01 02:00:00",
                    "FRES_OPENMETEO",
                    "Fresno - Open-Meteo",
                    90.0,
                ),
            )
            connection.commit()
        initialize_prediction_table(self.db_path)
        sync_prediction_csv(self.csv_path, self.db_path)

        aligned = load_validation_data(
            city_name="Fresno",
            scenario="Short-term Nowcasting (1h)",
            start_at=pd.Timestamp("2025-11-01 00:00:00"),
            end_at=pd.Timestamp("2025-11-01 23:59:59"),
            model_names=["LightGBM"],
            db_path=self.db_path,
        )

        actual = (
            aligned[["time", "actual_aqi"]]
            .drop_duplicates("time")
            .sort_values("time")
        )
        self.assertEqual(actual["actual_aqi"].tolist(), [50.0, 75.0, 90.0])
        self.assertEqual(aligned["predicted_aqi"].notna().sum(), 1)


if __name__ == "__main__":
    unittest.main()
