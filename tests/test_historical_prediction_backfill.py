from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app.diagnostics import sqlite_connection
from scr.backfill_historical_model_predictions import (
    EventWindow,
    upsert_prediction_records,
    wildfire_window_mask,
)


class HistoricalPredictionBackfillTest(unittest.TestCase):
    def test_wildfire_window_mask_matches_city_and_target_time(self):
        times = pd.Series(
            pd.to_datetime(
                [
                    "2020-09-05 00:00:00",
                    "2020-09-05 00:00:00",
                    "2020-12-01 00:00:00",
                ]
            )
        )
        station_ids = pd.Series(["FRES", "LA", "FRES"])
        windows = [
            EventWindow(
                name="Creek Fire",
                start=pd.Timestamp("2020-09-04"),
                end=pd.Timestamp("2020-09-30 23:59:59"),
                cities=("Fresno", "San Jose"),
            )
        ]

        self.assertEqual(
            wildfire_window_mask(times, station_ids, windows).tolist(),
            [True, False, False],
        )

    def test_prediction_upsert_is_idempotent(self):
        db_path = self._create_prediction_db()
        record = (
            "2020-09-05 00:00:00",
            "FRES",
            "Fresno - Garland",
            "Fresno",
            "Short-term Nowcasting (1h)",
            1,
            "LightGBM",
            121.5,
        )

        self.assertEqual(upsert_prediction_records([record], db_path), 1)
        self.assertEqual(upsert_prediction_records([record], db_path), 1)

        with sqlite_connection(db_path) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM model_predictions"
            ).fetchone()[0]
        self.assertEqual(count, 1)

    def _create_prediction_db(self) -> Path:
        handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = Path(handle.name)
        handle.close()
        self.addCleanup(lambda: db_path.unlink(missing_ok=True))
        return db_path


if __name__ == "__main__":
    unittest.main()
