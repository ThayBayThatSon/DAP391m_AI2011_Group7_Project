from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import requests

from app.current_aqi import resolve_current_aqi


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error")


class CurrentAQITest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "aqi_data.db"
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute(
                """
                CREATE TABLE meteorology_history (
                    time TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    target_aqi REAL
                )
                """
            )
            connection.executemany(
                "INSERT INTO meteorology_history VALUES (?, ?, ?)",
                [
                    ("2025-12-31 21:00:00", "FRES", 61.0),
                    ("2025-12-31 23:00:00", "FRES_OPENMETEO", 69.25),
                    ("2025-12-31 22:00:00", "LA_OPENMETEO", 78.2),
                ],
            )
            connection.commit()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_live_open_meteo_reading_is_current(self):
        reading = resolve_current_aqi(
            latitude=36.7378,
            longitude=-119.7871,
            station_ids=("FRES_OPENMETEO", "FRES"),
            db_path=self.db_path,
            http_get=lambda *args, **kwargs: FakeResponse(
                {"current": {"time": "2026-06-25T08:15", "us_aqi": 73.0}}
            ),
        )

        self.assertEqual(reading.label, "Current AQI")
        self.assertEqual(reading.value, 73.0)
        self.assertTrue(reading.is_current)
        self.assertEqual(reading.source, "Open-Meteo Air Quality")
        self.assertEqual(reading.observed_at.isoformat(), "2026-06-25T08:15:00+00:00")

    def test_malformed_api_uses_latest_matching_sqlite_record(self):
        reading = resolve_current_aqi(
            latitude=36.7378,
            longitude=-119.7871,
            station_ids=("FRES_OPENMETEO", "FRES"),
            db_path=self.db_path,
            http_get=lambda *args, **kwargs: FakeResponse({"current": {}}),
        )

        self.assertEqual(reading.label, "Last Recorded AQI")
        self.assertAlmostEqual(reading.value, 69.25)
        self.assertFalse(reading.is_current)
        self.assertEqual(reading.source, "SQLite history")
        self.assertEqual(reading.observed_at.isoformat(), "2025-12-31T23:00:00+00:00")

    def test_http_failure_uses_sqlite_fallback(self):
        def failing_get(*args, **kwargs):
            raise requests.Timeout("air quality API timed out")

        reading = resolve_current_aqi(
            latitude=34.0522,
            longitude=-118.2437,
            station_ids=("LA_OPENMETEO", "LA"),
            db_path=self.db_path,
            http_get=failing_get,
        )

        self.assertEqual(reading.label, "Last Recorded AQI")
        self.assertAlmostEqual(reading.value, 78.2)
        self.assertEqual(reading.source, "SQLite history")

    def test_missing_api_and_database_data_returns_unavailable(self):
        reading = resolve_current_aqi(
            latitude=37.3394,
            longitude=-121.8950,
            station_ids=("SJ_OPENMETEO",),
            db_path=self.db_path,
            http_get=lambda *args, **kwargs: FakeResponse({"current": {}}),
        )

        self.assertIsNone(reading.value)
        self.assertIsNone(reading.observed_at)
        self.assertEqual(reading.label, "Current AQI")
        self.assertEqual(reading.source, "Unavailable")
        self.assertFalse(reading.is_current)


if __name__ == "__main__":
    unittest.main()
