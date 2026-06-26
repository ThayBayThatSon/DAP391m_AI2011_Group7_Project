from __future__ import annotations

import unittest
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from app.diagnostics import (
    build_alignment_figure,
    calculate_model_metrics,
    classify_aqi,
    detect_aqi_peak_episodes,
    list_available_wildfire_events,
    load_validation_data,
    list_wildfire_events,
    load_wildfire_events,
    resolve_historical_window,
    sqlite_connection,
)


def sample_alignment_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2025-11-01 00:00:00",
                    "2025-11-01 01:00:00",
                    "2025-11-01 00:00:00",
                    "2025-11-01 01:00:00",
                ]
            ),
            "station_id": ["FRES_OPENMETEO"] * 4,
            "model_name": [
                "LightGBM",
                "LightGBM",
                "Linear Ridge",
                "Linear Ridge",
            ],
            "actual_aqi": [45.0, 155.0, 45.0, 155.0],
            "predicted_aqi": [47.0, 150.0, 50.0, 145.0],
        }
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
            "7 Days",
        )
        self.assertEqual(start, pd.Timestamp("2025-12-25 00:00:00"))
        self.assertEqual(end, pd.Timestamp("2025-12-31 23:59:59"))

        start, end = resolve_historical_window(
            date(2025, 11, 1),
            date(2025, 12, 31),
            "30 Days",
        )
        self.assertEqual(start, pd.Timestamp("2025-12-02 00:00:00"))
        self.assertEqual(end, pd.Timestamp("2025-12-31 23:59:59"))

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


class DiagnosticsFigureTest(unittest.TestCase):
    def setUp(self):
        self.frame = sample_alignment_frame()

    def test_actual_aqi_is_a_thick_solid_category_colored_path(self):
        figure = build_alignment_figure(self.frame, [])
        actual = next(trace for trace in figure.data if trace.name == "Actual AQI")
        self.assertEqual(actual.mode, "markers")
        self.assertEqual(actual.marker.color, "rgba(0,0,0,0)")
        category_traces = [
            trace
            for trace in figure.data
            if trace.legendgroup == "aqi-category"
        ]
        self.assertTrue(category_traces)
        self.assertTrue(all(trace.mode == "lines" for trace in category_traces))
        self.assertTrue(all(trace.line.width == 4 for trace in category_traces))
        self.assertTrue(all(trace.line.dash is None for trace in category_traces))
        self.assertFalse(
            any(
                trace.line.dash == "dash"
                for trace in figure.data
                if getattr(trace, "line", None) is not None
            )
        )
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
        category_traces = [
            trace
            for trace in figure.data
            if trace.legendgroup == "aqi-category"
        ]
        category_names = {trace.name for trace in category_traces}
        self.assertIn("Good (0-50)", category_names)
        self.assertIn("Unhealthy (151-200)", category_names)
        self.assertTrue(all(trace.mode == "lines" for trace in category_traces))
        self.assertTrue(all(trace.showlegend is False for trace in category_traces))
        self.assertTrue(all(trace.line.width == 4 for trace in category_traces))
        self.assertGreaterEqual(len(figure.layout.shapes), 6)

    def test_white_plot_uses_dark_text_in_any_streamlit_theme(self):
        figure = build_alignment_figure(self.frame, ["LightGBM"])
        self.assertEqual(figure.layout.font.color, "#111827")
        self.assertEqual(figure.layout.legend.bgcolor, "rgba(255,255,255,0.92)")
        self.assertEqual(figure.layout.xaxis.tickfont.color, "#111827")
        self.assertEqual(figure.layout.yaxis.tickfont.color, "#111827")

    def test_prediction_curves_are_visually_deemphasized(self):
        figure = build_alignment_figure(self.frame, ["LightGBM"])
        prediction = next(
            trace for trace in figure.data if trace.name == "LightGBM"
        )
        self.assertEqual(prediction.line.width, 1.25)
        self.assertEqual(prediction.line.dash, "dot")
        self.assertEqual(prediction.opacity, 0.58)
        self.assertEqual(figure.layout.height, 400)
        self.assertEqual(tuple(figure.layout.yaxis.range), (0, 180))


class DiagnosticsWildfireEventTest(unittest.TestCase):
    def test_load_wildfire_events_filters_by_city_and_date_overlap(self):
        csv_path = self._write_event_csv(
            "event_name,start_time,end_time,affected_cities,severity,context\n"
            "Creek Fire,2020-09-04,2020-09-30,Fresno;San Jose,Extreme,Smoke episode\n"
            "Southern Event,2020-09-10,2020-09-12,Los Angeles,Moderate,Local smoke\n"
        )

        events = load_wildfire_events(
            csv_path,
            city_name="Fresno",
            start_at=pd.Timestamp("2020-09-01"),
            end_at=pd.Timestamp("2020-09-20"),
        )

        self.assertEqual([event.event_name for event in events], ["Creek Fire"])
        self.assertEqual(events[0].severity, "Extreme")

    def test_list_wildfire_events_filters_by_city_without_date_window(self):
        csv_path = self._write_event_csv(
            "event_name,start_time,end_time,affected_cities,severity,context\n"
            "Creek Fire,2020-09-04,2020-09-30,Fresno;San Jose,Extreme,Smoke episode\n"
            "Bobcat Fire,2020-09-06,2020-11-27,Los Angeles,High,Local smoke\n"
        )

        events = list_wildfire_events(csv_path, city_name="San Jose")

        self.assertEqual([event.event_name for event in events], ["Creek Fire"])

    def test_detect_aqi_peak_episodes_groups_contiguous_high_aqi_hours(self):
        actual = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        "2025-12-01 00:00:00",
                        "2025-12-01 01:00:00",
                        "2025-12-01 02:00:00",
                        "2025-12-01 03:00:00",
                        "2025-12-01 04:00:00",
                    ]
                ),
                "station_id": ["FRES_OPENMETEO"] * 5,
                "actual_aqi": [58.0, 112.0, 135.0, 88.0, 145.0],
            }
        )

        episodes = detect_aqi_peak_episodes(
            actual,
            minimum_aqi=100.0,
            percentile=0.60,
            max_episodes=3,
        )

        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0]["peak_aqi"], 145.0)
        self.assertEqual(episodes[0]["peak_time"], pd.Timestamp("2025-12-01 04:00:00"))
        self.assertEqual(episodes[1]["peak_aqi"], 135.0)

    def test_alignment_figure_can_overlay_wildfire_bands_and_detected_peaks(self):
        csv_path = self._write_event_csv(
            "event_name,start_time,end_time,affected_cities,severity,context\n"
            "Creek Fire,2025-10-31,2025-11-02,Fresno,Extreme,Smoke episode\n"
        )

        figure = build_alignment_figure(
            sample_alignment_frame(),
            ["LightGBM"],
            city_name="Fresno",
            show_wildfire_events=True,
            show_detected_peaks=True,
            wildfire_event_path=csv_path,
        )

        annotation_texts = [annotation.text for annotation in figure.layout.annotations]
        trace_names = [trace.name for trace in figure.data]
        self.assertTrue(any("Creek Fire" in text for text in annotation_texts))
        self.assertIn("Detected AQI Peak", trace_names)

    def test_alignment_figure_with_event_overlays_is_json_serializable(self):
        csv_path = self._write_event_csv(
            "event_name,start_time,end_time,affected_cities,severity,context\n"
            "Creek Fire,2025-10-31,2025-11-02,Fresno,Extreme,Smoke episode\n"
        )

        figure = build_alignment_figure(
            sample_alignment_frame(),
            ["LightGBM"],
            city_name="Fresno",
            show_wildfire_events=True,
            show_detected_peaks=True,
            wildfire_event_path=csv_path,
        )

        self.assertIn("Creek Fire", figure.to_json())

    def _write_event_csv(self, content: str):
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            encoding="utf-8",
        )
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        with handle:
            handle.write(content)
        return Path(handle.name)


class DiagnosticsHistoricalStationTest(unittest.TestCase):
    def test_validation_data_reads_legacy_epa_station_for_fresno_wildfire_window(self):
        db_path = self._create_history_db()
        with sqlite_connection(db_path) as connection:
            connection.execute(
                """
                INSERT INTO meteorology_history
                    (time, station_id, station_name, target_aqi)
                VALUES
                    ('2020-09-05 00:00:00', 'FRES', 'Fresno - Garland', 122.0)
                """
            )
            connection.execute(
                """
                INSERT INTO model_predictions
                    (time, station_id, station_name, city_name, scenario,
                     horizon_hours, model_name, predicted_aqi)
                VALUES
                    ('2020-09-05 00:00:00', 'FRES', 'Fresno - Garland',
                     'Fresno', 'Short-term Nowcasting (1h)', 1, 'LightGBM', 118.0)
                """
            )

        aligned = load_validation_data(
            "Fresno",
            "Short-term Nowcasting (1h)",
            pd.Timestamp("2020-09-01"),
            pd.Timestamp("2020-09-30 23:59:59"),
            ["LightGBM"],
            db_path=db_path,
        )

        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned.iloc[0]["station_id"], "FRES")
        self.assertEqual(aligned.iloc[0]["actual_aqi"], 122.0)

    def test_available_wildfire_events_excludes_city_windows_without_actual_aqi(self):
        event_path = self._write_event_csv(
            "event_name,start_time,end_time,affected_cities,severity,context\n"
            "Mosquito Fire,2022-09-06,2022-10-22,Fresno;San Jose,High,Smoke episode\n"
        )
        db_path = self._create_history_db()
        with sqlite_connection(db_path) as connection:
            connection.execute(
                """
                INSERT INTO meteorology_history
                    (time, station_id, station_name, target_aqi)
                VALUES
                    ('2022-09-10 00:00:00', 'FRES', 'Fresno - Garland', 118.0)
                """
            )

        fresno_events = list_available_wildfire_events(
            city_name="Fresno",
            db_path=db_path,
            event_path=event_path,
        )
        san_jose_events = list_available_wildfire_events(
            city_name="San Jose",
            db_path=db_path,
            event_path=event_path,
        )

        self.assertEqual([event.event_name for event in fresno_events], ["Mosquito Fire"])
        self.assertEqual(san_jose_events, [])

    def _create_history_db(self) -> Path:
        db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = Path(db_file.name)
        db_file.close()
        self.addCleanup(lambda: db_path.unlink(missing_ok=True))
        with sqlite_connection(db_path) as connection:
            connection.executescript(
                """
                CREATE TABLE meteorology_history (
                    time TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    station_name TEXT NOT NULL,
                    target_aqi REAL
                );
                CREATE TABLE model_predictions (
                    time TEXT NOT NULL,
                    station_id TEXT NOT NULL,
                    station_name TEXT NOT NULL,
                    city_name TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    horizon_hours INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    predicted_aqi REAL NOT NULL
                );
                """
            )
        return db_path

    def _write_event_csv(self, content: str) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            encoding="utf-8",
        )
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        with handle:
            handle.write(content)
        return Path(handle.name)


if __name__ == "__main__":
    unittest.main()
