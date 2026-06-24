from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from app.diagnostics import (
    build_alignment_figure,
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
        self.frame = pd.DataFrame(
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
        category_traces = [
            trace
            for trace in figure.data
            if trace.legendgroup == "aqi-category"
        ]
        category_names = {trace.name for trace in category_traces}
        self.assertIn("Good (0-50)", category_names)
        self.assertIn("Unhealthy (151-200)", category_names)
        self.assertTrue(all(trace.mode == "markers" for trace in category_traces))
        self.assertTrue(all(trace.showlegend is False for trace in category_traces))
        self.assertTrue(all(trace.marker.size == 3 for trace in category_traces))
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
        self.assertEqual(prediction.line.width, 1.5)
        self.assertEqual(prediction.opacity, 0.72)
        self.assertEqual(figure.layout.height, 400)
        self.assertEqual(tuple(figure.layout.yaxis.range), (0, 180))


if __name__ == "__main__":
    unittest.main()
