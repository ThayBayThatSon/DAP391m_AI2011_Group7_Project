from __future__ import annotations

import unittest

from app.forecast_panel import (
    axis_maximum,
    build_forecast_range_panel,
    clamp_aqi,
    forecast_summary,
)


class ForecastRangePanelTest(unittest.TestCase):
    def test_panel_contains_single_forecast_current_and_exact_interval(self):
        figure, summary = build_forecast_range_panel(
            predicted_aqi=73.0,
            confidence_lower=61.0,
            confidence_upper=85.0,
            horizon=1,
            current_aqi=68.0,
            current_is_live=True,
        )

        self.assertEqual(tuple(figure.layout.xaxis.range), (0, 100))
        self.assertEqual(figure.layout.height, 260)
        self.assertEqual(len(figure.layout.shapes), 6)

        forecast = next(
            trace for trace in figure.data if trace.name == "Forecast +1h"
        )
        current = next(trace for trace in figure.data if trace.name == "Current")
        interval = next(
            trace
            for trace in figure.data
            if trace.name == "Confidence interval"
        )

        self.assertEqual(list(forecast.x), [73.0])
        self.assertEqual(forecast.marker.symbol, "diamond")
        self.assertEqual(list(current.x), [68.0])
        self.assertEqual(current.marker.symbol, "circle")
        self.assertEqual(list(interval.x), [61.0, 85.0])
        self.assertEqual(len(forecast.x), 1)
        self.assertIn("+5.0 AQI", summary)
        self.assertIn("stays Moderate", summary)

    def test_stale_or_missing_current_aqi_does_not_render_current_marker(self):
        for current_aqi, current_is_live in ((68.0, False), (None, False)):
            with self.subTest(
                current_aqi=current_aqi,
                current_is_live=current_is_live,
            ):
                figure, summary = build_forecast_range_panel(
                    predicted_aqi=73.0,
                    confidence_lower=61.0,
                    confidence_upper=85.0,
                    horizon=24,
                    current_aqi=current_aqi,
                    current_is_live=current_is_live,
                )
                names = [trace.name for trace in figure.data]
                self.assertNotIn("Current", names)
                self.assertIn("Forecast +24h", names)
                self.assertNotIn("vs current", summary)
                self.assertIn("CI 61.0-85.0 AQI", summary)

    def test_axis_breakpoints_and_clamping_use_us_aqi_domain(self):
        self.assertEqual(axis_maximum(30.0, 90.0), 100)
        self.assertEqual(axis_maximum(101.0, 180.0), 200)
        self.assertEqual(axis_maximum(201.0, 290.0), 300)
        self.assertEqual(axis_maximum(301.0, 600.0), 500)
        self.assertEqual(clamp_aqi(-12.0), 0.0)
        self.assertEqual(clamp_aqi(720.0), 500.0)

        figure, _ = build_forecast_range_panel(
            predicted_aqi=720.0,
            confidence_lower=-25.0,
            confidence_upper=900.0,
            horizon=24,
            current_aqi=None,
            current_is_live=False,
        )
        forecast = next(
            trace for trace in figure.data if trace.name == "Forecast +24h"
        )
        interval = next(
            trace
            for trace in figure.data
            if trace.name == "Confidence interval"
        )
        self.assertEqual(list(forecast.x), [500.0])
        self.assertEqual(list(interval.x), [0.0, 500.0])
        self.assertEqual(tuple(figure.layout.xaxis.range), (0, 500))

    def test_summary_reports_category_direction(self):
        worsened = forecast_summary(
            current_aqi=45.0,
            predicted_aqi=110.0,
            confidence_lower=90.0,
            confidence_upper=135.0,
            current_is_live=True,
        )
        improved = forecast_summary(
            current_aqi=165.0,
            predicted_aqi=80.0,
            confidence_lower=60.0,
            confidence_upper=105.0,
            current_is_live=True,
        )
        unchanged = forecast_summary(
            current_aqi=65.0,
            predicted_aqi=85.0,
            confidence_lower=55.0,
            confidence_upper=98.0,
            current_is_live=True,
        )

        self.assertIn("worsens from Good to Unhealthy for Sensitive Groups", worsened)
        self.assertIn("improves from Unhealthy to Moderate", improved)
        self.assertIn("stays Moderate", unchanged)


if __name__ == "__main__":
    unittest.main()
