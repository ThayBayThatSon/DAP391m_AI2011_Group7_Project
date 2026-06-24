from __future__ import annotations

import unittest

from app.alerts import evaluate_air_stagnation_alert


class AirStagnationAlertTest(unittest.TestCase):
    def test_moderate_aqi_uses_advisory_without_extreme_event_claims(self):
        alert = evaluate_air_stagnation_alert(
            wind_speed_mps=1.3,
            relative_humidity_pct=32.0,
            rain_mm=0.0,
            predicted_aqi=59.0,
        )

        self.assertIsNotNone(alert)
        self.assertEqual(alert.level, "advisory")
        self.assertIn("Low-dispersion conditions", alert.title)
        self.assertNotIn("99th percentile", alert.message)
        self.assertNotIn("Wildfire", alert.message)
        self.assertNotIn("CRITICAL", alert.title)

    def test_unhealthy_aqi_uses_critical_warning(self):
        alert = evaluate_air_stagnation_alert(
            wind_speed_mps=1.0,
            relative_humidity_pct=40.0,
            rain_mm=0.0,
            predicted_aqi=165.0,
        )

        self.assertIsNotNone(alert)
        self.assertEqual(alert.level, "critical")
        self.assertIn("Critical air quality risk", alert.title)
        self.assertIn("165", alert.message)

    def test_old_broad_wind_threshold_no_longer_triggers(self):
        alert = evaluate_air_stagnation_alert(
            wind_speed_mps=4.5,
            relative_humidity_pct=30.0,
            rain_mm=0.0,
            predicted_aqi=80.0,
        )
        self.assertIsNone(alert)

    def test_meaningful_rain_suppresses_stagnation_alert(self):
        alert = evaluate_air_stagnation_alert(
            wind_speed_mps=1.0,
            relative_humidity_pct=40.0,
            rain_mm=1.0,
            predicted_aqi=120.0,
        )
        self.assertIsNone(alert)


if __name__ == "__main__":
    unittest.main()
