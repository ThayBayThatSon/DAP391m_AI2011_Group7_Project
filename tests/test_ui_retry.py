from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_PATH = PROJECT_ROOT / "app" / "ui.py"


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Server Error", response=self)


class FakePanel:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def metric(self, *args, **kwargs):
        return None


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.errors: list[str] = []
        self.slider_kwargs: dict = {}

    def cache_data(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def cache_resource(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def set_page_config(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def columns(self, *args, **kwargs):
        spec = args[0] if args else 1
        count = len(spec) if isinstance(spec, list) else int(spec)
        return [FakePanel() for _ in range(count)]

    def selectbox(self, *args, **kwargs):
        return "Fresno"

    def slider(self, *args, **kwargs):
        self.slider_kwargs = kwargs
        return kwargs.get("value", 1)

    def map(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def altair_chart(self, *args, **kwargs):
        return None

    def error(self, message):
        self.errors.append(str(message))

    def spinner(self, *args, **kwargs):
        return FakePanel()


class FakeSession:
    def __init__(self):
        self.trust_env = True
        self.post_handler = None

    def post(self, *args, **kwargs):
        if self.post_handler is None:
            raise AssertionError("Unexpected HTTP POST in local Streamlit mode.")
        return self.post_handler(*args, **kwargs)


class DashboardPredictionModeTest(unittest.TestCase):
    def test_dashboard_defaults_to_local_prediction_and_http_retry_remains_available(self):
        fake_streamlit = FakeStreamlit()

        class FakePredictionRequest:
            def __init__(self, **kwargs):
                self.payload = kwargs

        class FakePredictionResponse:
            def model_dump(self):
                return {
                    "predicted_aqi": 65.68,
                    "model_horizon": "Long-Term 24-Hour Forecasting (t+24)",
                    "confidence_interval": {
                        "lower": 39.12,
                        "upper": 92.24,
                        "mae_baseline": 13.55,
                    },
                }

        fake_main = types.ModuleType("app.main")
        fake_main.startup_called = 0

        def fake_startup():
            fake_main.startup_called += 1

        def fake_predict(request):
            self.assertEqual(request.payload["target_hour_ahead"], 24)
            return FakePredictionResponse()

        fake_main.PredictionRequest = FakePredictionRequest
        fake_main.predict = fake_predict
        fake_main.startup = fake_startup

        def fake_get(*args, **kwargs):
            return FakeResponse(
                200,
                {
                    "hourly": {
                        "time": ["2026-06-23T05:00"],
                        "temperature_2m": [24.5],
                        "relative_humidity_2m": [42.0],
                        "wind_speed_10m": [4.8],
                        "wind_direction_10m": [270.0],
                        "surface_pressure": [1008.2],
                        "rain": [0.0],
                        "cloud_cover": [15.0],
                    }
                },
            )

        spec = importlib.util.spec_from_file_location("ui_under_test", UI_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None

        with patch.dict(os.environ, {"HOME": str(PROJECT_ROOT), "USERPROFILE": str(PROJECT_ROOT)}, clear=True):
            with patch.dict(sys.modules, {"streamlit": fake_streamlit, "app.main": fake_main}):
                with patch.object(requests, "get", side_effect=fake_get):
                    with patch.object(requests, "Session", side_effect=FakeSession):
                        spec.loader.exec_module(module)

        self.assertFalse(module.USE_REMOTE_API)
        self.assertFalse(module.PREDICTION_API_SESSION.trust_env)
        self.assertEqual(module.PREDICTION_ENGINE.startup_called, 1)
        self.assertEqual(fake_streamlit.slider_kwargs["value"], 24)
        self.assertEqual(fake_streamlit.errors, [])

        post_calls = {"count": 0}

        def fake_post(*args, **kwargs):
            self.assertEqual(kwargs["json"]["target_hour_ahead"], 24)
            post_calls["count"] += 1
            if post_calls["count"] == 1:
                return FakeResponse(503, {"detail": "Backend is starting"})
            return FakeResponse(
                200,
                {
                    "predicted_aqi": 75.59,
                    "model_horizon": "Long-Term 24-Hour Forecasting (t+24)",
                    "confidence_interval": {
                        "lower": 49.03,
                        "upper": 102.15,
                        "mae_baseline": 13.55,
                    },
                },
            )

        module.API_BASE_URL = "http://127.0.0.1:8000"
        module.USE_REMOTE_API = True
        module.PREDICTION_API_SESSION.post_handler = fake_post
        with patch("time.sleep", return_value=None):
            result = module.call_prediction_api(
                "Fresno",
                24,
                module.parse_open_meteo_time("2026-06-23T05:00"),
                {
                    "temperature_2m": 24.5,
                    "relative_humidity_2m": 42.0,
                    "wind_speed_10m": 4.8,
                    "wind_direction_10m": 270.0,
                    "surface_pressure": 1008.2,
                    "rain": 0.0,
                    "cloud_cover": 15.0,
                },
            )

        self.assertEqual(post_calls["count"], 2)
        self.assertEqual(result["predicted_aqi"], 75.59)


if __name__ == "__main__":
    unittest.main()
