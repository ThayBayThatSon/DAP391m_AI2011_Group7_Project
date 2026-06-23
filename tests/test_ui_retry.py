from __future__ import annotations

import importlib.util
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


class DashboardRetryTest(unittest.TestCase):
    def load_ui_module(self, fake_streamlit, fake_get, fake_post):
        class FakeSession:
            def __init__(self):
                self.trust_env = True

            def post(self, *args, **kwargs):
                return fake_post(*args, **kwargs)

        spec = importlib.util.spec_from_file_location("ui_under_test", UI_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None

        with patch.dict(sys.modules, {"streamlit": fake_streamlit}):
            with patch.object(requests, "get", side_effect=fake_get):
                with patch.object(requests, "post", side_effect=fake_post):
                    with patch.object(requests, "Session", side_effect=FakeSession):
                        with patch("time.sleep", return_value=None):
                            spec.loader.exec_module(module)
        return module

    def test_dashboard_retries_prediction_when_backend_temporarily_unavailable(self):
        fake_streamlit = FakeStreamlit()
        post_calls = {"count": 0}

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

        module = self.load_ui_module(fake_streamlit, fake_get, fake_post)

        self.assertEqual(post_calls["count"], 2)
        self.assertEqual(fake_streamlit.errors, [])
        self.assertFalse(module.PREDICTION_API_SESSION.trust_env)
        self.assertEqual(fake_streamlit.slider_kwargs["value"], 24)


if __name__ == "__main__":
    unittest.main()
