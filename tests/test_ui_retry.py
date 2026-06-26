from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
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
    def __init__(self, owner=None):
        self.owner = owner

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def markdown(self, body, *args, **kwargs):
        if self.owner is not None:
            self.owner.markdown_calls.append(str(body))

    def metric(self, *args, **kwargs):
        return None

    def selectbox(self, label, options, **kwargs):
        return options[0]

    def radio(self, label, options, **kwargs):
        return options[0]

    def warning(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def checkbox(self, label, value=False, **kwargs):
        if self.owner is not None:
            self.owner.checkbox_calls.append(
                {"label": label, "value": value, "kwargs": kwargs}
            )
        return value


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.slider_kwargs: dict = {}
        self.segmented_control_kwargs: dict = {}
        self.segmented_control_calls: list[dict] = []
        self.tab_labels: list[str] = []
        self.markdown_calls: list[str] = []
        self.plotly_chart_calls: list[dict] = []
        self.checkbox_calls: list[dict] = []

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

    def markdown(self, body, *args, **kwargs):
        self.markdown_calls.append(str(body))

    def columns(self, *args, **kwargs):
        spec = args[0] if args else 1
        count = len(spec) if isinstance(spec, list) else int(spec)
        return [FakePanel(self) for _ in range(count)]

    def tabs(self, labels):
        self.tab_labels = list(labels)
        return [FakePanel(self) for _ in labels]

    def selectbox(self, *args, **kwargs):
        return "Fresno"

    def radio(self, label, options, **kwargs):
        return options[0]

    def date_input(self, *args, **kwargs):
        return kwargs["value"]

    def multiselect(self, *args, **kwargs):
        return kwargs.get("default", [])

    def segmented_control(self, label, options, **kwargs):
        self.segmented_control_kwargs = kwargs
        self.segmented_control_calls.append(
            {"label": label, "options": list(options), "kwargs": kwargs}
        )
        return kwargs.get("default", options[0])

    def expander(self, *args, **kwargs):
        return FakePanel(self)

    def slider(self, *args, **kwargs):
        self.slider_kwargs = kwargs
        return kwargs.get("value", 1)

    def map(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def altair_chart(self, *args, **kwargs):
        return None

    def plotly_chart(self, figure, *args, **kwargs):
        self.plotly_chart_calls.append(
            {"figure": figure, "args": args, "kwargs": kwargs}
        )

    def dataframe(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, message):
        self.warnings.append(str(message))

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
    def test_ui_bootstraps_project_root_before_app_imports(self):
        tree = ast.parse(UI_PATH.read_text(encoding="utf-8"))
        app_import_index = next(
            index
            for index, node in enumerate(tree.body)
            if isinstance(node, ast.ImportFrom) and node.module == "app.diagnostics"
        )
        bootstrap_index = next(
            index
            for index, node in enumerate(tree.body)
            if isinstance(node, ast.If)
            and "sys.path" in ast.unparse(node.test)
        )
        self.assertLess(bootstrap_index, app_import_index)

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

        fake_diagnostics = types.ModuleType("app.diagnostics")
        fake_diagnostics.DEFAULT_DB_PATH = PROJECT_ROOT / "test-aqi-data.db"
        fake_diagnostics.DEFAULT_PREDICTION_PATH = (
            PROJECT_ROOT / "test-model-predictions.csv"
        )
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
        fake_diagnostics.ensure_prediction_data = (
            lambda prediction_path, db_path: 0
        )
        fake_diagnostics.resolve_historical_window = (
            lambda start_date, end_date, quick_range: (
                pd.Timestamp(start_date),
                pd.Timestamp(end_date)
                + pd.Timedelta(days=1)
                - pd.Timedelta(seconds=1),
            )
        )
        fake_diagnostics.load_validation_data = (
            lambda *args, **kwargs: pd.DataFrame()
        )
        fake_diagnostics.calculate_model_metrics = (
            lambda frame: pd.DataFrame()
        )
        fake_diagnostics.build_alignment_figure = (
            lambda frame, models, **kwargs: object()
        )

        fake_current_aqi = types.ModuleType("app.current_aqi")

        class FakeAQIReading:
            def __init__(
                self,
                value,
                observed_at,
                label,
                source,
                is_current,
            ):
                self.value = value
                self.observed_at = observed_at
                self.label = label
                self.source = source
                self.is_current = is_current

        fake_current_aqi.AQIReading = FakeAQIReading
        fake_current_aqi.resolve_current_aqi = lambda **kwargs: FakeAQIReading(
            value=57.0,
            observed_at=datetime(2026, 6, 25, 8, 15, tzinfo=timezone.utc),
            label="Current AQI",
            source="Open-Meteo Air Quality",
            is_current=True,
        )

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
            with patch.dict(
                sys.modules,
                {
                    "streamlit": fake_streamlit,
                    "app.main": fake_main,
                    "app.diagnostics": fake_diagnostics,
                    "app.current_aqi": fake_current_aqi,
                },
            ):
                with patch.object(requests, "get", side_effect=fake_get):
                    with patch.object(requests, "Session", side_effect=FakeSession):
                        spec.loader.exec_module(module)

        self.assertFalse(module.USE_REMOTE_API)
        self.assertFalse(module.PREDICTION_API_SESSION.trust_env)
        self.assertEqual(module.PREDICTION_ENGINE.startup_called, 1)
        forecast_horizon_control = next(
            call
            for call in fake_streamlit.segmented_control_calls
            if call["label"] == "Forecast Horizon"
        )
        self.assertEqual(forecast_horizon_control["options"], [1, 24])
        self.assertEqual(forecast_horizon_control["kwargs"]["default"], 24)
        self.assertEqual(
            fake_streamlit.tab_labels,
            ["AQI Forecast", "Model Validation"],
        )
        self.assertEqual(
            fake_streamlit.segmented_control_kwargs["default"],
            "30 Days",
        )
        self.assertEqual(
            [call["label"] for call in fake_streamlit.checkbox_calls],
            [
                *list(fake_diagnostics.MODEL_NAMES),
                "Show wildfire / smoke event markers",
                "Show detected AQI peak episodes",
            ],
        )
        self.assertEqual(fake_streamlit.errors, [])

        rendered_markup = "\n".join(fake_streamlit.markdown_calls)
        self.assertNotIn("Visible Models", rendered_markup)
        self.assertIn(".aqi-accent-strip", rendered_markup)
        self.assertIn(".metric-card-grid", rendered_markup)
        self.assertIn('data-testid="stPlotlyChart"', rendered_markup)
        self.assertIn("border: 1px solid", rendered_markup)
        self.assertIn("var(--text-color)", rendered_markup)
        self.assertIn("var(--secondary-background-color)", rendered_markup)
        self.assertNotIn(".aqi-title {\n        color: #f8fafc;", rendered_markup)
        current_index = rendered_markup.index("Current AQI")
        predicted_index = rendered_markup.index("Predicted AQI")
        self.assertLess(current_index, predicted_index)
        self.assertIn("Open-Meteo Air Quality", rendered_markup)
        self.assertIn("+8.7 vs current", rendered_markup)
        self.assertIn("metric-meta", rendered_markup)
        self.assertGreaterEqual(len(fake_streamlit.plotly_chart_calls), 1)
        live_figure = fake_streamlit.plotly_chart_calls[0]["figure"]
        self.assertIn(
            "Forecast +24h",
            [trace.name for trace in live_figure.data],
        )
        self.assertNotIn("Forecast Hour", str(live_figure))
        self.assertIn("Forecast comparison", rendered_markup)
        for category_color in (
            "#16a34a",
            "#eab308",
            "#f97316",
            "#dc2626",
            "#7e22ce",
            "#7f1d1d",
        ):
            self.assertIn(category_color, rendered_markup)

        module.render_metric_cards(
            pd.DataFrame(
                [
                    {
                        "model_name": model_name,
                        "relative_accuracy": 90.0,
                        "r2": 0.8,
                    }
                    for model_name in fake_diagnostics.MODEL_NAMES
                ]
            )
        )
        rendered_markup = "\n".join(fake_streamlit.markdown_calls)
        for model_accent in (
            "#3b82f6",
            "#f59e0b",
            "#a855f7",
            "#a16207",
            "#14b8a6",
        ):
            self.assertIn(model_accent, rendered_markup)

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
