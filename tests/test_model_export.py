from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = PROJECT_ROOT / "scr" / "train_combined_panel_models.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location("train_combined_panel_models", TRAINING_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ModelExportTest(unittest.TestCase):
    def test_production_model_paths_and_exact_forecast_lags(self):
        training = load_training_module()
        nowcast_path = Path(training.MODEL_OUTPUT_PATHS["Short-term Autoregressive (Lag 1-3h)"]).as_posix()
        forecast_path = Path(training.MODEL_OUTPUT_PATHS["Long-term Forecasting (Lag 24h)"]).as_posix()

        self.assertTrue(nowcast_path.endswith("models/lightgbm_nowcast.txt"))
        self.assertTrue(forecast_path.endswith("models/lightgbm_forecast24h.txt"))

        long_config = training.CONFIGURATIONS["Long-term Forecasting (Lag 24h)"]
        self.assertEqual(long_config["target_lags"], [24, 48, 72])
        self.assertEqual(long_config["spatial_lags"], [24, 48, 72])
        self.assertEqual(long_config["rolling_windows"], [24, 72])

    def test_lightgbm_production_matrices_keep_raw_numeric_values_and_ohe_categories(self):
        training = load_training_module()
        x_train = pd.DataFrame(
            {
                "temperature_2m": [10.0, 20.0],
                "relative_humidity_2m": [60.0, 70.0],
                "source": ["openmeteo", "epa_aqs_station"],
                "station_id": ["FRES_OPENMETEO", "LA_OPENMETEO"],
            }
        )
        x_val = pd.DataFrame(
            {
                "temperature_2m": [30.0],
                "relative_humidity_2m": [80.0],
                "source": ["openmeteo"],
                "station_id": ["FRES_OPENMETEO"],
            }
        )
        x_test = pd.DataFrame(
            {
                "temperature_2m": [40.0],
                "relative_humidity_2m": [90.0],
                "source": ["new_source"],
                "station_id": ["NEW_STATION"],
            }
        )

        train_matrix, val_matrix, test_matrix = training.prepare_lightgbm_production_matrices(x_train, x_val, x_test)

        self.assertEqual(train_matrix.loc[0, "temperature_2m"], 10.0)
        self.assertEqual(train_matrix.loc[1, "relative_humidity_2m"], 70.0)
        self.assertIn("source_openmeteo", train_matrix.columns)
        self.assertIn("station_id_FRES_OPENMETEO", train_matrix.columns)
        self.assertEqual(val_matrix.columns.tolist(), train_matrix.columns.tolist())
        self.assertEqual(test_matrix.columns.tolist(), train_matrix.columns.tolist())
        self.assertTrue(test_matrix.filter(like="new_source").empty)
        self.assertTrue(test_matrix.filter(like="NEW_STATION").empty)

    def test_lightgbm_only_mode_limits_training_to_export_model(self):
        training = load_training_module()

        self.assertEqual(list(training.model_dictionary(lightgbm_only=True)), ["LightGBM"])

    def test_prediction_report_preserves_station_identity_and_city(self):
        training = load_training_module()
        timestamps = pd.to_datetime(
            ["2025-01-01 00:00:00", "2025-01-01 01:00:00"]
        )
        y_true = pd.Series([50.0, 75.0], index=timestamps)
        predictions = {
            "LightGBM": (
                y_true,
                pd.Series([48.0, 78.0], index=timestamps),
            )
        }

        report = training.build_prediction_report(
            "Short-term Autoregressive (Lag 1-3h)",
            predictions,
            pd.Series(["FRES_OPENMETEO", "SJ_OPENMETEO"]),
            {
                "FRES_OPENMETEO": "Fresno - Open-Meteo",
                "SJ_OPENMETEO": "Downtown San Jose, California",
            },
        )

        self.assertEqual(
            report["station_id"].tolist(),
            ["FRES_OPENMETEO", "SJ_OPENMETEO"],
        )
        self.assertEqual(report["city_name"].tolist(), ["Fresno", "San Jose"])
        self.assertEqual(report["Model"].tolist(), ["LightGBM", "LightGBM"])


if __name__ == "__main__":
    unittest.main()
