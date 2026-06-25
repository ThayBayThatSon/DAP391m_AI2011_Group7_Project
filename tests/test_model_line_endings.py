from pathlib import Path
import tempfile
import unittest

from app.model_loader import load_lightgbm_booster


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = (
    PROJECT_ROOT / "models" / "lightgbm_nowcast.txt",
    PROJECT_ROOT / "models" / "lightgbm_forecast24h.txt",
    PROJECT_ROOT / "models" / "nowcast_1h" / "lightgbm.txt",
    PROJECT_ROOT / "models" / "forecast_24h" / "lightgbm.txt",
)


class ModelLineEndingTests(unittest.TestCase):
    def test_lightgbm_models_preserve_lf_line_endings(self):
        for model_path in MODEL_PATHS:
            with self.subTest(model=model_path):
                self.assertNotIn(
                    b"\r\n",
                    model_path.read_bytes(),
                    "LightGBM native model files must not be converted to CRLF.",
                )

    def test_git_attributes_disable_text_conversion_for_model_artifacts(self):
        attributes = (PROJECT_ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("models/**/*.txt -text", attributes)
        self.assertIn("models/*.txt -text", attributes)

    def test_loader_repairs_crlf_model_before_lightgbm_parses_it(self):
        source = PROJECT_ROOT / "models" / "lightgbm_forecast24h.txt"
        corrupted_bytes = source.read_bytes().replace(b"\n", b"\r\n")

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "windows-checkout-model.txt"
            model_path.write_bytes(corrupted_bytes)

            booster = load_lightgbm_booster(model_path)

        self.assertEqual(booster.num_trees(), 492)
        self.assertEqual(booster.num_feature(), 49)


if __name__ == "__main__":
    unittest.main()
