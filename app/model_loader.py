from __future__ import annotations

import logging
from pathlib import Path


logger = logging.getLogger("aqi-model-loader")


def load_lightgbm_booster(model_path: Path):
    import lightgbm as lgb

    model_bytes = model_path.read_bytes()
    if b"\r\n" in model_bytes:
        logger.warning("Normalizing CRLF line endings in LightGBM model %s", model_path)
        model_bytes = model_bytes.replace(b"\r\n", b"\n")
    return lgb.Booster(model_str=model_bytes.decode("utf-8"))
