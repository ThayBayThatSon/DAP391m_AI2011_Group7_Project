from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AirQualityAlert:
    level: str
    title: str
    message: str


def evaluate_air_stagnation_alert(
    wind_speed_mps: float,
    relative_humidity_pct: float,
    rain_mm: float,
    predicted_aqi: float,
) -> AirQualityAlert | None:
    low_dispersion = (
        float(wind_speed_mps) < 2.0
        and float(relative_humidity_pct) < 55.0
        and float(rain_mm) <= 0.1
    )
    if not low_dispersion:
        return None

    aqi = max(float(predicted_aqi), 0.0)
    if aqi >= 151.0:
        return AirQualityAlert(
            level="critical",
            title="Critical air quality risk",
            message=(
                f"Predicted AQI is {aqi:.0f}. Very low wind and dry conditions "
                "may limit pollutant dispersion. Reduce prolonged outdoor "
                "activity and follow local air-quality guidance."
            ),
        )
    if aqi >= 101.0:
        return AirQualityAlert(
            level="warning",
            title="Elevated pollution risk",
            message=(
                f"Predicted AQI is {aqi:.0f}. Low wind, dry conditions, and "
                "little rain may allow pollution to accumulate. Sensitive "
                "groups should monitor updates."
            ),
        )
    return AirQualityAlert(
        level="advisory",
        title="Low-dispersion conditions",
        message=(
            f"Wind is light and conditions are dry, but the predicted AQI is "
            f"{aqi:.0f} and does not currently indicate unhealthy air. "
            "Monitor updates if conditions persist."
        ),
    )
