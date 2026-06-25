from __future__ import annotations

import plotly.graph_objects as go


AQI_BANDS = (
    (0.0, 50.0, "Good", "#16a34a"),
    (50.0, 100.0, "Moderate", "#eab308"),
    (100.0, 150.0, "Unhealthy for Sensitive Groups", "#f97316"),
    (150.0, 200.0, "Unhealthy", "#dc2626"),
    (200.0, 300.0, "Very Unhealthy", "#7e22ce"),
    (300.0, 500.0, "Hazardous", "#7f1d1d"),
)


def clamp_aqi(value: float) -> float:
    return min(max(float(value), 0.0), 500.0)


def axis_maximum(*values: float) -> int:
    maximum = max((clamp_aqi(value) for value in values), default=100.0)
    for breakpoint in (100, 200, 300, 500):
        if maximum <= breakpoint:
            return breakpoint
    return 500


def aqi_category(value: float) -> tuple[str, str, int]:
    clamped = clamp_aqi(value)
    for index, (lower, upper, name, color) in enumerate(AQI_BANDS):
        if lower <= clamped <= upper or (
            index > 0 and lower < clamped <= upper
        ):
            return name, color, index
    return AQI_BANDS[-1][2], AQI_BANDS[-1][3], len(AQI_BANDS) - 1


def forecast_summary(
    *,
    current_aqi: float | None,
    predicted_aqi: float,
    confidence_lower: float,
    confidence_upper: float,
    current_is_live: bool,
) -> str:
    predicted = clamp_aqi(predicted_aqi)
    lower, upper = sorted(
        (clamp_aqi(confidence_lower), clamp_aqi(confidence_upper))
    )
    forecast_name, _, forecast_rank = aqi_category(predicted)
    interval_text = f"CI {lower:.1f}-{upper:.1f} AQI"

    if not current_is_live or current_aqi is None:
        return f"Forecast category: {forecast_name} | {interval_text}"

    current = clamp_aqi(current_aqi)
    current_name, _, current_rank = aqi_category(current)
    change = predicted - current
    if forecast_rank > current_rank:
        direction = f"worsens from {current_name} to {forecast_name}"
    elif forecast_rank < current_rank:
        direction = f"improves from {current_name} to {forecast_name}"
    else:
        direction = f"stays {forecast_name}"
    return f"{change:+.1f} AQI vs current | Category {direction} | {interval_text}"


def build_forecast_range_panel(
    *,
    predicted_aqi: float,
    confidence_lower: float,
    confidence_upper: float,
    horizon: int,
    current_aqi: float | None,
    current_is_live: bool,
) -> tuple[go.Figure, str]:
    predicted = clamp_aqi(predicted_aqi)
    lower, upper = sorted(
        (clamp_aqi(confidence_lower), clamp_aqi(confidence_upper))
    )
    live_current = (
        clamp_aqi(current_aqi)
        if current_is_live and current_aqi is not None
        else None
    )
    axis_max = axis_maximum(
        predicted,
        upper,
        *(tuple() if live_current is None else (live_current,)),
    )
    forecast_name, forecast_color, _ = aqi_category(predicted)

    figure = go.Figure()
    for band_lower, band_upper, band_name, band_color in AQI_BANDS:
        figure.add_shape(
            type="rect",
            x0=band_lower,
            x1=band_upper,
            y0=-0.45,
            y1=0.45,
            fillcolor=band_color,
            opacity=0.14,
            line={"width": 0},
            layer="below",
            name=band_name,
        )

    figure.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[0.0, 0.0],
            mode="lines",
            name="Confidence interval",
            line={"color": "#475569", "width": 12},
            hovertemplate="Confidence interval: %{x:.1f} AQI<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[predicted],
            y=[0.0],
            mode="markers+text",
            name=f"Forecast +{horizon}h",
            marker={
                "color": forecast_color,
                "size": 18,
                "symbol": "diamond",
                "line": {"color": "#ffffff", "width": 2},
            },
            text=[f"Forecast +{horizon}h<br>{predicted:.1f}"],
            textposition="top center",
            hovertemplate=(
                f"Forecast +{horizon}h: %{{x:.1f}} AQI"
                f"<br>{forecast_name}<extra></extra>"
            ),
        )
    )

    if live_current is not None:
        current_name, current_color, _ = aqi_category(live_current)
        figure.add_trace(
            go.Scatter(
                x=[live_current],
                y=[0.0],
                mode="markers+text",
                name="Current",
                marker={
                    "color": current_color,
                    "size": 15,
                    "symbol": "circle",
                    "line": {"color": "#ffffff", "width": 2},
                },
                text=[f"Current<br>{live_current:.1f}"],
                textposition="bottom center",
                hovertemplate=(
                    f"Current: %{{x:.1f}} AQI"
                    f"<br>{current_name}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        height=260,
        margin={"l": 20, "r": 20, "t": 58, "b": 48},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#94a3b8"},
        hovermode="closest",
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "bgcolor": "rgba(0,0,0,0)",
        },
        xaxis={
            "title": "US AQI",
            "range": [0, axis_max],
            "showgrid": True,
            "gridcolor": "rgba(148,163,184,0.18)",
            "zeroline": False,
            "tickmode": "linear",
            "dtick": 25 if axis_max <= 200 else 50,
        },
        yaxis={
            "range": [-0.75, 0.75],
            "visible": False,
            "fixedrange": True,
        },
    )

    summary = forecast_summary(
        current_aqi=live_current,
        predicted_aqi=predicted,
        confidence_lower=lower,
        confidence_upper=upper,
        current_is_live=live_current is not None,
    )
    return figure, summary
