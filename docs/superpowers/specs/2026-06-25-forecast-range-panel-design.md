# Forecast Range Panel Design

## Goal

Replace the Live Forecast line chart with a visualization that accurately
represents the model output: one AQI prediction at either t+1 or t+24 with a
confidence interval.

## Problem

The existing chart repeats one prediction across every intermediate hour and
expands the interval gradually. The model does not produce those intermediate
predictions, so the line implies a forecast trajectory that does not exist.

## Visualization

Use a compact horizontal AQI range panel:

- The x-axis uses the US AQI scale.
- Background bands identify the six US AQI health categories:
  - Good: 0-50
  - Moderate: 51-100
  - Unhealthy for Sensitive Groups: 101-150
  - Unhealthy: 151-200
  - Very Unhealthy: 201-300
  - Hazardous: 301-500
- A thick horizontal interval shows the returned confidence interval.
- A prominent marker shows the predicted AQI at the selected horizon.
- A second marker shows Current AQI only when a live current reading is
  available.
- The predicted marker label states `Forecast +1h` or `Forecast +24h`.
- The current marker label states `Current`.

## Summary Text

Above or directly below the panel, show:

- Signed AQI change from current to predicted when live Current AQI exists.
- Current and forecast health categories.
- Whether the category improves, worsens, or stays unchanged.
- The confidence interval as explicit numeric text.

If Current AQI is unavailable or only a stale SQLite fallback exists, omit the
current marker and comparison statement. Continue to show the forecast marker,
forecast category, and confidence interval.

## Axis Range

The displayed maximum should be the smallest suitable standard breakpoint that
contains the prediction and upper confidence bound:

- 100 when all displayed values are at or below 100
- 200 when all displayed values are at or below 200
- 300 when all displayed values are at or below 300
- 500 otherwise

The minimum remains zero. Values and confidence bounds must be clamped to the
US AQI domain of 0-500.

## Styling

- Match the dashboard's restrained border and dark-theme-compatible treatment.
- Keep category colors recognizable but use low-opacity background bands.
- Use a dark neutral interval line so it remains distinct from category bands.
- Use a diamond marker for the forecast and a circle marker for current AQI.
- Keep the panel around 240-280 pixels high.
- Preserve responsive width without horizontal page overflow.

## Implementation Boundary

- Remove `timeline_frame` and `render_timeline_chart` from `app/ui.py`.
- Build the panel using Plotly in a pure helper so figure structure is directly
  testable.
- Do not change model inference, confidence interval calculation, Current AQI
  retrieval, alerts, or diagnostics.

## Testing

- Verify one forecast marker is rendered at the predicted value.
- Verify the confidence interval uses the returned lower and upper values.
- Verify Current AQI is rendered only for a live reading.
- Verify no intermediate hourly forecast points are generated.
- Verify dynamic axis breakpoints and clamping.
- Verify summary text for improved, worsened, and unchanged categories.
- Run the full test suite and inspect the running dashboard at desktop and
  720-pixel widths.

## Acceptance Criteria

- The visualization no longer implies hourly intermediate predictions.
- Users can immediately compare Current AQI, forecast AQI, and uncertainty.
- Health-category context is visible without overwhelming the chart.
- The panel remains useful when Current AQI is unavailable.
