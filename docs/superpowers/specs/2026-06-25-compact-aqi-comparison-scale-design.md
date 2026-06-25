# Compact AQI Comparison Scale Design

## Goal

Refine the Live Forecast visualization into a compact comparison scale that
communicates current AQI, forecast AQI, uncertainty, and health categories
without chart conventions that duplicate information or imply additional data.

## Navigation Labels

Rename the dashboard tabs:

- `Live Forecast` becomes `AQI Forecast`
- `Live Validation & Model Diagnostics` becomes `Model Validation`

These labels are shorter and describe the user task rather than the underlying
implementation.

## Comparison Scale

- Remove the Plotly legend entirely.
- Reduce the figure height from 260 pixels to approximately 180 pixels.
- Render the visible US AQI health categories as one thin horizontal colored
  scale.
- Write category names directly inside or immediately above each visible band.
- Render the confidence interval as a translucent horizontal rectangle rather
  than a thick line.
- Render Current AQI as a vertical circular-ended marker line.
- Render Forecast AQI as a vertical diamond-ended marker line.
- Position Current and Forecast labels on opposite sides of the scale so nearby
  values do not overlap.
- Keep direct hover values for both markers.

## Summary

Retain the existing forecast comparison summary above the scale. It remains the
primary textual explanation of:

- signed AQI change
- category direction
- confidence interval

The scale should complement this summary rather than repeat it through a
legend.

## Responsive Behavior

- The figure uses the full available width.
- Category labels may be abbreviated at narrow widths:
  - `Sensitive Groups`
  - `Very Unhealthy`
- Marker labels must remain inside the figure margins.
- The page must not gain horizontal overflow at 720 pixels.

## Implementation Boundary

- Update `app/forecast_panel.py` figure construction.
- Update the tab labels in `app/ui.py`.
- Update tests that assert the old tab names or legend structure.
- Do not change forecast values, Current AQI retrieval, confidence intervals,
  category thresholds, model inference, or diagnostics content.

## Testing

- Verify `showlegend` is false.
- Verify figure height is 180 pixels.
- Verify the confidence interval is a translucent rectangle with exact bounds.
- Verify category labels are annotations rather than legend entries.
- Verify Current and Forecast use vertical marker lines with distinct end
  symbols and non-overlapping label positions.
- Verify the new tab names.
- Run the full test suite and inspect desktop and narrow layouts.

## Acceptance Criteria

- No legend appears.
- The confidence interval no longer dominates the scale.
- Current and Forecast values are distinguishable even when close together.
- The scale occupies less vertical space than the previous panel.
- Navigation displays `AQI Forecast` and `Model Validation`.
