# Dashboard Visual Polish Design

## Goal

Improve visual hierarchy in the Streamlit dashboard without making it feel
decorative or marketing-oriented. The result should remain a restrained,
professional data product that is easier to scan on the existing dark theme.

## Scope

This change is limited to presentation in `app/ui.py`. It does not alter
forecasting logic, model selection, API requests, database queries, chart data,
or alert thresholds.

## Visual System

- Keep the current dark neutral page background.
- Add a slim AQI category accent strip beneath the page introduction using the
  standard Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very
  Unhealthy, and Hazardous colors.
- Use one-pixel neutral borders and a maximum eight-pixel radius for primary
  controls, metric cards, chart containers, tables, and expandable filter
  panels.
- Avoid gradients, large filled color panels, nested cards, decorative shapes,
  and excessive shadows.
- Use subdued hover and focus states so controls remain accessible without
  introducing visual noise.

## Live Forecast

- Give the station selector, forecast horizon selector, and map a shared
  low-contrast border treatment.
- Style the four forecast metric cards as a coherent row with a neutral surface.
- Apply an AQI-category accent to the Predicted AQI card.
- Use distinct contextual accents for Temperature, Humidity, and Wind while
  keeping their values visually dominant.
- Preserve existing AQI-aware advisory, warning, and critical alert colors, but
  add a subtle outer border so alerts sit cleanly within the page.

## Model Diagnostics

- Give each model metric card a colored top border:
  - LightGBM: blue
  - XGBoost: orange
  - CatBoost: purple
  - Random Forest: brown
  - Linear Ridge: teal
- Keep these model accents separate from the categorical colors used for the
  Actual AQI history line.
- Add a restrained border around the Plotly chart and leaderboard, with no
  secondary card nested inside either container.
- Improve tab, expander, multiselect, radio, and date-input boundaries while
  retaining Streamlit's normal interaction behavior.

## Implementation Boundary

The styling will be implemented through a small CSS generator or constants in
`app/ui.py` so required design tokens are testable without rendering a browser.
Existing Streamlit components and application structure remain unchanged.

## Testing And Verification

- Add a focused unit test that verifies the generated stylesheet includes the
  AQI accent strip, neutral panel borders, and all five model color accents.
- Run the complete unit test suite.
- Reload the Streamlit app at `http://localhost:8502/`.
- Check both tabs at desktop width for consistent borders, readable contrast,
  unclipped labels, and absence of nested-card styling.
- Check a narrow viewport to confirm metric cards wrap cleanly and controls do
  not overlap.

## Acceptance Criteria

- The dashboard has visibly clearer section boundaries without becoming busy.
- AQI category colors are recognizable but occupy only a small visual area.
- Every active model metric card can be identified by its accent color.
- Charts, controls, alerts, and tables remain readable in the existing dark
  theme.
- No forecasting or diagnostics behavior changes.
