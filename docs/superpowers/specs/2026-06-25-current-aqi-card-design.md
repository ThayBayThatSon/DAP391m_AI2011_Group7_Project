# Current AQI Card Design

## Goal

Add a clearly labeled current US AQI reading to the Live Forecast tab so users
can compare present air quality with the model forecast without confusing live
modeled conditions, historical observations, and predicted AQI.

## Data Sources

### Primary Source

Request `current=us_aqi` from the Open-Meteo Air Quality API at
`https://air-quality-api.open-meteo.com/v1/air-quality` for the selected
station coordinates.

Open-Meteo documents `us_aqi` as the consolidated United States AQI and states
that current conditions use 15-minute model data. The dashboard must identify
the source as `Open-Meteo Air Quality` rather than imply that it is a regulatory
monitor reading.

### Fallback Source

If the Air Quality API request fails, query the newest non-null `target_aqi`
record for the selected station's configured history identifiers in
`aqi_data.db`.

Fallback data must use the label `Last Recorded AQI`, include its stored
timestamp, and identify the source as `SQLite history`. It must never be labeled
as current AQI.

If neither source is available, render an unavailable card rather than failing
the rest of the live forecast.

## Data Contract

Represent the reading as a small immutable value object with:

- AQI value or `None`
- observation timestamp
- display label
- source label
- whether the value is current

The Open-Meteo parser must reject responses with a missing or non-numeric
`current.us_aqi`. The SQLite lookup must use parameterized SQL and station
identifiers already configured by the application.

## User Interface

- Add the AQI reading as the first card in the Live Forecast metric grid.
- The card order becomes Current/Last Recorded AQI, Predicted AQI, Temperature,
  Humidity, and Wind.
- Color the AQI reading card using the same US AQI category mapping as the
  predicted AQI card.
- Show the category as the primary detail and source/time as supporting text.
- Keep the existing responsive auto-fit grid so five cards wrap without
  horizontal overflow.
- Update Predicted AQI supporting text to include the signed difference from
  the current reading when a primary live AQI value is available, for example
  `Moderate | +8.2 vs current`.
- Do not calculate a forecast difference against stale SQLite fallback data.

## Error Handling

- Cache the live Air Quality request for 15 minutes to reduce repeated network
  calls.
- Handle connection, timeout, HTTP, malformed JSON, and database errors inside
  the AQI retrieval boundary.
- A Current AQI failure must not prevent weather retrieval, model prediction,
  charts, alerts, or diagnostics from rendering.

## Testing

- Test successful parsing of `current.us_aqi` and its timestamp.
- Test SQLite fallback selection for each station identifier set.
- Test that malformed API data falls back instead of crashing.
- Test that unavailable primary and fallback data produces an unavailable card.
- Test Current AQI card ordering, category color, source text, and predicted
  difference markup.
- Run the complete unit test suite.
- Verify desktop and 720-pixel layouts in the running Streamlit app.

## Acceptance Criteria

- A user can see a source-attributed current AQI before the predicted AQI.
- The dashboard never presents the December 2025 SQLite value as current.
- The predicted comparison is shown only against the live Open-Meteo reading.
- Network or database failures do not break the Live Forecast tab.
- Five metric cards remain readable without horizontal page overflow.
