# Live Validation and Model Diagnostics Design

## Objective

Add a dedicated English-language Streamlit tab named
`Live Validation & Model Diagnostics` to the California AQI application.
The tab audits historical forecasts from five trained models against Actual AQI
stored in the local SQLite database.

The supported models are:

- LightGBM
- XGBoost
- CatBoost
- Random Forest
- Linear Ridge

The supported forecast scenarios are:

- Short-term Nowcasting (1h)
- Long-term Forecasting (24h)

## User Experience

The application will use two top-level Streamlit tabs:

1. `Live Forecast`
2. `Live Validation & Model Diagnostics`

The diagnostics tab uses a vertical analysis layout:

1. Filter panel
2. Historical alignment chart
3. Relative Prediction Accuracy cards
4. Accuracy formula explanation
5. Live metrics leaderboard

All visible UI copy in the new tab will be in English.

## Filter Panel

The filter panel contains:

- A city `st.selectbox` with Fresno, Los Angeles, and San Jose.
- A scenario `st.radio` with Short-term Nowcasting (1h) and Long-term
  Forecasting (24h).
- A custom `st.date_input` range. Its default is November 1, 2025 through
  December 31, 2025.
- Quick historical range controls for `24 Hours`, `7 Days`, `30 Days`, and
  `Full Custom Range`.
- A `st.multiselect` containing all five model names. The default selection is
  LightGBM and Linear Ridge.

Each model can be added or removed independently. The selected model list
controls the prediction curves, metric cards, and leaderboard rows.

Actual AQI is not part of the multiselect and always remains visible. If the
user removes every model, the chart still shows Actual AQI and the metrics area
displays an instruction to select at least one model.

The effective end timestamp is the latest available hourly observation on or
before 23:59:59 of the selected end date. Quick historical ranges are measured
backward from that timestamp:

- `24 Hours`: final 24 hourly observations ending at the effective end
  timestamp.
- `7 Days`: final seven calendar days ending on the selected end date.
- `30 Days`: final 30 calendar days ending on the selected end date.
- `Full Custom Range`: the complete start and end dates from `st.date_input`.

## Historical Alignment Chart

The chart will use `plotly.graph_objects` and `hovermode="x unified"`.

### Actual AQI

Actual AQI is read from `meteorology_history.target_aqi` in `aqi_data.db`.
It is rendered in two coordinated layers:

- A prominent black dashed reference line with `color="black"`, `width=3`,
  and `dash="dash"`.
- Colored markers or line segments placed over the reference path according to
  the US AQI category of each observation.

US AQI colors and thresholds are:

| AQI range | Category | Color |
|---|---|---|
| 0-50 | Good | Green |
| 51-100 | Moderate | Yellow |
| 101-150 | Unhealthy for Sensitive Groups | Orange |
| 151-200 | Unhealthy | Red |
| 201-300 | Very Unhealthy | Purple |
| 301+ | Hazardous | Maroon |

The unified tooltip includes timestamp, Actual AQI, AQI health category, and
all active model predictions.

### Prediction Curves

Selected model predictions use this academic palette:

- LightGBM: Blue
- XGBoost: Orange
- Linear Ridge: Green
- CatBoost: Purple
- Random Forest: Brown

The plot background is white. Both axes use visible lines and subtle
`whitesmoke` gridlines. The chart title, axes, legend, and hover labels use
English text.

## Dynamic Metrics

Metrics are recalculated on every filter change from the rows currently visible
for the selected city, scenario, date range, and active models.

The metrics are:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared score (R2)
- Relative Prediction Accuracy

Relative Prediction Accuracy is defined as:

```text
WMAPE = sum(abs(Actual AQI - Predicted AQI)) / sum(abs(Actual AQI))
Relative Prediction Accuracy = max(0, 1 - WMAPE) * 100
```

If the Actual AQI denominator is zero, Relative Prediction Accuracy is
unavailable rather than infinite or misleading.

The UI explanation states:

- A value of 100% means the predictions exactly match Actual AQI in the
  selected window.
- R2 measures explained AQI variance and is not an accuracy percentage.

One `st.metric` card is shown per active model. Cards display Relative
Prediction Accuracy as the primary value and dynamic R2 as supporting context.
Cards are ordered by Relative Prediction Accuracy, highest first.

The `st.dataframe` leaderboard uses these English column titles:

- Model
- Relative Accuracy
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R2 Score (Explained Variance)

All numeric values are formatted to four decimal places. Relative Accuracy is
formatted as a percentage with four decimal places.

## SQLite Data Model

SQLite remains the runtime source of truth.

Existing Actual AQI is read from:

```text
meteorology_history(
    time,
    station_id,
    station_name,
    target_aqi,
    ...
)
```

A new table stores historical backtest predictions:

```sql
CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL,
    station_id TEXT NOT NULL,
    station_name TEXT NOT NULL,
    city_name TEXT NOT NULL,
    scenario TEXT NOT NULL,
    horizon_hours INTEGER NOT NULL CHECK (horizon_hours IN (1, 24)),
    model_name TEXT NOT NULL,
    predicted_aqi REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(time, station_id, scenario, model_name)
);
```

Indexes will cover `(city_name, scenario, time)` and
`(model_name, scenario, time)`.

The diagnostics query:

1. Filters `model_predictions` by city, scenario, model, and time.
2. Joins to `meteorology_history` by exact `station_id` and timestamp.
3. Uses only `meteorology_history.target_aqi` as Actual AQI.
4. Orders rows by timestamp and model name.

Rows missing Actual AQI or predicted AQI are excluded from metric calculations.
A model with no valid aligned rows is reported without breaking other models.

City normalization maps station variants to the three UI labels. In
particular, Fresno station IDs map to Fresno, Los Angeles station IDs map to
Los Angeles, and San Jose station IDs map to San Jose.

## Training Output and Prediction Synchronization

`scr/train_combined_panel_models.py` will retain `test_station_ids` when
building prediction reports. The historical prediction output will include:

- Configuration
- Model
- time
- station_id
- station_name
- city_name
- Predicted_AQI

`Actual_AQI` may remain in the training report for offline auditing, but the
Streamlit runtime will ignore it and retrieve Actual AQI from SQLite.

A synchronization command will create the `model_predictions` table and upsert
the prediction report into SQLite. The operation is idempotent because of the
unique key.

The full five-model training command will write a report containing all five
models for both scenarios. The `--lightgbm-only` production export path must
not overwrite the five-model leaderboard, scenario evaluation, or historical
prediction report.

## Model Artifact Packaging

Training will persist ten model artifacts: five algorithms for each of two
forecast horizons.

```text
models/
  nowcast_1h/
    lightgbm.txt
    xgboost.json
    catboost.cbm
    random_forest.joblib
    linear_ridge.joblib
    metadata.json
  forecast_24h/
    lightgbm.txt
    xgboost.json
    catboost.cbm
    random_forest.joblib
    linear_ridge.joblib
    metadata.json
```

Serialization formats are native where available:

- LightGBM: booster text format
- XGBoost: JSON model format
- CatBoost: CBM format
- Random Forest: joblib
- Linear Ridge: joblib

Each horizon-level `metadata.json` contains:

- Scenario and horizon
- Training, validation, and test periods
- Exact feature order for each model family
- Categorical feature metadata where applicable
- Library versions
- Training timestamp
- Test metrics for every packaged model

The existing backend compatibility paths
`models/lightgbm_nowcast.txt` and `models/lightgbm_forecast24h.txt` will remain
available, either as additional exports or compatibility copies.

## Code Organization

The implementation will keep `app/ui.py` responsible for Streamlit layout and
interaction. Testable data and metric logic will be placed in a focused helper
module under `app/` rather than embedded in top-level Streamlit execution.

Responsibilities are:

- `app/ui.py`: tabs, controls, Plotly rendering, cards, warnings, and table.
- Diagnostics helper: SQLite initialization/query, city normalization, AQI
  category mapping, data alignment, metric calculation, and figure creation.
- Training script: station-aware prediction export and model serialization.
- Synchronization script or callable helper: idempotent CSV-to-SQLite import.

Relative paths are resolved from the project root with `pathlib.Path`.
SQLite connections use context managers and are closed after each operation.

## Error Handling

The tab will handle these states without crashing:

- SQLite database is missing.
- Required tables are missing.
- The selected range has no Actual AQI or prediction rows.
- A selected model has no aligned rows.
- The user selects no models.
- Relative Accuracy has a zero denominator.
- R2 is undefined because fewer than two aligned observations exist.
- The prediction import contains unsupported city, model, or scenario values.

For an empty selected range, the English UI warning is:

```text
No historical records were found for the selected period. Please update the filters.
```

Database and import errors are logged with actionable details while the UI
shows a concise error message.

## Performance

Database queries select only required columns and use indexed filters.
Streamlit caches read-only diagnostics data for a short TTL. Filtering and
metric calculations occur on the selected subset rather than loading the full
2018-2025 panel into the browser.

Plot rendering and metric calculations use every aligned hourly observation in
the effective selected range.

## Test Strategy

Automated tests will cover:

- Creation and idempotent upsert of `model_predictions`.
- Preservation of station identity in training prediction reports.
- City, scenario, model, and inclusive date filtering.
- Alignment of predictions with Actual AQI from SQLite.
- MAE, RMSE, R2, WMAPE, and Relative Prediction Accuracy.
- Zero Actual AQI denominator behavior.
- AQI category thresholds and colors.
- Actual AQI always being present in Plotly figures.
- Adding and removing individual model traces.
- Empty database, empty range, and no-model states.
- Artifact paths and metadata for all five models and two horizons.
- `--lightgbm-only` not overwriting full five-model reports.

Existing UI retry and model export tests must continue to pass.

## Acceptance Criteria

The feature is complete when:

1. The Streamlit application contains the approved English diagnostics tab.
2. Users can independently add or remove any of the five model curves.
3. Actual AQI always remains visible and is category-colored by US AQI level.
4. The four historical range modes work with city and scenario filters.
5. Plotly unified hover shows Actual AQI, category, and active predictions.
6. Metrics and Relative Prediction Accuracy recalculate from the filtered,
   SQLite-aligned rows.
7. Empty and partial data states produce warnings instead of exceptions.
8. SQLite contains station-aware predictions for all five models and both
   scenarios.
9. All ten trained model artifacts and both metadata files are persisted.
10. The complete automated test suite passes.
