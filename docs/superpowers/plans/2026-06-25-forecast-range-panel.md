# Forecast Range Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the misleading interpolated forecast timeline with a single-horizon AQI comparison and uncertainty panel.

**Architecture:** Add a pure Plotly builder in `app/forecast_panel.py` that clamps AQI values, selects an axis breakpoint, builds health-category bands, and returns both a figure and summary text. `app/ui.py` will remove the Altair timeline functions and render this helper's output.

**Tech Stack:** Python 3.11, Plotly, Streamlit, unittest

---

### Task 1: Specify The Forecast Panel Contract

**Files:**
- Create: `tests/test_forecast_panel.py`
- Create: `app/forecast_panel.py`

- [ ] **Step 1: Write Failing Figure Tests**

Create tests that call:

```python
figure, summary = build_forecast_range_panel(
    predicted_aqi=73.0,
    confidence_lower=61.0,
    confidence_upper=85.0,
    horizon=1,
    current_aqi=68.0,
)
```

Assert:

```python
self.assertEqual(tuple(figure.layout.xaxis.range), (0, 100))
self.assertEqual(figure.layout.height, 260)
self.assertEqual(len(figure.layout.shapes), 7)
forecast = next(trace for trace in figure.data if trace.name == "Forecast +1h")
current = next(trace for trace in figure.data if trace.name == "Current")
self.assertEqual(list(forecast.x), [73.0])
self.assertEqual(forecast.marker.symbol, "diamond")
self.assertEqual(list(current.x), [68.0])
self.assertEqual(current.marker.symbol, "circle")
interval = next(trace for trace in figure.data if trace.name == "Confidence interval")
self.assertEqual(list(interval.x), [61.0, 85.0])
self.assertEqual(len(forecast.x), 1)
self.assertIn("+5.0 AQI", summary)
```

Add separate tests for:

- no current marker when `current_aqi=None`
- no current marker when `current_is_live=False`
- 200, 300, and 500 axis breakpoints
- clamping negative and above-500 values
- improved, worsened, and unchanged category summary wording

- [ ] **Step 2: Run Tests And Confirm RED**

Run:

```powershell
& '.venv\Scripts\python.exe' -m unittest tests.test_forecast_panel -v
```

Expected: FAIL because `app.forecast_panel` does not exist.

- [ ] **Step 3: Implement Pure Panel Helpers**

Create:

```python
AQI_BANDS = (
    (0.0, 50.0, "Good", "#16a34a"),
    (50.0, 100.0, "Moderate", "#eab308"),
    (100.0, 150.0, "Unhealthy for Sensitive Groups", "#f97316"),
    (150.0, 200.0, "Unhealthy", "#dc2626"),
    (200.0, 300.0, "Very Unhealthy", "#7e22ce"),
    (300.0, 500.0, "Hazardous", "#7f1d1d"),
)
```

Implement:

- `clamp_aqi(value) -> float`
- `axis_maximum(*values) -> int`
- `aqi_category(value) -> tuple[str, str, int]`
- `forecast_summary(...) -> str`
- `build_forecast_range_panel(...) -> tuple[go.Figure, str]`

Use six low-opacity rectangular category shapes and one confidence interval
line trace. Add exactly one forecast marker and at most one current marker.

- [ ] **Step 4: Run Tests And Confirm GREEN**

Run:

```powershell
& '.venv\Scripts\python.exe' -m unittest tests.test_forecast_panel -v
```

Expected: all forecast panel tests pass.

### Task 2: Replace The Live Forecast Timeline

**Files:**
- Modify: `app/ui.py`
- Modify: `tests/test_ui_retry.py`

- [ ] **Step 1: Write Failing UI Integration Assertions**

Extend `FakeStreamlit` to record `plotly_chart` calls:

```python
self.plotly_chart_calls: list[dict] = []

def plotly_chart(self, figure, *args, **kwargs):
    self.plotly_chart_calls.append(
        {"figure": figure, "args": args, "kwargs": kwargs}
    )
```

After importing the UI, assert:

```python
self.assertGreaterEqual(len(fake_streamlit.plotly_chart_calls), 1)
live_figure = fake_streamlit.plotly_chart_calls[0]["figure"]
self.assertIn("Forecast +24h", [trace.name for trace in live_figure.data])
self.assertNotIn("Forecast Hour", str(live_figure))
self.assertIn("Forecast comparison", rendered_markup)
```

- [ ] **Step 2: Run The UI Test And Confirm RED**

Run:

```powershell
& '.venv\Scripts\python.exe' -m unittest tests.test_ui_retry.DashboardPredictionModeTest.test_dashboard_defaults_to_local_prediction_and_http_retry_remains_available -v
```

Expected: FAIL because the Live Forecast still renders an Altair timeline.

- [ ] **Step 3: Replace Altair With The Plotly Panel**

In `app/ui.py`:

- remove the `altair` import
- remove `timeline_frame`
- remove `render_timeline_chart`
- import `build_forecast_range_panel`
- build the panel using returned prediction and Current AQI
- pass `current_aqi.value` only when `current_aqi.is_current`
- render summary as a compact bordered HTML block or caption
- call:

```python
st.plotly_chart(
    figure,
    width="stretch",
    config={"displaylogo": False, "displayModeBar": False},
)
```

- [ ] **Step 4: Run Focused And Full Tests**

Run:

```powershell
& '.venv\Scripts\python.exe' -m unittest tests.test_forecast_panel tests.test_ui_retry -v
& '.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Expected: all tests pass.

### Task 3: Verify And Commit

**Files:**
- Modify only if visual defects are found: `app/forecast_panel.py`
- Modify only if visual defects are found: `app/ui.py`

- [ ] **Step 1: Verify The Running Dashboard**

Reload `http://localhost:8502/` and confirm:

- no interpolated hourly line chart exists
- forecast marker, Current marker, and CI are visible
- category bands are readable but subdued
- summary describes the signed change and category direction

- [ ] **Step 2: Verify 720-Pixel Layout**

Confirm no horizontal page overflow and no clipped marker labels. Restore the
default viewport afterward.

- [ ] **Step 3: Run Final Verification**

Run:

```powershell
git diff --check
& '.venv\Scripts\python.exe' -m unittest discover -s tests -v
Invoke-WebRequest -Uri 'http://127.0.0.1:8502/' -UseBasicParsing
git status --short
```

Expected: no whitespace errors, all tests pass, Streamlit responds HTTP 200,
and only intended files are modified.

- [ ] **Step 4: Commit**

```powershell
git add app/forecast_panel.py app/ui.py tests/test_forecast_panel.py tests/test_ui_retry.py
git commit -m "Replace forecast timeline with AQI range panel"
```
