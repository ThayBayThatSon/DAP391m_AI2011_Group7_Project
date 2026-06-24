# Dashboard Visual Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add restrained AQI accents, thin panel borders, and model-colored metric cards to the existing Streamlit dashboard.

**Architecture:** Keep the Streamlit page structure and application behavior unchanged. Add pure HTML/CSS helpers in `app/ui.py`, render live and diagnostic metric cards through a shared helper, and verify the emitted markup through the existing fake Streamlit integration test before checking the running app in a browser.

**Tech Stack:** Python 3.11, Streamlit, unittest, HTML/CSS, Playwright through the in-app browser

---

### Task 1: Specify The Rendered Theme Contract

**Files:**
- Modify: `tests/test_ui_retry.py`
- Test: `tests/test_ui_retry.py`

- [ ] **Step 1: Capture Markdown Render Calls**

Update the fake Streamlit implementation so panels share the parent recorder:

```python
class FakePanel:
    def __init__(self, owner=None):
        self.owner = owner

    def markdown(self, body, *args, **kwargs):
        if self.owner is not None:
            self.owner.markdown_calls.append(str(body))


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        ...
        self.markdown_calls: list[str] = []

    def markdown(self, body, *args, **kwargs):
        self.markdown_calls.append(str(body))

    def columns(self, *args, **kwargs):
        ...
        return [FakePanel(self) for _ in range(count)]
```

- [ ] **Step 2: Write Failing Theme Assertions**

After importing `app/ui.py`, assert that the rendered stylesheet and accent strip
contain the approved visual tokens:

```python
rendered_markup = "\n".join(fake_streamlit.markdown_calls)
self.assertIn(".aqi-accent-strip", rendered_markup)
self.assertIn('data-testid="stPlotlyChart"', rendered_markup)
self.assertIn("border: 1px solid", rendered_markup)
self.assertIn("#16a34a", rendered_markup)
self.assertIn("#eab308", rendered_markup)
self.assertIn("#f97316", rendered_markup)
self.assertIn("#dc2626", rendered_markup)
self.assertIn("#7e22ce", rendered_markup)
self.assertIn("#7f1d1d", rendered_markup)
```

Render a five-row metrics frame and assert that each model accent is emitted:

```python
module.render_metric_cards(
    pd.DataFrame(
        [
            {"model_name": name, "relative_accuracy": 90.0, "r2": 0.8}
            for name in fake_diagnostics.MODEL_NAMES
        ]
    )
)
rendered_markup = "\n".join(fake_streamlit.markdown_calls)
for accent in ("#3b82f6", "#f59e0b", "#a855f7", "#a16207", "#14b8a6"):
    self.assertIn(accent, rendered_markup)
```

- [ ] **Step 3: Run The Focused Test And Confirm RED**

Run:

```powershell
& '.venv\Scripts\python.exe' -m unittest tests.test_ui_retry.DashboardPredictionModeTest.test_dashboard_defaults_to_local_prediction_and_http_retry_remains_available -v
```

Expected: FAIL because the AQI accent strip and model-card accents are not yet
rendered.

### Task 2: Implement The Restrained Dashboard Theme

**Files:**
- Modify: `app/ui.py`
- Test: `tests/test_ui_retry.py`

- [ ] **Step 1: Add Theme Tokens And Safe Card Markup**

Import `html` and define model accents:

```python
MODEL_CARD_ACCENTS = {
    "LightGBM": "#3b82f6",
    "XGBoost": "#f59e0b",
    "CatBoost": "#a855f7",
    "Random Forest": "#a16207",
    "Linear Ridge": "#14b8a6",
}
```

Add a safe shared card renderer:

```python
def metric_card_html(
    label: str,
    value: str,
    detail: str,
    accent: str,
    variant: str = "model",
) -> str:
    return (
        f'<div class="metric-card metric-card-{html.escape(variant)}" '
        f'style="--metric-accent:{html.escape(accent)}">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-detail">{html.escape(detail)}</div>'
        "</div>"
    )
```

- [ ] **Step 2: Render Live And Model Cards With Stable Accents**

Replace the four `st.metric` calls with `metric_card_html`, using the current AQI
category color for Predicted AQI and fixed contextual colors for weather:

```python
live_metrics = (
    ("Predicted AQI", f"{prediction['predicted_aqi']:.2f}", category, category_color),
    ("Temperature", f"{weather['temperature_2m']:.1f} C", "Air temperature", "#f97316"),
    ("Humidity", f"{weather['relative_humidity_2m']:.0f}%", "Relative humidity", "#38bdf8"),
    ("Wind", f"{weather['wind_speed_10m']:.2f} m/s", "10 m wind speed", "#14b8a6"),
)
for column, metric in zip(st.columns(4), live_metrics):
    column.markdown(metric_card_html(*metric, variant="live"), unsafe_allow_html=True)
```

Update `render_metric_cards` to use `MODEL_CARD_ACCENTS[row.model_name]` and
display Relative Accuracy as the value with R² as the detail.

- [ ] **Step 3: Add The AQI Accent Strip And Component Borders**

Replace the inline style block with a stylesheet that includes:

```css
.aqi-accent-strip {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    height: 4px;
    margin: 0 0 1.15rem;
    overflow: hidden;
    border-radius: 4px;
}
.metric-card {
    min-height: 126px;
    padding: 0.85rem 0.9rem;
    border: 1px solid #2b3442;
    border-top: 3px solid var(--metric-accent);
    border-radius: 8px;
    background: #151a22;
}
[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
[data-testid="stDeckGlJsonChart"],
[data-testid="stExpander"] {
    border: 1px solid #2b3442;
    border-radius: 8px;
}
```

Add controlled styles for tabs, inputs, focus states, and alerts, then render the
six-color strip immediately below the subtitle.

- [ ] **Step 4: Run The Focused Test And Confirm GREEN**

Run:

```powershell
& '.venv\Scripts\python.exe' -m unittest tests.test_ui_retry.DashboardPredictionModeTest.test_dashboard_defaults_to_local_prediction_and_http_retry_remains_available -v
```

Expected: PASS.

- [ ] **Step 5: Run The Complete Test Suite**

Run:

```powershell
& '.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Expected: all tests pass.

### Task 3: Browser Verification And Commit

**Files:**
- Modify only if visual defects are found: `app/ui.py`

- [ ] **Step 1: Reload The Existing Streamlit App**

Open `http://localhost:8502/` and wait for both tabs to become interactive.

- [ ] **Step 2: Verify Desktop Layout**

At approximately 1440x900:

- Check that the title accent strip is narrow and unobtrusive.
- Check that all four live metric cards have aligned heights and distinct accents.
- Check that the map and forecast chart have thin borders.
- Open diagnostics and confirm model cards retain their assigned colors after
  changing selected models.
- Confirm the Plotly chart and leaderboard have one border each and no nested
  card effect.

- [ ] **Step 3: Verify Narrow Layout**

At approximately 720x900:

- Confirm metric cards wrap without clipped values.
- Confirm tabs, selector labels, and date controls do not overlap.
- Confirm chart and table remain scrollable or responsive.

- [ ] **Step 4: Run Final Verification**

Run:

```powershell
git diff --check
& '.venv\Scripts\python.exe' -m unittest discover -s tests -v
git status --short
```

Expected: no whitespace errors, all tests pass, and only intended files are
modified.

- [ ] **Step 5: Commit The Visual Polish**

```powershell
git add app/ui.py tests/test_ui_retry.py
git commit -m "Polish AQI dashboard visuals"
```
