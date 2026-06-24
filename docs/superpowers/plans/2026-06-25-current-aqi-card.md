# Current AQI Card Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display a source-attributed current US AQI card before Predicted AQI, with a clearly labeled SQLite fallback and live-only forecast comparison.

**Architecture:** Add `app/current_aqi.py` as the boundary for Open-Meteo Air Quality parsing, HTTP retrieval, and SQLite fallback. Keep Streamlit caching and rendering in `app/ui.py`, using the existing responsive metric grid and extending cards with optional metadata.

**Tech Stack:** Python 3.11, requests, sqlite3, Streamlit, unittest

---

### Task 1: Build The Current AQI Data Boundary

**Files:**
- Create: `app/current_aqi.py`
- Create: `tests/test_current_aqi.py`

- [ ] **Step 1: Write Failing Parser And Fallback Tests**

Create tests for a live reading, malformed API response with SQLite fallback,
and complete unavailability:

```python
class CurrentAQITest(unittest.TestCase):
    def test_live_open_meteo_reading_is_current(self):
        reading = resolve_current_aqi(
            latitude=36.7378,
            longitude=-119.7871,
            station_ids=("FRES_OPENMETEO", "FRES"),
            db_path=self.db_path,
            http_get=lambda *args, **kwargs: FakeResponse(
                {"current": {"time": "2026-06-25T08:15", "us_aqi": 73.0}}
            ),
        )
        self.assertEqual(reading.label, "Current AQI")
        self.assertEqual(reading.value, 73.0)
        self.assertTrue(reading.is_current)
        self.assertEqual(reading.source, "Open-Meteo Air Quality")

    def test_malformed_api_uses_latest_sqlite_record(self):
        reading = resolve_current_aqi(
            latitude=36.7378,
            longitude=-119.7871,
            station_ids=("FRES_OPENMETEO", "FRES"),
            db_path=self.db_path,
            http_get=lambda *args, **kwargs: FakeResponse({"current": {}}),
        )
        self.assertEqual(reading.label, "Last Recorded AQI")
        self.assertFalse(reading.is_current)
        self.assertEqual(reading.source, "SQLite history")

    def test_missing_api_and_database_data_returns_unavailable(self):
        reading = resolve_current_aqi(
            latitude=37.3394,
            longitude=-121.8950,
            station_ids=("SJ_OPENMETEO",),
            db_path=self.empty_db_path,
            http_get=lambda *args, **kwargs: FakeResponse({"current": {}}),
        )
        self.assertIsNone(reading.value)
        self.assertEqual(reading.label, "Current AQI")
        self.assertEqual(reading.source, "Unavailable")
```

- [ ] **Step 2: Run Tests And Confirm RED**

Run:

```powershell
& '.venv\Scripts\python.exe' -m unittest tests.test_current_aqi -v
```

Expected: FAIL because `app.current_aqi` does not exist.

- [ ] **Step 3: Implement The Immutable Reading And Resolver**

Implement:

```python
@dataclass(frozen=True)
class AQIReading:
    value: float | None
    observed_at: datetime | None
    label: str
    source: str
    is_current: bool
```

Add `parse_current_payload`, `load_latest_recorded_aqi`, and
`resolve_current_aqi`. The resolver requests:

```python
params = {
    "latitude": latitude,
    "longitude": longitude,
    "current": "us_aqi",
    "timezone": "GMT",
}
```

Use parameter placeholders for every SQLite station identifier and return an
unavailable reading if both sources fail.

- [ ] **Step 4: Run Tests And Confirm GREEN**

Run:

```powershell
& '.venv\Scripts\python.exe' -m unittest tests.test_current_aqi -v
```

Expected: all Current AQI tests pass.

### Task 2: Integrate Current AQI Into The Live Metric Grid

**Files:**
- Modify: `app/ui.py`
- Modify: `tests/test_ui_retry.py`

- [ ] **Step 1: Write Failing UI Assertions**

Inject a fake `app.current_aqi` module into the existing UI integration test and
return:

```python
AQIReading(
    value=57.0,
    observed_at=datetime(2026, 6, 25, 8, 15, tzinfo=timezone.utc),
    label="Current AQI",
    source="Open-Meteo Air Quality",
    is_current=True,
)
```

Assert the rendered metric grid contains, in order:

```python
current_index = rendered_markup.index("Current AQI")
predicted_index = rendered_markup.index("Predicted AQI")
self.assertLess(current_index, predicted_index)
self.assertIn("Open-Meteo Air Quality", rendered_markup)
self.assertIn("+8.7 vs current", rendered_markup)
self.assertIn("metric-meta", rendered_markup)
```

- [ ] **Step 2: Run The UI Test And Confirm RED**

Run:

```powershell
& '.venv\Scripts\python.exe' -m unittest tests.test_ui_retry.DashboardPredictionModeTest.test_dashboard_defaults_to_local_prediction_and_http_retry_remains_available -v
```

Expected: FAIL because the Current AQI card is not rendered.

- [ ] **Step 3: Add Cached Retrieval And Card Metadata**

Add history identifiers to `STATIONS`, import `AQIReading` and
`resolve_current_aqi`, and define:

```python
@st.cache_data(ttl=900, show_spinner=False)
def fetch_current_aqi(station_name: str) -> AQIReading:
    station = STATIONS[station_name]
    return resolve_current_aqi(
        latitude=station["lat"],
        longitude=station["lon"],
        station_ids=tuple(station["history_station_ids"]),
        db_path=DEFAULT_DB_PATH,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )
```

Extend `metric_card_html` with an optional `meta` row and update
`metric_cards_grid_html` to accept five fields per card.

- [ ] **Step 4: Render Current AQI And Live Comparison**

Fetch the reading independently from the forecast pipeline. Render:

```python
current_card = (
    reading.label,
    "N/A" if reading.value is None else f"{reading.value:.0f}",
    category_or_unavailable,
    category_color_or_neutral,
    source_and_timestamp,
)
```

Place it before Predicted AQI. Append a signed one-decimal difference to the
Predicted AQI detail only when `reading.is_current` and `reading.value` is not
`None`.

- [ ] **Step 5: Run Focused And Full Tests**

Run:

```powershell
& '.venv\Scripts\python.exe' -m unittest tests.test_current_aqi tests.test_ui_retry -v
& '.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

Expected: all tests pass.

### Task 3: Verify Live And Fallback Presentation

**Files:**
- Modify only if defects are found: `app/ui.py`
- Modify only if defects are found: `app/current_aqi.py`

- [ ] **Step 1: Verify The Live API Directly**

Call `resolve_current_aqi` for Fresno and confirm the returned label is
`Current AQI`, the source is `Open-Meteo Air Quality`, and a timestamp is
present.

- [ ] **Step 2: Verify Desktop Dashboard**

Reload `http://localhost:8502/` and confirm:

- Current AQI appears before Predicted AQI.
- Both AQI cards use health-category colors.
- Source and update time remain legible.
- Predicted AQI shows the signed comparison.

- [ ] **Step 3: Verify 720-Pixel Layout**

Set the browser viewport to 720 by 900 pixels and confirm:

- Five cards wrap without horizontal page overflow.
- Card labels, values, detail, and metadata are not clipped.
- Restore the default viewport after verification.

- [ ] **Step 4: Run Final Verification And Commit**

Run:

```powershell
git diff --check
& '.venv\Scripts\python.exe' -m unittest discover -s tests -v
git status --short
```

Then commit:

```powershell
git add app/current_aqi.py app/ui.py tests/test_current_aqi.py tests/test_ui_retry.py
git commit -m "Add current AQI comparison card"
```
