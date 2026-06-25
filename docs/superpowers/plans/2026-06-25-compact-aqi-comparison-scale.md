# Compact AQI Comparison Scale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the AQI forecast panel and rename the two dashboard tabs.

**Architecture:** Keep the existing pure Plotly helper and change only its visual representation: category band and interval shapes, two marker traces, direct annotations, and no legend. Update Streamlit tab labels and corresponding integration assertions.

**Tech Stack:** Python 3.11, Plotly, Streamlit, unittest

---

### Task 1: Define The Compact Figure Contract

**Files:**
- Modify: `tests/test_forecast_panel.py`

- [ ] Assert figure height is 180 and `showlegend` is false.
- [ ] Assert the confidence interval is a translucent rectangle with exact x0/x1.
- [ ] Assert six category band annotations exist.
- [ ] Assert Current and Forecast marker traces have distinct symbols and label positions.
- [ ] Run the focused test and confirm it fails against the current 260-pixel legend-based panel.

### Task 2: Implement The Compact Scale

**Files:**
- Modify: `app/forecast_panel.py`

- [ ] Replace the confidence interval line trace with a translucent rectangle.
- [ ] Make category bands thin and add direct category annotations.
- [ ] Retain one Forecast marker and at most one Current marker.
- [ ] Place forecast text above and current text below the scale.
- [ ] Disable the legend, reduce height to 180, and tighten margins.
- [ ] Run `tests.test_forecast_panel` and confirm it passes.

### Task 3: Rename Dashboard Tabs

**Files:**
- Modify: `app/ui.py`
- Modify: `tests/test_ui_retry.py`

- [ ] Change labels to `AQI Forecast` and `Model Validation`.
- [ ] Update integration assertions.
- [ ] Run focused UI tests and the complete suite.

### Task 4: Verify And Commit

**Files:**
- Modify only if visual defects are found: `app/forecast_panel.py`
- Modify only if visual defects are found: `app/ui.py`

- [ ] Reload `http://localhost:8502/`.
- [ ] Verify no legend, a subtle CI band, direct labels, and compact height.
- [ ] Confirm no horizontal overflow.
- [ ] Run `git diff --check` and the full test suite.
- [ ] Commit with `Refine AQI comparison scale`.
