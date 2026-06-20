# VPD EDA Paper Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the EDA notebook and paper wording around vapor pressure deficit.

**Architecture:** Add a compact VPD diagnostic section to `notebook/Data Analysis with Python.ipynb` by deriving `vpd_kpa` from temperature and relative humidity, then soften VPD claims in `main.tex` so the paper presents VPD as an engineered dryness feature rather than a proven independent performance driver.

**Tech Stack:** Jupyter Notebook JSON, Python, pandas, matplotlib, LaTeX.

---

### Task 1: Regression Check

**Files:**
- Verify: `notebook/Data Analysis with Python.ipynb`
- Verify: `main.tex`

- [ ] **Step 1: Run a structure check before editing**

```powershell
@'
import json
from pathlib import Path

nb_text = "\n".join("".join(cell.get("source", [])) for cell in json.loads(Path("notebook/Data Analysis with Python.ipynb").read_text(encoding="utf-8")).get("cells", []))
tex = Path("main.tex").read_text(encoding="utf-8")
missing = []
for needle in [
    "## 13. Vapor Pressure Deficit (VPD) Diagnostic",
    "vpd_kpa",
    "Monthly AQI and Vapor Pressure Deficit",
]:
    if needle not in nb_text:
        missing.append(f"notebook:{needle}")
if "benefits from climate-aware validation, VPD, and spatial lag structure" in tex:
    missing.append("main.tex:overstrong VPD benefit claim")
if "engineered dryness feature" not in tex and "VPD-based dryness feature" not in tex:
    missing.append("main.tex:softened VPD wording")
if missing:
    raise SystemExit("Missing or overstrong content: " + "; ".join(missing))
print("VPD paper/notebook structure check passed")
'@ | python -
```

Expected before implementation: fail because the notebook has no VPD EDA section and the paper contains an overstrong VPD claim.

### Task 2: Add VPD EDA

**Files:**
- Modify: `notebook/Data Analysis with Python.ipynb`

- [ ] **Step 1: Insert two notebook cells before the extreme-event section**

Add a markdown cell explaining that VPD is derived from temperature and relative humidity, then a code cell that computes `vpd_kpa`, prints summary statistics, displays correlations, saves a monthly AQI/VPD figure, and plots it.

### Task 3: Soften Paper Claims

**Files:**
- Modify: `main.tex`

- [ ] **Step 1: Replace causal/benefit phrasing**

Use wording such as "incorporates VPD-based engineered dryness features" and "may provide nonlinear threshold context" instead of claiming that VPD independently improves model performance.

### Task 4: Verify

**Files:**
- Verify: `notebook/Data Analysis with Python.ipynb`
- Verify: `main.tex`

- [ ] **Step 1: Re-run the structure check**

Expected after implementation: pass.

- [ ] **Step 2: Execute the inserted VPD code in a notebook-like namespace**

Expected: `vpd_kpa` is finite, non-negative, and the VPD figure is saved.

- [ ] **Step 3: Compile LaTeX if available**

Expected: PDF build succeeds, or report the exact tool/runtime limitation.
