# Model Report Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `notebook/Model Report.ipynb` into a hybrid academic report and tutorial.

**Architecture:** Replace the notebook body with curated markdown sections and compact code cells that load existing CSV/plot artifacts, summarize results, and visualize model performance. Keep data loading centralized and avoid model training inside the notebook.

**Tech Stack:** Jupyter Notebook JSON, Python, pandas, matplotlib, IPython display.

---

### Task 1: Add A Structural Regression Check

**Files:**
- Test by command only: `notebook/Model Report.ipynb`

- [ ] **Step 1: Run the failing structure check**

```powershell
@'
import json
from pathlib import Path
nb = json.loads(Path("notebook/Model Report.ipynb").read_text(encoding="utf-8"))
text = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
required = [
    "# Model Report: Hybrid Report + Tutorial",
    "Learning notes",
    "Artifact availability check",
    "How to read MAE, RMSE, and R2",
    "Scenario-level robustness",
    "Leakage audit interpretation",
]
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit("Missing required notebook content: " + ", ".join(missing))
print("Notebook structure check passed")
'@ | python -
```

Expected: FAIL before the rewrite because the old notebook does not contain the hybrid tutorial sections.

### Task 2: Rewrite The Notebook

**Files:**
- Modify: `notebook/Model Report.ipynb`

- [ ] **Step 1: Replace the current cells**

Create markdown and code cells for title/objective, artifact checks, dataset overview, task definitions, split/leakage explanation, feature notes, leaderboard visualization, scenario evaluation, figures, audit interpretation, and takeaways.

- [ ] **Step 2: Keep runtime lightweight**

Use only existing CSV files from `data/processed` and images from `data/plots`. Do not call training scripts.

### Task 3: Verify The Rewrite

**Files:**
- Verify: `notebook/Model Report.ipynb`

- [ ] **Step 1: Validate JSON**

```powershell
python -m json.tool "notebook/Model Report.ipynb" > $null
```

Expected: exit code 0.

- [ ] **Step 2: Re-run the structure check**

```powershell
@'
import json
from pathlib import Path
nb = json.loads(Path("notebook/Model Report.ipynb").read_text(encoding="utf-8"))
text = "\n".join("".join(cell.get("source", [])) for cell in nb.get("cells", []))
required = [
    "# Model Report: Hybrid Report + Tutorial",
    "Learning notes",
    "Artifact availability check",
    "How to read MAE, RMSE, and R2",
    "Scenario-level robustness",
    "Leakage audit interpretation",
]
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit("Missing required notebook content: " + ", ".join(missing))
print("Notebook structure check passed")
'@ | python -
```

Expected: PASS.

- [ ] **Step 3: Execute code-cell smoke test**

Run the notebook code cells through a Python smoke test that shares one namespace and skips only rich display rendering.

Expected: exit code 0 and no Python exceptions.
