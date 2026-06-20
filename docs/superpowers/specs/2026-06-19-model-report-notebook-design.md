# Model Report Notebook Rewrite Design

## Goal

Rewrite `notebook/Model Report.ipynb` as a hybrid academic report and tutorial for the California AQI forecasting project.

## Approved Direction

The notebook should keep the academic structure of the current report while adding concise explanations at the points where readers most need context: forecasting task definitions, split strategy, metrics, benchmark interpretation, scenario evaluation, and leakage checks.

## Scope

- Keep the notebook reproducible from either the project root or the `notebook` directory.
- Use existing artifacts in `data/processed` and `data/plots`.
- Do not retrain models in the notebook.
- Preserve the final research story: short-term AQI nowcasting is easier than 24-hour forecasting, extreme-event performance remains the main limitation, and leakage controls are central to the credibility of the evaluation.
- Report missing plot artifacts clearly instead of silently skipping them.

## Notebook Structure

1. Title and reader-facing objective.
2. Setup and artifact availability check.
3. Dataset overview and station/time coverage.
4. Forecasting task definitions.
5. Climate-context split and leakage guard.
6. Feature-engineering explanation.
7. Global leaderboard with metric interpretation.
8. Scenario-level robustness evaluation.
9. Paper figures with availability reporting.
10. Leakage audit interpretation.
11. Academic takeaways and limitations.

## Verification

The rewrite is acceptable when:

- The notebook JSON is valid.
- The notebook contains the approved hybrid/tutorial sections.
- The setup and core analysis cells execute without errors.
- Missing figures are surfaced as a readable table or message.
