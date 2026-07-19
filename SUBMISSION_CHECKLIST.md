# Submission Checklist

This checklist reflects the current finalized project state after switching to the merged California model-ready dataset and removing unsupported named-event assumptions.

## Submit

- `main.tex`
- `Project_Proposal.md`
- `AI_AuditLog.md`
- `notebook/Data Analysis with Python.ipynb`
- `notebook/Model Report.ipynb`
- `notebook/Model Report VI.ipynb`
- `scr/train_combined_panel_models.py`
# Submission Checklist

This checklist reflects the current finalized project state after switching to the merged California model-ready dataset and removing unsupported named-event assumptions.

## Submit

- `main.tex`
- `Project_Proposal.md`
- `AI_AuditLog.md`
- `notebook/Data Analysis with Python.ipynb`
- `notebook/Model Report.ipynb`
- `notebook/Model Report VI.ipynb`
- `scr/train_combined_panel_models.py`
- `scr/generate_paper_figures.py`
- `scr/collect_data.py`
- `scr/combine_aqi_datasets.py`
- `scr/clean_openmeteo_dataset.py`
- `scr/prepare_ca_pipeline_input.py`
- `scr/data_visulization.py`
- `scr/leakage_audit.py`
- `scr/run_lightgbm_ablation.py`
- `scr/run_all.py`
- `data/processed/california_aqi_model_ready.csv`
- `data/processed/california_aqi_model_leaderboard.csv`
- `data/processed/california_aqi_scenario_evaluation.csv`
- `data/processed/california_aqi_lightgbm_ablation.csv`
- `data/processed/california_aqi_leakage_audit.csv`
- `data/plots/paper_correlation_heatmap_en.png`
- `data/plots/paper_station_map.png`
- `data/plots/paper_workflow_pipeline_architecture.jpg`
- `data/plots/paper_before_during_after_wildfire_context.png`
- `requirements.txt`

## Optional, For Reproducibility

- `data/processed/california_aqi_merged_panel.csv`
- `data/processed/california_aqi_merged_source_report.csv`
- `data/processed/openmeteo_california_hourly_raw.csv`
- `data/processed/openmeteo_california_hourly_clean.csv`
- `data/processed/epa_aqs_station_hourly_raw.csv` (prepared EPA AQS station component)
- `data/processed/epa_aqs_station_hourly_clean.csv` (clean prepared EPA AQS station component)
- `data/processed/epa_aqs_station_missingness_report.csv`
- `data/processed/openmeteo_california_missingness_report.csv`

## Do Not Submit

- `.venv/`
- `.cache.sqlite`
- `catboost_info/`
- `notebook/catboost_info/`
- `scr/__pycache__/`
- `.ipynb_checkpoints/`
- Old/conflicting processed benchmark outputs unless specifically requested.
- Legacy modeling scripts not used for the final reported numbers:
  - `scr/benchmark_framework.py`
  - `scr/model_pipeline.py`
  - `scr/data_preparation.py`
  - `scr/wildfire_dual_config_benchmark.py`

## Current Result Source of Truth

All paper/report model numbers should come from:

- `data/processed/california_aqi_model_leaderboard.csv`
- `data/processed/california_aqi_scenario_evaluation.csv`
- `data/processed/california_aqi_leakage_audit.csv`

The final experiment should be described as:

- Train: 2018, 2019, 2021--2024
- Validation: 2020 (Extreme-AQI/wildfire-context holdout)
- Test: 2025
- Main dataset: `data/processed/california_aqi_model_ready.csv`
- Station-observation source: U.S. EPA Air Quality System (AQS), prepared as `epa_aqs_station_hourly_clean.csv` and labeled `epa_aqs_station` in the final dataset.
- Supplemental source: Open-Meteo records for Fresno, Los Angeles, and San Jose.
- Scenario evaluation: full 2025 test year, non-event baseline, January 2025 winter extreme-AQI window, top-5% AQI observations, and wildfire season 2025 (Jul-Sep).
- Configuration `Short-term Autoregressive (Lag 1-3h)` uses recent AQI lags (nowcasting).
- Configuration `Long-term Forecasting (Lag 24h)` explicitly masks local AQI lags from 1-23 hours to force 24h-ahead forecasting.
