# Comparative US AQI Time Series Forecasting in California with Extreme-Event and Climate-Context Evaluation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LaTeX Manuscript](https://img.shields.io/badge/Paper-Springer%20Nature%20%28PDF%29-007ACC?style=flat-square&logo=latex&logoColor=white)](paper/main.pdf)

This repository contains the data pipeline, machine learning modeling framework, interactive web dashboard, and LaTeX research manuscript for forecasting hourly US Air Quality Index (AQI) in California (Fresno, Los Angeles, and San Jose).

---

## 📌 Project Overview & Abstract

This project presents a comprehensive hourly US AQI forecasting benchmark in California using a merged panel dataset of **181,122 hourly observations** from **2018 to 2025**. The dataset integrates ground-truth observations from the **U.S. EPA Air Quality System (AQS)** with meteorological and atmospheric indicators from **Open-Meteo**.

Five tabular machine learning regressors (**Linear Ridge, Random Forest, LightGBM, CatBoost, and XGBoost**) are rigorously benchmarked against two naive baselines (**Persistence** and **Climatology**).

### Experimental Setup & Split Strategy
- **Climate-Context Temporal Split**:
  - **Train Set**: 2018–2019, 2021–2024
  - **Validation Set (Wildfire/Extreme Holdout)**: 2020 (Evaluating model resilience during historic California wildfire events)
  - **Test Set (Independent Out-of-Distribution Year)**: 2025
- **Forecast Horizons**:
  1. **Short-term Nowcasting (Lag 1–3h)**: Autoregressive setting where recent AQI lags are available.
  2. **Long-term Forecasting (Lag 24h)**: Strict 24-hour-ahead forecast masking local AQI indicators between $t-1$ and $t-23$.

---

## 📁 Repository Structure

```gcode
Project/
├── app/                        # Streamlit web application & interactive dashboard
│   ├── main.py                 # Application entry point
│   ├── ui.py                   # UI components, layout, and tabs
│   └── diagnostics.py          # Interactive diagnostic plots & chart renderers
├── paper/                      # LaTeX manuscript formatted for Springer Nature (sn-jnl)
│   ├── main.tex                # Primary LaTeX document source
│   ├── main.pdf                # Compiled manuscript PDF
│   ├── references.bib          # BibTeX reference database
│   ├── sn-jnl.cls              # Springer Nature LaTeX class file
│   ├── sn-mathphys-num.bst     # Springer Nature bibliography style
│   └── figures/                # High-resolution figures embedded in the paper
├── presentation/               # Slide deck & presentation deliverables
│   ├── presentation.html       # Interactive HTML slide presentation
│   ├── presentation_script.md  # Detailed presentation transcript & speaking notes
│   ├── presentation_analysis.md# Presentation structural analysis
│   └── AI_AuditLog_*.xlsx      # Team AI contribution audit logs
├── notebook/                   # Jupyter Notebooks for analysis & reporting
│   ├── Data Analysis with Python.ipynb # Exploratory Data Analysis (EDA)
│   └── Model Report.ipynb      # In-depth model evaluation report
├── data/                       # Datasets and generated visualizations
│   ├── processed/              # Cleaned CSVs (leaderboard, ablation, audit, scenarios)
│   └── plots/                  # Exported plots and EDA figures
├── models/                     # Serialized model artifacts (Joblib, JSON, CBM)
├── scr/                        # Python scripts for data collection, training, and evaluation
│   ├── run_all.py              # Main automated pipeline runner
│   ├── train_combined_panel_models.py # Model training & validation suite
│   ├── generate_paper_figures.py      # Automated figure generator for paper
│   ├── run_lightgbm_ablation.py       # Feature ablation experiment
│   ├── leakage_audit.py               # Temporal data leakage verification
│   └── collect_data.py                # Data collection & scraping pipeline
├── docs/                       # Project architecture documentation
│   └── architecture.md         # System design & architecture details
├── aqi_data.db                 # Local SQLite database (git-ignored, ~200MB)
└── requirements.txt            # Python dependencies
```

---

## ⚡ Quick Start & Installation

### 1. Environment Setup

It is recommended to use a Python virtual environment (`.venv`):

```bash
# Clone the repository
git clone https://github.com/ThayBayThatSon/DAP391m_AI2011_Group7_Project.git
cd DAP391m_AI2011_Group7_Project

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\activate

# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launching the Interactive Web Dashboard

To run the Streamlit web application locally:

```bash
streamlit run app/main.py
```
Open your browser at `http://localhost:8501` to explore live AQI forecasts, model diagnostics, extreme-event heatmaps, and feature importance analyses.

---

## 🔄 Reproducibility & Pipeline Execution

To execute the end-to-end pipeline (model training, evaluation leaderboards, scenario analyses, feature ablation, data leakage audits, and figure generation):

```bash
python scr/run_all.py
```

### Individual Script Execution:
- **Train All Models**: `python scr/train_combined_panel_models.py`
- **Generate Paper Figures**: `python scr/generate_paper_figures.py`
- **Feature Ablation Study**: `python scr/run_lightgbm_ablation.py`
- **Audit Data Leakage**: `python scr/leakage_audit.py`

### Key Generated Artifacts:
- **Model Leaderboards**: `data/processed/california_aqi_model_leaderboard.csv`
- **Scenario Evaluations**: `data/processed/california_aqi_scenario_evaluation.csv`
- **Feature Ablation**: `data/processed/california_aqi_lightgbm_ablation.csv`
- **Leakage Audit**: `data/processed/california_aqi_leakage_audit.csv`
- **Compiled Paper**: [paper/main.pdf](paper/main.pdf)

---

## 📊 Key Research Findings

1. **Short-Term Nowcasting (Lag 1–3h)**:
   - **XGBoost** achieves the highest overall test performance ($R^2 = 0.8711, \text{MAE} = 5.83, \text{RMSE} = 9.47$), followed closely by **LightGBM** ($R^2 = 0.8706, \text{MAE} = 6.06, \text{RMSE} = 9.49$).
   - The **Persistence** baseline performs remarkably strong during acute pollution spikes ($>95^{\text{th}}$ percentile AQI, $R^2 = 0.3004$) and Wildfire season windows ($R^2 = 0.8595$), outperforming complex ML models during extreme peak events due to heavy short-term temporal autocorrelation.
2. **Long-Term 24h Forecasting (Lag 24h)**:
   - **XGBoost** achieves the highest accuracy ($R^2 = 0.4648, \text{MAE} = 13.64, \text{RMSE} = 19.30$), followed by **LightGBM** ($R^2 = 0.4629$) and **Linear Ridge** ($R^2 = 0.4601$).
   - Naive baselines fail completely when autoregressive short lags are masked, proving that 24h forecasting relies on learning meteorology (VPD, Temperature, Dew Point, Boundary Layer Height) and regional spatial dynamics.
3. **Extreme Event Resilience (2020 Wildfire Holdout)**:
   - Tree-based ensemble models maintain reliable error bounds during severe wildfire-driven AQI anomalies, whereas linear models degrade significantly.

---

## 📜 Paper & Presentation

- 📄 **Research Paper (PDF)**: [paper/main.pdf](paper/main.pdf)
- 📝 **LaTeX Manuscript**: [paper/main.tex](paper/main.tex)
- 📊 **Interactive Presentation**: [presentation/presentation.html](presentation/presentation.html)
- 🗣️ **Presentation Transcript**: [presentation/presentation_script.md](presentation/presentation_script.md)

