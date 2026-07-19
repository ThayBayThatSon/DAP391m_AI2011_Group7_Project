# Modeling Strategy

This document outlines the machine learning strategy used to forecast Air Quality Index (AQI) values.

## 1. Forecast Configurations
The pipeline supports predicting AQI over several horizons to accommodate both immediate (nowcasting) and future (forecasting) needs:
- **Short-term Autoregressive**: 1-3 hours ahead.
- **Timeline Lags**: 6h, 12h, and 18h ahead.
- **Long-term Forecasting**: 24 hours ahead.

Each horizon configuration dynamically defines which target lags are included to strictly prevent data leakage. For example, a 24-hour forecast only accesses AQI readings observed at least 24 hours prior.

## 2. Models Evaluated
The pipeline evaluates multiple algorithms to find the most robust predictor for tabular time-series data:
- **LightGBM**: Highly efficient gradient boosting framework (selected as the production model).
- **XGBoost**: Standard, robust gradient boosting.
- **CatBoost**: Gradient boosting specialized for categorical variables.
- **Random Forest**: Ensembling through bagging.
- **Linear Ridge**: A regularized linear baseline.

### Baseline Models
To quantify the skill of the machine learning models, two naive baselines are strictly evaluated:
- **Persistence**: Assumes the future AQI will be exactly the same as the most recent observable AQI.
- **Climatology**: Predicts the historical average AQI for that specific month and hour, ignoring current conditions.

## 3. Climate-Context Aware Data Splitting
Because air quality is highly seasonal and subject to rare extreme events (like wildfires), random temporal splitting is insufficient. The project uses a **climate-context aware split**:
- **Train Years**: 2018, 2019, 2021, 2022, 2023, 2024
- **Validation Year**: 2020 (Isolated as a climate-extreme early-stopping holdout).
- **Test Year**: 2025 (Strictly future out-of-sample data).

### Leakage Prevention (Lag Guard)
When transitioning from the 2020 Validation year to the 2021 Train year, a buffer (lag guard) is applied. This removes early 2021 training samples whose rolling history windows might reference late-2020 validation data, preventing information leakage.

## 4. Scenario Evaluation
Models are not just evaluated on overall MAE and RMSE. They are stress-tested against specific scenarios to ensure reliability during critical events:
- **Extreme AQI in Test**: Evaluating performance exclusively on the top 5% highest AQI readings.
- **Wildfire Season**: Evaluating performance strictly during known wildfire months (Jul-Sep).
- **Non-Event Baseline**: Evaluating during normal conditions (AQI <= 100).
- **Source-specific metrics**: Disaggregating performance by data source (EPA AQS vs. Open-Meteo).

## 5. Interpretability
We integrate **SHAP (SHapley Additive exPlanations)** into the production LightGBM pipeline to explain predictions in real-time. This provides granular insights into exactly how weather (e.g., wind speed) and historical AQI are driving the forecast for any given hour.
