# HƯỚNG DẪN CHI TIẾT TẠO HTML PRESENTATION CHO AI-AGENT
*(Dành cho các AI-Agent sử dụng Marp, Reveal.js hoặc sinh HTML/CSS thuần)*

**System Prompt cho AI-Agent Gen Code:**
> "Bạn là một chuyên gia thiết kế Presentation bằng HTML/CSS/JS (hoặc Marp/Reveal.js). Hãy đọc kỹ từng slide dưới đây và tạo ra mã nguồn tương ứng. 
> Yêu cầu bắt buộc:
> 1. Dùng font chữ hiện đại (Inter, Roboto). Tone màu chủ đạo là Xanh dương đậm (Dark Blue) và Trắng.
> 2. Các đoạn có tag `[IMAGE]` phải được hiển thị bằng thẻ `<img>` với đúng đường dẫn được cung cấp. Phải canh giữa hình ảnh và set width/height phù hợp (khoảng 60-80% khung hình).
> 3. Các đoạn công thức toán học (`$...$`) cần được render đúng định dạng.
> 4. Hiển thị bảng biểu (Table) rõ ràng, viền gọn gàng.
> 5. Các thẻ `[HIGHLIGHT]` cần được tô đậm hoặc đổi màu chữ (vàng/đỏ) để gây sự chú ý."

---

## SLIDE 1: Title Slide (Cover)
- **Layout:** Center Alignment, Title style.
- **Main Title:** Comparative US AQI Time Series Forecasting in California
- **Subtitle:** with Extreme-Event and Climate-Context Evaluation
- **Text:** 
  - DAP391m Final Project | FPT University Ho Chi Minh City
  - **Group 7:** Ta Nguyen Anh Minh (SE200416), Trinh Quoc Sang (SE194517)
  - **Supervisor/Mentor:** Le Vo Minh Thu

## SLIDE 2: Table of Contents
- **Layout:** Left Alignment, Two columns.
- **Title:** Outline (10 Data Science Steps)
- **Bullet points:**
  1. **PART 1:** Problem & Data Understanding (Steps 1-3)
  2. **PART 2:** Feature Engineering & Visualization (Steps 3-4)
  3. **PART 3:** Modeling, Evaluation & AI Application (Steps 5-9)
  4. **PART 4:** Conclusion & AI Reflection (Step 10 + Q&A)

## SLIDE 3: STEP 1 - PROBLEM UNDERSTANDING
- **Layout:** Title + Bullet points
- **Title:** Step 1: Problem Understanding
- **Bullet points:**
  - **Business Context:** California wildfires pose severe health risks. Early warning is critical.
  - **Task Type:** Regression (Predicting continuous AQI).
  - **Target:** Hourly PM2.5-derived AQI (Highly right-skewed with extreme upper-tail spikes).
  - **Features:** Time-series (Lags), Meteorological (Temp, Humidity), and Engineered (VPD).
  - **Success Metrics:** $R^2$ (Explained Variance), MAE, and RMSE.
  - **RBL Justification:** Grounded in baseline paper (Vu et al., 2022). Supported by AI Audit Log.

## SLIDE 4: Research Questions
- **Title:** Research Questions (RQs)
- **Numbered list:**
  1. **RQ1:** How accurately can common tabular machine-learning models forecast hourly US AQI in California under a leakage-safe temporal evaluation protocol?
  2. **RQ2:** How does the availability of recent local AQI history affect forecasting performance between short-term autoregressive nowcasting and strict 24-hour-ahead forecasting?
  3. **RQ3:** How stable are the forecasting models under climate-context and extreme-AQI evaluation, and what do VPD and spatial lag features contribute?

## SLIDE 5: STEP 2 - DATA UNDERSTANDING
- **Layout:** Title + Bullet points
- **Title:** Step 2: Data Understanding & Preprocessing
- **Bullet points:**
  - **Dataset Profile & AI Audit:** 181,122 rows $\times$ 21 base columns. Sources: EPA AQS & Open-Meteo.
  - **Missing Values:** Conservative Linear Interpolation (6h limit) to fix short dropouts; `dropna` for long gaps.
  - **Outliers:** Extreme upper-tail AQI spikes detected. **Decision:** RETAINED, as predicting wildfire events is the primary goal.
  - **Inconsistent:** Handled UTC-to-Local timezone misalignment and removed duplicated timestamps.
  - **Skewness & Imbalance:** Target is heavily right-skewed. **Decision:** No log-transform used, because tree-based models handle skewness natively and we predict raw AQI directly.

## SLIDE 6: Data Alignment & Preprocessing
- **Layout:** Title + Bullet points
- **Title:** Step 1: Data Alignment & Preprocessing
- **Bullet points:**
  - Merged EPA AQS observations with Open-Meteo weather parameters.
  - Built a 181,122-record hourly panel (2018–2025) for Fresno, LA, and San Jose.
  - Handled missing data via **Conservative Linear Interpolation** (6-hour limit).
  - Dropped long missing gaps to prevent "hallucinating" data during critical wildfire spikes.

## SLIDE 7: Geospatial Analysis (Geospatial EDA)
- **Layout:** Title + Image
- **Title:** Step 3: EDA - Geospatial Analysis
- **[IMAGE]:** `c:/FPT/SU2026/DAP391m/Project/data/plots/paper_station_map.png`
- **Caption:** Mapped AQS stations using Latitude/Longitude. Ensures spatial coverage across Fresno, Los Angeles, and San Jose.

## SLIDE 8: Multivariate Correlation (Multivariate EDA)
- **Layout:** Title + Image
- **Title:** Step 3: EDA - Multivariate Analysis
- **[IMAGE]:** `c:/FPT/SU2026/DAP391m/Project/data/plots/paper_correlation_heatmap_en.png`
- **Caption:** Correlation matrix across multiple features. **Insight:** Weak linear correlations justify the RBL decision to use non-linear (Tree-based) models to answer RQ1.

## SLIDE 9: Temporal & Univariate Analysis (Univariate EDA)
- **Layout:** Title + Image
- **Title:** Step 3: EDA - Univariate & Bivariate Timeline
- **[IMAGE]:** `c:/FPT/SU2026/DAP391m/Project/data/plots/eda_aqi_category_timeline.png`
- **Caption:** Analyzed AQI distribution and temporal patterns. **Insight:** Severe upper-tail spikes cluster in late summer. Code generation logged in AI Audit.

## SLIDE 10: STEP 4 - FEATURE ENGINEERING
- **Layout:** Title + Bullet points
- **Title:** Step 4: Feature Engineering
- **Bullet points:**
  - **Handling Review:** Outliers retained, Missing filled via 6h-interpolation, Target not log-transformed (As justified in Step 2).
  - **Feature Enrichment:** Created Temporal Lags (1-3h) and Spatial Lags (cross-station averages).
  - **Feature Transformation:** Derived non-linear physical feature **VPD** (Vapor Pressure Deficit) from Temp & Humidity.
  - **Feature Selection:** Dropped near-zero variance features; monitored Multicollinearity.
  - **Feature Encoding:** Extracted cyclic temporal features (Month, Hour) and applied OneHotEncoder for categorical `city`.
  - **Feature Scaling:** None applied. Tree-based models (XGBoost) split on thresholds and are scale-invariant (no need for StandardScaler/MinMaxScaler).

## SLIDE 11: Interactive Dashboard Demo (Streamlit & Plotly)
- **Layout:** Split layout (Left text, Right QR Code)
- **Title:** Step 9: Interactive AI Dashboard
- **Left Text:**
  - **Interactive Charts:** 3+ Plotly charts (Timeline Line Chart, SHAP Bar Chart, Alignment Scatter).
  - **Interactivity:** Dropdown filters for Station, Time Horizon, and Wildfire Events.
  - **KPI Cards:** Custom HTML metric cards showing live AQI predictions and deltas.
  - **Geospatial Map:** PyDeck interactive 3D map.
  - **[HIGHLIGHT] RBL & AI Audit Log:** Streamlit selected over Dash for native ML inference. Dashboard UI generated via AI prompts (logged in Audit).
- **Right Image:** 
  - **Top:** QR Code -> `https://starlink.tail334064.ts.net/`
  - **Bottom:** Screenshot of the "Critical Air Stagnation Alert" UI and the "Wildfire event focus" dropdown.](https://starlink.tail334064.ts.net/)

## SLIDE 12: Dataset Partition (Human Delta)
- **Title:** Step 5: Climate-context-aware Temporal Split
- **Text:** 
  - Avoided random `train_test_split` to prevent chronological Data Leakage & Context Leakage.
  - **Training:** 2018, 2019, 2021-2024
  - **Validation (Holdout):** 2020 (August Complex Wildfires)
  - **Testing:** 2025 (Palisades Fire Extreme Year)
- **[HIGHLIGHT: Climatological Justification]**: 2022-2023 saw record rains (Atmospheric Rivers) causing low pollution. If models train purely on recent calm years, they severely underestimate the extreme wildfire spikes of 2025. 
- **[HIGHLIGHT: Human Delta]**: Forcing the model to validate strictly on the 2020 extreme wildfire season prevents this "shock", a domain-knowledge decision standard AI fails to make.

## SLIDE 13: Pipeline & GridSearchCV Tuning
- **Layout:** Title + Image
- **Title:** Step 8 & 9: Sklearn Pipeline & GridSearchCV
- **[IMAGE Placeholder]:** Schematic diagram of the ML pipeline.
- **Caption:** The rigorous leakage-safe pipeline using `sklearn.pipeline.Pipeline`.
- **Left Text:**
  - **Step 8 (GridSearchCV):** Applied TimeSeriesSplit (CV) and GridSearchCV to tune hyperparameters (max_depth, learning_rate, n_estimators) for the best models (XGBoost/LightGBM) to prevent overfitting.
  - **Step 9 (Pipeline Saved):** The entire workflow (imputation, encoding, scaling, modeling) was packaged and saved as a single `.pkl` artifact via `joblib`, ensuring zero data leakage during production inference.
- **[HIGHLIGHT: Chronological Boundary Leakage Guard]**: Enforced strict safety margins to prevent future data from leaking through lag features (Dropped 4 hours at boundaries for Nowcasting, and 215 hours for 24h Forecasting).

## SLIDE 14: Model Evaluation (Model Comparison Template)
- **Layout:** Split Layout (Left Table, Right Image)
- **Title:** Step 6-7: Model Comparison
- **Left Table (Matched to Rubric Template):**
  | Model | RMSE | MAE | R2 | vs Baseline |
  |---|---|---|---|---|
  | Linear Regression (Ridge) | 14.24 | 9.21 | 0.8473 | -0.0027 |
  | Random Forest | 10.09 | 5.95 | 0.8595 | +0.0095 |
  | XGBoost | **9.47** | **5.83** | **0.8711** | **+0.0211** |
  | LightGBM | 9.48 | 6.06 | 0.8706 | +0.0206 |
  | Baseline (Vu et al., 2022) | ~9.70 | ~6.50 | 0.8500 | reference |
  - **Insight:** XGBoost and LightGBM outperform the Baseline Paper. Persistence (Naive) was also evaluated and failed in 24h forecasting.
- **Right Image / Box:** 
  - **[IMAGE Placeholder]:** Screenshot of the Baseline Paper Title/Abstract ("Vu et al. 2022: Estimating surface PM2.5...").
  - **Caption:** Beating the state-of-the-art baseline using pure tabular Machine Learning.

## SLIDE 15: Extreme Scenario Analysis
- **Layout:** Title + Image
- **Title:** Step 7: Extreme-Event Scenario (Top 5% AQI)
- **[IMAGE]:** `c:/FPT/SU2026/DAP391m/Project/data/plots/paper_january_2025_stress_window.png`
- **Text:** Evaluated solely on the Top 5% extreme AQI slices, complex ML models struggle. Surprisingly, **Persistence** emerges as the best performer ($R^2 = 0.30$).
- **Insight:** Short-term particulate matter concentration is overwhelmingly governed by autocorrelation.

## SLIDE 16: Prediction Diagnostics
- **Layout:** Title + Image
- **Title:** Step 8: Prediction Diagnostics
- **[IMAGE]:** `c:/FPT/SU2026/DAP391m/Project/data/plots/paper_xgboost_predicted_vs_actual.png`
- **Caption:** Actual vs Predicted plot. The model captures central AQI ranges perfectly but tends to underestimate the extremely rare hazardous peaks.

## SLIDE 17: Conclusion
- **Title:** Step 10: Conclusion
- **Numbered list:**
  1. **RQ1:** ML models can achieve high accuracy ($R^2 = 0.8711$) under a strict leakage-safe protocol.
  2. **RQ2:** Short-term predictions rely heavily on recent local history, while 24h forecasting forces models to rely on climate-context (VPD).
  3. **RQ3:** Models face stability issues during top 5% extreme events; engineered VPD and spatial lags provide vital physical context but modest global metric gains.

## SLIDE 18: AI Reflection & Hallucinations
- **Title:** AI Audit Log & Human Delta
- **Bullet points:**
  - 2 Excel Audit Logs provided (Tracking > 20 prompts by Minh & Sang).
  - **[HIGHLIGHT: Top 3 AI Hallucinations Caught (Directly from Audit Logs)]**:
    1. **Logic Error (Entry #008):** AI suggested using a standard random train/test split. $\rightarrow$ *Rejected:* This causes data leakage in time-series data. Fixed by implementing a climate-context-aware chronological split.
    2. **Oversimplification (Entry #016):** AI claimed that the VPD feature strongly improves model accuracy. $\rightarrow$ *Rejected:* Ablation study (Table 5) showed removing VPD changed $R^2$ by <0.003. Fixed by rewording VPD as a "dryness context feature."
    3. **Causal Overclaim (Entry #012):** AI stated the Jan 2025 extreme-AQI window was "caused by wildfire smoke." $\rightarrow$ *Rejected:* Dataset lacks external satellite smoke features to prove causality. Fixed by using a neutral term "Winter Extreme-AQI Window."
  - **Key Takeaway:** Domain Knowledge drives the architecture; AI is merely an execution assistant.

## SLIDE 19: Q&A
- **Layout:** Center Alignment
- **Main Text:** Thank You & Q&A Session
- **Sub Text:** 
  - Live Demo: [starlink.tail334064.ts.net](https://starlink.tail334064.ts.net/)
  - Jupyter Notebooks & SQL Data are ready for review.

