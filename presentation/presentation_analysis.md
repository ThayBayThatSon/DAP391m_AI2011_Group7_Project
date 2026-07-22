# DAP391m Presentation Analysis: AQI Forecasting Project

Dựa trên cấu trúc slide mẫu (10 Steps) của môn học DAP391m và thông tin từ dự án của bạn (US AQI Time Series Forecasting in California), dưới đây là bản phân tích và dàn ý chi tiết giúp bạn xây dựng slide báo cáo hoàn hảo.

## PART 1 — Problem & Data Understanding (Steps 1-3)

### Step 1: Problem Understanding
- **Problem Statement:** Dự báo mức độ ô nhiễm không khí (chỉ số AQI) hàng giờ tại 3 thành phố lớn của California (Fresno, Los Angeles, San Jose).
- **Feature:** Dữ liệu khí tượng từ Open-Meteo (nhiệt độ, độ ẩm, VPD, v.v.), chỉ số AQI trong quá khứ (lag features), và các đặc trưng không gian (tọa độ/thành phố).
- **Target:** Chỉ số `AQI` (dạng liên tục, được chuyển đổi từ PM2.5).
- **Classification or Regression?** Đây là bài toán **Regression** (Hồi quy) vì dự báo một giá trị liên tục (AQI).
- **Business Context:** Ô nhiễm không khí và cháy rừng ở California gây ảnh hưởng nghiêm trọng đến sức khỏe cộng đồng. Việc cảnh báo sớm (Nowcasting và Forecasting 24h) giúp chính phủ và người dân có biện pháp phòng tránh kịp thời.
- **Success Metrics:** $R^2$ Score, RMSE, và MAE.

### Research Questions (RQs) - Gợi ý
1. **[RQ1 - Feature/EDA]:** Yếu tố khí tượng và không gian nào (nhiệt độ, VPD, vị trí) ảnh hưởng mạnh nhất đến sự biến động của chỉ số AQI tại California?
2. **[RQ2 - Model]:** Trong số các mô hình học máy (LightGBM, XGBoost, Random Forest...), mô hình nào dự báo chỉ số AQI chính xác nhất cho kịch bản Nowcasting (1-3h) so với kịch bản dài hạn (24h)?
3. **[RQ3 - Pipeline/App]:** Làm thế nào để tự động hóa 파i-p-line kết hợp dữ liệu EPA AQS và Open-Meteo nhằm cung cấp các cảnh báo ô nhiễm (Air Stagnation) theo thời gian thực?

### Step 2: Data Understanding
- **Dataset:** 181,122 bản ghi (observations) dạng panel data từ 2018–2025.
- **Data sources:** U.S. EPA Air Quality System và Open-Meteo.
- **Data Issues (Kỳ vọng):**
  - *Missing/Inconsistent:* Độ trễ báo cáo của AQS (đặc biệt trong tập Test 2025).
  - *Outlier / Skewness:* Các giá trị AQI cực đoan (>95th percentile) xảy ra trong mùa cháy rừng.

### Step 3: Feature Understanding (EDA)
- **Gợi ý biểu đồ:**
  - *Univariate:* Phân phối histogram của AQI (lệch phải, tập trung nhiều ở nhóm AQI thấp, đuôi dài ở nhóm nguy hại do cháy rừng).
  - *Bivariate / Multivariate:* Tương quan giữa VPD (Vapor Pressure Deficit) và mức độ AQI cực đoan.
  - *Geospatial:* Bản đồ phân bố mức độ ô nhiễm giữa Fresno, LA, và San Jose.

---

## PART 2 — Feature Engineering & Visualization (Steps 3-4)

### Step 4: Feature Engineering
- **Time/Lag Features:** Tạo các biến trễ (lag 1-3h cho Nowcasting) và rolling averages.
- **Feature Enrichment:** Tính toán VPD từ nhiệt độ và độ ẩm; các biến theo mùa/tháng để bắt được "Wildfire season".
- **Handling Outliers/Extremes:** Thay vì loại bỏ, hệ thống giữ lại các giá trị AQI cực đoan vì đây là mục tiêu chính của mô hình (Dự báo cháy rừng/ô nhiễm cao).

### Advanced Visualizations & Interactive Dashboard
- **Dashboard (Streamlit):** Giao diện tương tác người dùng `ui.py`.
- **Min. 3 Interactive Charts:**
  1. *Geospatial chart:* Bản đồ nhiệt không gian 3D (PyDeck) hiển thị AQI giữa các thành phố.
  2. *Line/Area Chart (Timeline):* Biểu đồ Plotly timeline so sánh AQI thực tế và dự đoán qua các mốc thời gian (horizon).
  3. *SHAP Feature Importance:* Giải thích nguyên nhân gây ra dự báo cao (Model interpretability).
- **KPI Summary Cards:** Cảnh báo "Air Stagnation Alerts" khi điều kiện khí tượng xấu đi.

---

## PART 3 — Modeling, Evaluation & AI Application (Steps 5-9)

### Step 5: Dataset Partition
- **Cách chia Data cực kỳ sáng tạo (Climate-context-aware temporal split):** Thay vì chia random, dự án chia theo thời gian thực tế để test sự kiện cực đoan:
  - *Train:* 2018, 2019, 2021-2024.
  - *Validation (Holdout):* 2020 (Năm có cháy rừng lịch sử / Extreme AQI).
  - *Test (Out-of-distribution):* 2025 (Dữ liệu mới nhất với hạn chế về trễ báo cáo EPA).

### Step 6: Data Modelling & Pipeline
- **Các thuật toán đã thử:** Linear Ridge, Random Forest, LightGBM, CatBoost, XGBoost.
- **Baselines so sánh:** Persistence (Giá trị hiện tại = Tương lai) và Climatology (Trung bình lịch sử).
- **2 Kịch bản:** Short-term Nowcasting (có sẵn lag 1-3h) vs Long-term Forecasting (strict 24h, giấu hoàn toàn data $t-1$ đến $t-23$).

### Step 7 & 8: Evaluation & Tuning
- **Short-term Nowcasting:** **LightGBM** chiến thắng ($R^2$: 0.8695). Đáng chú ý: mô hình *Persistence* (baseline đơn giản) hoạt động cực tốt trong các đợt bùng phát cháy rừng do tính tự tương quan (autocorrelation) của AQI trong ngắn hạn.
- **Long-term Forecasting (24h):** Baseline sụp đổ hoàn toàn. **XGBoost** vượt lên ($R^2$: 0.4611) nhờ khả năng học được bối cảnh khí hậu/thời tiết thay vì chỉ copy giá trị cũ.

### Step 9: Integrate Pipeline & App
- **Kiến trúc hệ thống (Application Architecture):**
  - Backend: **FastAPI** phục vụ model và SHAP giải thích.
  - DB: **SQLite** cache dữ liệu Open-Meteo & EPA.
  - Frontend: **Streamlit** gọi FastAPI để render Dashboard.

---

## PART 4 — Conclusion & AI Reflection (Step 10 + Q&A)

### Step 10: Conclusion
- **Trả lời RQs:** Yếu tố khí tượng quan trọng đối với dự báo dài hạn (24h), nhưng giá trị quá khứ gần (lag) lại thống trị trong ngắn hạn (Nowcasting).
- **Thực tiễn (Practical Implications):** Mô hình giúp phát ra "Air Stagnation Alerts", hỗ trợ thành phố chuẩn bị y tế trước 24h.
- **Limitations:** Độ trễ API của EPA AQS (latency) làm ảnh hưởng đến hiệu năng thực tế năm 2025.

### Chuẩn bị Q&A & AI Reflection (Cực kỳ quan trọng để lấy điểm tuyệt đối)
1. **Human Delta (Giá trị của con người so với AI):** Bạn cần nhấn mạnh cách nhóm quyết định *thiết kế bộ chia dữ liệu (Dataset Partition)* (Holdout năm 2020 để bắt được bối cảnh cháy rừng) thay vì dùng `train_test_split` ngẫu nhiên thông thường của AI. Sự hiểu biết về "domain knowledge" này là điểm sáng của dự án.
2. **AI Audit Log & Hallucination:** Hãy chắc chắn bạn điền đầy đủ file Excel "AI Audit Log Template.xlsx" (15-20 prompts). Bạn nên ghi nhận lại những lần AI sinh lỗi (Ví dụ: sinh sai shape data khi merge Open-Meteo và EPA, hoặc AI khuyên dùng phân chia ngẫu nhiên nhưng nhóm phản bác) để có đủ >=3 Hallucinations.
3. **Q&A - Hiểu rõ Model Pipeline:** Tại sao lại chia ra Nowcasting và Forecasting? Tại sao Pipeline dùng FastAPI tách biệt với Streamlit? (Trả lời: Để tách biệt logic AI model serving và UI, dễ dàng mở rộng như phần *Future Roadmap*).

> [!TIP]
> **Checklist Thuyết trình:**
> - Slide phải có sơ đồ Kiến trúc (từ file `architecture.md`).
> - Giao diện Demo Streamlit phải chuẩn bị sẵn video/screenshot phòng trường hợp lỗi mạng API Open-Meteo.
> - Bảng so sánh mô hình (Slide 21 trong mẫu) hãy để **LightGBM (ngắn hạn)** và **XGBoost (dài hạn)** màu xanh nổi bật so với Baseline (Persistence).
