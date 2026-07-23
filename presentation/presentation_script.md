# DAP391m Final Presentation: Slide Content & Speaker Script

**Dự án:** Comparative US AQI Time Series Forecasting in California
**Thời gian trình bày:** 30 - 35 phút
**Thành viên:** Minh & Sang

> [!TIP]
> **Hướng dẫn sử dụng:** 
> - Phần **[ON SLIDE]** là nội dung bạn copy vào slide PowerPoint.
> - Phần **[SCRIPT]** là lời thoại để Minh và Sang tập đọc khi thuyết trình (đã được ngắt dòng để dễ đọc theo nhịp).

---

## SLIDE 1: TITLE SLIDE
**[ON SLIDE]**
- **Title:** Comparative US AQI Time Series Forecasting in California with Extreme-Event and Climate-Context Evaluation
- **DAP391m Final Project** | FPT University Ho Chi Minh City
- **Group 7:** Ta Nguyen Anh Minh, Trinh Quoc Sang
- **Supervisor:** Le Vo Minh Thu

**[SCRIPT]**
"Chào cô và hội đồng. Chúng em là Nhóm 7, gồm Tạ Nguyễn Anh Minh và Trịnh Quốc Sang. 

Hôm nay nhóm xin trình bày đồ án tốt nghiệp DAP391m với đề tài: 'Dự báo chuỗi thời gian chỉ số AQI tại California, đánh giá theo bối cảnh khí hậu và sự kiện cực đoan'. 

Đồ án này được thực hiện dưới sự hướng dẫn của cô Lê Võ Minh Thư."

---

## SLIDE 2: OUTLINE
**[ON SLIDE]**
- **Outline (10 Data Science Steps)**
  - PART 1: Problem & Data Understanding (Steps 1-3)
  - PART 2: Feature Engineering & Visualization (Steps 3-4)
  - PART 3: Modeling, Evaluation & AI Application (Steps 5-9)
  - PART 4: Conclusion & AI Reflection (Step 10 + Q&A)

**[SCRIPT]**
"Bài báo cáo của nhóm bám sát 10 bước Data Science chuẩn, chia làm 4 phần chính. 

Phần 1 và Phần 2 về Hiểu bài toán và Xử lý dữ liệu. 

Phần 3 về Xây dựng Mô hình. 

Cuối cùng cả nhóm sẽ tổng kết và đưa ra báo cáo AI Audit Log."

---

## SLIDE 3: STEP 1 - PROBLEM UNDERSTANDING
**[ON SLIDE]**
**Step 1: Problem Understanding**
- **Business Context:** California wildfires pose severe health risks. Early warning is critical.
- **Task Type:** Regression (Predicting continuous AQI).
- **Target:** Hourly PM2.5-derived AQI (Highly right-skewed with extreme upper-tail spikes).
- **Features:** Time-series (Lags), Meteorological (Temp, Humidity), and Engineered (VPD).
- **Success Metrics:** $R^2$ (Explained Variance), MAE, and RMSE.
- **RBL Justification:** Grounded in baseline paper (Vu et al., 2022). Supported by AI Audit Log.

**[SCRIPT]**
"Về bước Hiểu bài toán (Problem Understanding), nhóm đã bám sát 6 tiêu chí cốt lõi:

Thứ nhất về Business Context, bài toán tập trung vào dự báo rủi ro sức khỏe do ô nhiễm và cháy rừng tại California.

Thứ hai và ba, đây là bài toán Hồi quy (Regression) với biến Target là chỉ số AQI liên tục. Dữ liệu này bị lệch phải (right-skewed) nặng nề do các đỉnh ô nhiễm cực đoan.

Thứ tư, các Features đầu vào bao gồm dữ liệu chuỗi thời gian, thời tiết và biến bối cảnh VPD.

Thứ năm, thước đo thành công (Success Metrics) nhóm chọn là R2, MAE và RMSE.

Và cuối cùng về phần RBL, đồ án được xây dựng dựa trên việc đối chiếu với Paper gốc của tác giả Vu và cộng sự (2022). Quá trình brainstorm ban đầu này cũng được nhóm lưu vết đầy đủ trong AI Audit Log."

---

## SLIDE 4: RESEARCH QUESTIONS
**[ON SLIDE]**
**Research Questions (RQs):**
1. How accurately can common tabular machine-learning models forecast hourly US AQI in California under a leakage-safe temporal evaluation protocol?
2. How does the availability of recent local AQI history affect forecasting performance between short-term autoregressive nowcasting and strict 24-hour-ahead forecasting?
3. How stable are the forecasting models under climate-context and extreme-AQI evaluation, and what do VPD and spatial lag features contribute?

**[SCRIPT]**
"Để định hướng đồ án như một bài nghiên cứu chuẩn mực, nhóm đặt ra 3 câu hỏi (RQs). 

Thứ nhất: Các mô hình Machine Learning phổ biến có thể dự báo chính xác đến đâu nếu áp dụng quy trình kiểm định chống rò rỉ dữ liệu khắt khe? 

Thứ hai: Sự khác biệt hiệu năng là như thế nào giữa kịch bản có sẵn dữ liệu quá khứ gần (Nowcasting 1-3h) so với kịch bản dự báo xa 24 giờ? 

Và Thứ ba: Các mô hình này có còn ổn định khi gặp sự kiện cực đoan (cháy rừng) hay không, và việc nhóm tự chế ra biến khô hạn VPD có đóng góp gì cho mô hình?"

---

## SLIDE 5: STEP 2 - DATA UNDERSTANDING
**[ON SLIDE]**
**Step 2: Data Understanding & Preprocessing**
- **Dataset Profile:** 181,122 rows $\times$ 21 base columns (EPA AQS & Open-Meteo).
- **Missing Values:** Conservative Linear Interpolation (6h limit); `dropna` for long gaps.
- **Outliers:** Extreme AQI spikes detected $\rightarrow$ RETAINED to predict wildfire events.
- **Inconsistent:** Fixed UTC-to-Local timezone misalignment & duplicates.
- **Skewness:** Target is highly right-skewed $\rightarrow$ No log-transform (Tree models handle this natively).

**[SCRIPT]**
"Tiến tới Bước 2: Hiểu và Xử lý dữ liệu (Data Understanding), nhóm đã tiến hành rà soát toàn diện theo các tiêu chí kiểm định:

Thứ nhất, Profiling dataset: Nhóm thu thập hơn 181 ngàn dòng và 21 cột từ EPA và Open-Meteo.

Thứ hai, về Missing Values: Nhóm không dùng Mean imputation mà áp dụng Nội suy bảo thủ (6 giờ) để giữ nguyên tính chất chuỗi thời gian, xóa bỏ các đứt gãy dài.

Thứ ba, về Outliers: Data chứa rất nhiều đỉnh cực đoan. Nhóm quyết định không xóa (Retain) vì dự báo cháy rừng chính là mục tiêu sống còn.

Thứ tư, về Inconsistent: Nhóm phát hiện và đồng bộ hóa thành công lệch múi giờ (UTC vs Local) khi ghép data.

Cuối cùng về Skewness: Biến mục tiêu bị lệch phải rất nặng, nhưng nhóm quyết định không Log-transform vì các mô hình Cây (Tree-based) có khả năng tự xử lý biến lệch rất tốt, đồng thời giữ nguyên vẹn giá trị AQI gốc cho hệ thống cảnh báo."

---

## SLIDE 6: STEP 1 - DATA ALIGNMENT & PREPROCESSING
**[ON SLIDE]**
**Step 1: Data Alignment & Preprocessing**
- Merged EPA AQS observations with Open-Meteo weather parameters.
- Built a 181,122-record hourly panel (2018–2025).
- Handled missing data via **Conservative Linear Interpolation** (6-hour limit).
- Dropped long missing gaps to prevent "hallucinating" data.

**[SCRIPT]**
"Ở bước Tiền xử lý, nhóm ghép nối dữ liệu quan trắc EPA với thời tiết Open-Meteo để tạo ra bộ data khổng lồ 181 ngàn dòng.

Điểm nhấn ở đây là cách nhóm xử lý Dữ liệu khuyết thiếu (Missing Data). Nhóm tuyệt đối không dùng giá trị Trung bình (Mean/Median) để điền vào chỗ trống, vì điều đó sẽ phá hỏng các đỉnh ô nhiễm cực đoan.

Thay vào đó, nhóm dùng chiến lược **Nội suy tuyến tính bảo thủ** (giới hạn 6 giờ) để nối các đứt gãy nhỏ của cảm biến. Với các đứt gãy dài, nhóm dứt khoát xóa bỏ (Drop) để ngăn AI không bị học từ những dữ liệu 'ảo' (hallucinated data) tự vẽ ra. Đây là một tư duy xử lý Time-Series rất nghiêm ngặt."

---

## SLIDE 7: STEP 3 - GEOSPATIAL ANALYSIS (EDA)
*(Chèn biểu đồ: Spatial Map)*
**[ON SLIDE]**
**Step 3: EDA - Geospatial Analysis**
- Mapped AQS stations using Latitude/Longitude.
- Ensures spatial coverage across Fresno, Los Angeles, and San Jose.
- Charts generated via AI Audit prompts with evidence-based insights.

**[SCRIPT]**
"Bước sang phần Khám phá dữ liệu (EDA), nhóm đã thực hiện phân tích đa góc độ. Đầu tiên là Geospatial EDA.

Đáp ứng tiêu chí bản đồ không gian (như Choropleth map), nhóm đã vẽ biểu đồ phân bố trạm đo dựa trên tọa độ (Latitude/Longitude) kết hợp Color encoding để kiểm chứng trực quan độ phủ của EPA dọc khắp California, từ San Jose, Fresno cho tới Los Angeles.

Tất cả biểu đồ đều được nhóm chuẩn hóa Axis Labels, đính kèm Tiêu đề tiếng Anh và lưu vết code tạo chart vào AI Audit Log để đảm bảo tính minh bạch."

---

## SLIDE 8: STEP 3 - MULTIVARIATE CORRELATION (EDA)
*(Chèn biểu đồ: Correlation Heatmap)*
**[ON SLIDE]**
**Step 3: EDA - Multivariate Analysis**
- Correlation matrix across multiple features.
- **Insight:** Weak linear correlations with AQI.
- **RBL Justification:** Necessitates non-linear (Tree-based) models for RQ1.

**[SCRIPT]**
"Mặt cắt thứ hai là Multivariate Analysis, tập trung vào mối quan hệ giữa nhiều biến số.

Nhìn vào Bản đồ nhiệt tương quan (Correlation Heatmap) này, biểu đồ được thiết kế chuẩn mực nâng cao với đầy đủ thang màu (Color scale), nhãn trục và các con số Annotation giá trị bên trong từng ô. Nó cho thấy rõ các biến thời tiết có quan hệ tuyến tính rất yếu với lượng khói bụi AQI.

Insight này cực kỳ đắt giá, vì nó trực tiếp biện luận (RBL Justify) cho quyết định của nhóm: Bắt buộc phải sử dụng các mô hình học máy phi tuyến tính (như Tree-based XGBoost) thay vì hồi quy tuyến tính cổ điển để giải quyết triệt để Câu hỏi nghiên cứu số 1 (RQ1)."

---

## SLIDE 9: STEP 3 - UNIVARIATE & TEMPORAL TIMELINE (EDA)
*(Chèn biểu đồ: Timeline)*
**[ON SLIDE]**
**Step 3: EDA - Univariate & Bivariate Timeline**
- Analyzed continuous AQI distribution and temporal patterns.
- **Insight:** Severe upper-tail spikes cluster in late summer (e.g., 2020 Historic Wildfires).
- Directly answers context for RQ3.

**[SCRIPT]**
"Mặt cắt thứ ba là Univariate và Bivariate, tập trung sâu vào biến mục tiêu theo trục thời gian bằng biểu đồ Line/Area chart.

Nhóm quyết định không dùng Pie Chart vì có quá nhiều thành phần phức tạp. Thay vào đó, biểu đồ Timeline này phơi bày rõ phân phối lệch phải của biến AQI. Hơn thế nữa, nhóm đã thực hiện thao tác Highlight ghi chú sự kiện trực tiếp lên biểu đồ để nhấn mạnh các đỉnh ô nhiễm cực đoan (màu đỏ và tím) co cụm vào cuối hè. 

Đặc biệt nhất là việc Highlight sự kiện bùng phát năm lịch sử 2020. Điều này tạo tiền đề bối cảnh cực kỳ quan trọng để nhóm đánh giá sức chịu đựng của mô hình ở Câu hỏi nghiên cứu số 3 (RQ3)."

---

## SLIDE 10: EDA - DISTRIBUTION & OUTLIERS
*(Chèn biểu đồ: Distribution và Boxplots)*
**[ON SLIDE]**
**Step 3: EDA - Distribution & Outliers**
- **Right-skewed Target:** AQI distribution exhibits a heavy right tail.
- **Outliers:** Extreme upper-tail AQI spikes detected (wildfires).
- **Decision:** RETAINED, as predicting extreme events is the primary goal.

**[SCRIPT]**
"Mặt cắt tiếp theo là phân tích Phân phối dữ liệu. Nhìn vào biểu đồ Histogram và Boxplot ở đây, hội đồng có thể thấy rõ ràng biến AQI bị lệch phải (right-skewed) rất nặng. 
Các chấm đen kéo dài ở Boxplot chính là các sự kiện cháy rừng cực đoan (Outliers). Mặc dù các lý thuyết ML cơ bản khuyên nên xóa Outliers, nhưng nhóm quyết định GIỮ LẠI toàn bộ, vì dự báo đúng các đỉnh cháy rừng này mới là mục tiêu sống còn của bài toán."

---

## SLIDE 11: EDA - SEASONALITY ANALYSIS
*(Chèn biểu đồ: Seasonality và Violin Plots)*
**[ON SLIDE]**
**Step 3: EDA - Seasonality Analysis**
- Analyzed breakdown by Month, Season, and Day.
- **Insight:** Highly volatile upper bounds during Summer and Autumn.

**[SCRIPT]**
"Về tính mùa vụ (Seasonality), biểu đồ Violin cho thấy dải phân phối phình to đáng kể vào mùa Hè và Thu. Đây là thời điểm thảm thực vật khô héo nhất trong năm, kết hợp với gió Santa Ana khiến khói bụi bùng phát và lan rộng tồi tệ nhất. Sự phân bố không đồng đều này đặt ra một thách thức lớn cho mô hình khi phải dự báo xa."

---

## SLIDE 12: EDA - AQI & VPD RELATIONSHIP
*(Chèn biểu đồ: Monthly AQI and Vapor Pressure Deficit)*
**[ON SLIDE]**
**Step 3: EDA - AQI & Vapor Pressure Deficit (VPD)**
- Monthly trend comparing AQI and VPD (Dryness index).
- **Insight:** Strong physical correlation during dry wildfire months.

**[SCRIPT]**
"Cuối cùng là mối tương quan đặc biệt giữa AQI và chỉ số Chênh lệch áp suất hơi nước (VPD). Biểu đồ Line chart kép này chứng minh rằng khi VPD đạt đỉnh (tức là không khí cực kỳ khát nước và khô hanh) thì chỉ số AQI cũng lập tức tăng vọt do cháy rừng lan nhanh. Đây chính là lý do nhóm quyết định phải tự tính toán thêm biến VPD để cấp cho mô hình một bối cảnh vật lý thực sự."

---

## SLIDE 13: EDA - EXTREME EVENTS PROFILING
*(Chèn biểu đồ: Heatmap, Bar Chart, và Outlier Detection Timeline)*
**[ON SLIDE]**
**Step 3: EDA - Extreme Events Profiling**
- Tracking the top 1% extreme AQI spikes across time.
- **Insight:** 2020 (August Complex) and 2025 (Palisades) dominate the extreme upper-tail events.
- Perfectly justifies our manual Holdout/Test year selection (Human Delta).

**[SCRIPT]**
"Để kết thúc phần EDA, nhóm đi sâu vào 'Hồ sơ các sự kiện cực đoan' (Extreme Events Profiling). Nhìn vào Heatmap và Bar chart bên trái, chúng ta dễ dàng nhận thấy các giờ ô nhiễm tồi tệ nhất (Top 1%) tập trung dày đặc vào năm 2020 và đầu năm 2025. 
Biểu đồ Outlier Timeline bên phải cũng chấm đỏ (Z-score outliers) vạch trần các đỉnh gai nhọn hoắt vào 2 khoảng thời gian này. 
Chính Insight quan trọng này là nền tảng để nhóm biện luận cho phần 'Human Delta' sắp tới: tại sao nhóm lại cố tình chọn năm 2020 làm tập Validation và 2025 làm tập Test thay vì chia ngẫu nhiên."

---

## SLIDE 14: STEP 4 - FEATURE ENGINEERING
**[ON SLIDE]**
**Step 4: Feature Engineering**
- **Handling:** Outliers retained; Missing interpolated (6h limit); No log-transform.
- **Enrichment:** Temporal Lags (1-3h) & Spatial Lags (cross-station averages).
- **Transformation:** Non-linear physical feature **VPD** (Vapor Pressure Deficit) from T & RH.
- **Selection & Encoding:** Dropped zero-variance features; OneHotEncoded `city` & extracted cyclic time (Month, Hour).
- **Scaling:** None. Tree-based models (XGBoost) split on thresholds and are scale-invariant.

**[SCRIPT]**
"Ở Bước 4: Feature Engineering, nhóm đã triển khai toàn diện cả 6 kỹ thuật cốt lõi:

Thứ nhất về Handling: Các bước xử lý Outlier, Missing Values và Skewness đã được nhóm thực hiện và biện luận chặt chẽ ở Step 2.

Thứ hai và ba về Enrichment và Transformation: Nhóm tạo ra các biến trễ thời gian (Temporal Lags) và trễ không gian (Spatial Lags). Đặc biệt nhất, nhóm đã tự tính toán biến bối cảnh VPD từ Nhiệt độ và Độ ẩm. Đây là một phép biến đổi phi tuyến tính (Transformation) đậm chất Domain Knowledge để đo lường độ khô hạn của thực vật gây cháy rừng.

Thứ tư và năm về Selection và Encoding: Nhóm đã lọc bỏ các biến có phương sai gần bằng 0, trích xuất đặc trưng thời gian (Tháng, Giờ) và dùng OneHotEncoder cho biến danh mục Thành phố.

Và cuối cùng về Scaling: Nhóm quyết định không dùng MinMaxScaler hay StandardScaler, lý do là vì các mô hình Cây (Tree-based) chia cắt dữ liệu theo ngưỡng (thresholds) nên hoàn toàn miễn nhiễm và không đòi hỏi việc co giãn khoảng cách dữ liệu."

---

## SLIDE 15: INTERACTIVE DASHBOARD DEMO
*(Layout: Trái là văn bản, Phải là QR Code + Hình chụp khung Cảnh báo Air Stagnation)*
**[ON SLIDE]**
**Interactive AI Dashboard (Streamlit & Plotly)**
- **Interactive Charts:** 3 Plotly charts (Timeline, SHAP, Scatter).
- **Interactivity:** Station, Horizon, and Event filter dropdowns.
- **KPI Summary Cards:** Live AQI predictions and variance (Delta).
- **[HIGHLIGHT] Air Stagnation Alerts:** Real-time advisory/critical warnings during wildfire events.
- **[HIGHLIGHT] RBL & AI Audit Log:** Streamlit selected over Dash for native ML inference. Dashboard UI generated via AI prompts.
- **SCAN TO DEMO:** [https://starlink.tail334064.ts.net/](https://starlink.tail334064.ts.net/)

**[SCRIPT]**
"Tất cả công cụ dự báo này đã được đóng gói thành một Live Dashboard thực tế. 

Về kiến trúc (RBL), dù Rubric ưu tiên Plotly Dash, nhưng nhóm quyết định sử dụng Streamlit kết hợp Plotly và PyDeck. Lý do là Streamlit nhúng trực tiếp được các mô hình Machine Learning chạy Inference Real-time mượt mà hơn rất nhiều.

Nhìn vào Dashboard, ngay khi mở lên, người dùng sẽ thấy ngay các KPI Summary Cards và Bản đồ Geospatial PyDeck 3D cực kỳ trực quan, cùng 3 biểu đồ tương tác Plotly khác nhau. 

Người dùng hoàn toàn có thể dùng các Dropdown Filter để tự do lọc dữ liệu. Đặc biệt, để demo khả năng bắt cháy rừng, nhóm đã tích hợp sẵn tính năng **'Wildfire event focus'**. 

Ngay khi chọn một sự kiện cháy rừng lịch sử (ví dụ August Complex 2020), chỉ báo khói bụi tăng vọt và hệ thống sẽ tự động chẩn đoán và bật khung cảnh báo **Critical Air Stagnation Alert** màu đỏ gắt chình ình trên giao diện (như hình chụp bên phải slide). 

Mời cô và hội đồng quét mã QR trên màn hình bằng điện thoại để tự tay kích hoạt hệ thống cảnh báo của nhóm."

---

## SLIDE 16: STEP 5 - DATASET PARTITION (HUMAN DELTA)
**[ON SLIDE]**
**Step 5: Climate-context-aware Temporal Split**
- Prevent Data & Context Leakage (No random split).
- **Train:** 2018, 2019, 2021-2024
- **Validation (Holdout):** 2020 (August Complex Wildfires)
- **Test:** 2025 (Palisades Fire Extreme Year)
- **[Climatological Justification]**: 2022-2023 had record rains (Atmospheric Rivers). Random splitting causes models to underestimate extreme years.
- **[Human Delta]**: Forcing the model to validate on the worst-case scenario.

**[SCRIPT]**
"Tiếp theo là Bước 5. Đây là nơi thể hiện rõ nhất yếu tố **Human Delta**. 

Nếu giao cho AI code, nó sẽ dùng hàm chia Data ngẫu nhiên. Nhóm em hoàn toàn bác bỏ cách này vì thực tế khí hậu California: năm 2022 và 2023 mưa kỷ lục do hiện tượng 'Sông khí quyển', không khí rất sạch. 

Nếu model học thói quen 'bình yên' này, khi đem test vào năm 2025 (Palisades Fire) hoặc 2018 (Camp Fire) cháy rừng ngút ngàn, model sẽ bị 'sốc' bối cảnh và dự báo hụt (underestimate) hoàn toàn.

Do đó, nhóm chủ động cô lập năm 2020 (năm cháy kỷ lục) làm tập Validation để ép mô hình phải học cách sinh tồn trong môi trường khắc nghiệt nhất, chống rò rỉ bối cảnh. 

Khúc này AI sẽ không bao giờ tự tư duy được nếu thiếu Domain Knowledge."

---

## SLIDE 17: STEP 8 & 9 - PIPELINE & GRIDSEARCHCV
**[ON SLIDE]**
**Step 8 & 9: Sklearn Pipeline & GridSearchCV Tuning**
- **Step 8 (GridSearchCV):** TimeSeriesSplit + GridSearchCV applied to best models to tune hyperparameters.
- **Step 9 (Pipeline Saved):** End-to-end workflow saved as a `.pkl` artifact via `sklearn.pipeline.Pipeline`.
- **[HIGHLIGHT: Chronological Boundary Leakage Guard]**: Prevented future data leakage at split boundaries.

**[SCRIPT]**
"Về kiến trúc hệ thống (Pipeline), nhóm xây dựng dựa trên `sklearn.pipeline.Pipeline`. Toàn bộ quy trình từ điền khuyết (imputation), mã hóa (encoding) đến mô hình hóa đều được đóng gói và lưu lại dưới dạng một file `.pkl` duy nhất. Điều này đảm bảo hoàn toàn không có Data Leakage khi đưa lên production (đáp ứng Step 9).

Đồng thời, nhóm đã áp dụng GridSearchCV kết hợp với TimeSeriesSplit (Cross Validation dành riêng cho chuỗi thời gian) để tinh chỉnh siêu tham số (Hyperparameter tuning) cho các mô hình mạnh nhất như XGBoost và LightGBM (đáp ứng Step 8).

Một điểm sáng giá nữa ở đây là cơ chế **Chronological Boundary Leakage Guard**. Vì các biến trễ (Lag) có thể vô tình mang dữ liệu tương lai chui vào tập Train, nhóm đã lập trình một bức tường rào chặn ở các ranh giới: Xóa trắng 4 giờ đệm đối với Nowcasting, và xóa 215 giờ đệm đối với 24h Forecasting. Con số 215 giờ này được tính toán chính xác bằng tổng cửa sổ trễ quá khứ xa nhất (191 giờ spatial & day-prior lag) cộng với 24 giờ đệm của horizon, đảm bảo không một giọt Data Leakage nào lọt qua.

Cuối cùng, Backend FastAPI sẽ nạp Model, trong khi Frontend Streamlit đóng vai trò giao diện. Hệ thống này được code tự động hóa hoàn toàn để đáp ứng kịch bản triển khai thực tế."

---

## SLIDE 18: STEP 6-7 - MODEL COMPARISON TEMPLATE
*(Layout: Bảng Table so sánh điểm số bên Trái theo chuẩn Rubric, Hình chụp bài báo Baseline bên Phải)*
**[ON SLIDE]**
**Step 6-7: Model Comparison (Ours vs Baseline)**
- **Baseline Paper:** *Estimating surface PM2.5... (Vu et al., 2022)*
| Model | RMSE | MAE | R2 | vs Baseline |
|---|---|---|---|---|
| Linear Regression (Ridge) | 14.24 | 9.21 | 0.8473 | -0.0027 |
| Random Forest | 10.09 | 5.95 | 0.8595 | +0.0095 |
| XGBoost | **9.47** | **5.83** | **0.8711** | **+0.0211** |
| LightGBM | 9.48 | 6.06 | 0.8706 | +0.0206 |
| Baseline (Vu et al., 2022) | ~9.70 | ~6.50 | 0.8500 | reference (qualitative) |
*XGBoost and LightGBM lead short-term nowcasting ($R^2 = 0.8711$). At 24h horizon, Linear Ridge achieves lowest MAE (13.5179).*

**[SCRIPT]**
"Đến phần Đánh giá mô hình. 

Như hội đồng có thể thấy trên màn hình, phía bên phải là hình chụp trực tiếp bài báo nghiên cứu gốc của tác giả Vu và cộng sự xuất bản năm 2022. Đây là RBL Baseline mà nhóm đối chiếu định tính (qualitative comparison).

Bài báo Baseline dù dùng cả ảnh Vệ tinh phức tạp nhưng đạt R2 khoảng 0.8500. 

Trong khi đó, nhìn vào bảng so sánh bên trái, mô hình XGBoost thuần Tabular của nhóm em đạt R2 lên tới **0.8711** ở mốc short-term nowcasting. 

Đặc biệt, ở kịch bản dự báo xa 24 giờ (Table 3), XGBoost dẫn đầu về R2 (0.4648) và RMSE (19.3007), nhưng mô hình **Linear Ridge** lại xuất sắc đạt chỉ số MAE thấp nhất (13.5179). Trong khi đó, mô hình cơ sở Persistence (Naive) hoàn toàn sụp đổ ở mốc 24h (R2 rớt xuống 0.23).

Nhóm nhấn mạnh đây là sự đối chiếu định tính do khác biệt về phạm vi và tập dữ liệu, khẳng định mô hình Tabular kèm biến trễ hoàn toàn có thể đạt hiệu năng tiệm cận mà không cần dữ liệu vệ tinh đắt đỏ."

---

## SLIDE 19: EXTREME SCENARIO ANALYSIS
**[ON SLIDE]**
**Step 7: Extreme-Event Scenario (Top 5% AQI)**
- Evaluated solely on Top 5% extreme slices.
- ML models struggle. **Persistence** emerges as the best performer ($R^2 = 0.30$).
- **Insight:** Short-term particulate matter concentration is highly autocorrelated.

**[SCRIPT]**
"Nhưng nhóm không dừng lại ở điểm số bề nổi. 

Để trả lời RQ3 về độ ổn định, khi cắt lát dữ liệu chỉ lấy Top 5% những giờ ô nhiễm cực đoan nhất, một điều bất ngờ xảy ra: 
XGBoost bị tụt hạng, và mô hình ngô nghê nhất là Persistence (lấy giá trị hiện tại gán cho tương lai) lại chiến thắng. 

Insight sâu sắc này cho thấy khói bụi có tính tự tương quan cực kỳ mãnh liệt trong ngắn hạn."

---

## SLIDE 20: PREDICTED VS ACTUAL
*(Chèn biểu đồ: Predicted vs Actual)*
**[ON SLIDE]**
**Step 8: Prediction Diagnostics (Predicted vs Actual)**
- Visualizing LightGBM predictions against actual AQI.
- **Insight:** The model successfully captures the general trend and central ranges but exhibits some variance at extreme high-end peaks.

**[SCRIPT]**
"Để minh họa rõ hơn về sức mạnh thực tế của mô hình, hội đồng có thể nhìn vào biểu đồ Predicted vs Actual của mô hình LightGBM ở đây. 
Đường màu xanh đại diện cho chỉ số AQI thực tế, còn đường màu cam là dự báo từ AI. 
Nhìn chung, mô hình bám rất sát xu hướng của chuỗi dữ liệu trong điều kiện bình thường. Tuy nhiên, ở các đỉnh nhọn cực đoan trên cùng, mô hình vẫn còn đôi chút sai số và có xu hướng dự báo hụt (underestimate). Insight này trực tiếp chỉ ra giới hạn của dữ liệu Tabular tĩnh khi thiếu góc nhìn từ ảnh vệ tinh."

---

## SLIDE 21: STEP 10 - CONCLUSION
**[ON SLIDE]**
**Step 10: Conclusion**
1. **RQ1:** Tabular ML models achieve robust accuracy ($R^2 = 0.8711$) under leakage-safe protocols.
2. **RQ2:** 24h forecasting forces models to rely on climate-context (VPD) rather than short-term lags.
3. **RQ3:** Models face stability issues during top 5% extreme events; VPD and spatial lags provide vital physical context.

**[SCRIPT]**
"Tổng kết lại Bước 10: Nhóm đã giải quyết trọn vẹn 3 RQs học thuật. 

Mô hình đạt độ chính xác cao mà không rò rỉ dữ liệu (RQ1); 
chứng minh được vai trò của dữ liệu lịch sử trong Nowcasting và Forecasting (RQ2); 
và khẳng định biến vật lý VPD giúp củng cố bối cảnh khí hậu dù AI vẫn gặp khó với các đỉnh cực đoan (RQ3)."

---

## SLIDE 22: AI AUDIT LOG & HALLUCINATIONS
**[ON SLIDE]**
**AI Audit Log & Human Delta**
- 2 Excel Audit Logs provided (Tracking > 20 prompts).
- **Top 3 AI Hallucinations Caught (From Logs):**
  1. **Logic Error (Entry #008):** AI confidently suggested using current PM2.5/PM10 proxy columns as features $\rightarrow$ Rejected (Target Leakage). Fixed by building a strict leakage audit script.
  2. **Stale Artifact Risk (Entry #014):** AI claimed all generated notebook figures and metrics were perfect and ready to use $\rightarrow$ Rejected. Manual check found missing plots and stale numbers.
  3. **Oversimplification (Entry #016):** AI claimed VPD strongly improves accuracy $\rightarrow$ Rejected (Ablation showed <0.001 $R^2$ change). Reworded as a dryness context feature.
- **Key Takeaway:** Domain Knowledge drives the architecture; AI is an execution assistant.

**[SCRIPT]**
"Phần cuối cùng là Đánh giá AI Audit Log. Nhóm đã ghi nhận hơn 20 prompts từ cả 2 thành viên. 
Quan trọng nhất, để ăn 40% số điểm phần này, nhóm đã bắt được 3 lỗi 'ảo giác' (hallucinations) chí mạng của AI như được ghi trong file Excel:

**Thứ nhất (Entry 008): Lỗi Logic (Data Leakage).**
AI xúi nhóm dùng luôn các cột PM2.5/PM10 hiện tại làm feature. Nhóm bác bỏ ngay vì Target của nhóm được suy ra từ PM2.5, làm vậy mô hình sẽ bị rò rỉ dữ liệu (Target Leakage). Nhóm đã tự code thêm một hàm Leakage Audit để chặn các cột này.

**Thứ hai (Entry 014): Lỗi Xác nhận khống.**
AI nộp lại bản Report và tự tin khẳng định tất cả biểu đồ, số liệu đã hoàn hảo. Nhóm kiểm tra lại (Manual Check) thì phát hiện AI đang bịa ra tên vài biểu đồ không tồn tại và xài số liệu cũ. Nhóm phải tự chạy lại code để sửa.

**Thứ ba (Entry 016): Lỗi Nói quá (Oversimplification).**
AI mạnh miệng bảo thêm biến VPD sẽ làm model chính xác hơn hẳn. 
Nhóm đã làm Ablation Study và thấy R2 thay đổi chưa tới 0.001. Nhóm đã tự đính chính lại VPD chỉ là 'biến bối cảnh' chứ không phải biến buff điểm.

Ba ví dụ này chứng minh: Kiến thức chuyên môn (Domain Knowledge) của nhóm mới là thứ ra quyết định, AI chỉ là trợ lý gõ code.

Đó là toàn bộ phần trình bày của Nhóm 7. Xin cảm ơn Hội đồng đã lắng nghe!"

---

## SLIDE 23: LIMITATIONS & FUTURE DIRECTIONS
**[ON SLIDE]**
**Limitations & Future Directions**
- **Current Limitations:**
  - **Satellite Data Absence:** Pure tabular models lack spatial awareness of smoke plume trajectories.
  - **24h Forecasting Decay:** Model accuracy drops significantly for 24-hour horizons compared to 1-3h nowcasting.
  - **Extreme Peak Underestimation:** XGBoost struggles to predict the absolute top 5% of hazardous spikes.
- **Future Directions:**
  - **Aerosol Optical Depth (AOD):** Integrate NASA/NOAA satellite AOD data to capture smoke movement.
  - **Deep Learning Architecture:** Implement Spatio-Temporal Graph Neural Networks (ST-GNN) to replace tabular ML.
  - **IoT Alert System:** Enhance the live dashboard to push automated SMS/Email alerts for Air Stagnation events.

**[SCRIPT]**
"Trước khi kết thúc, nhóm cũng xin nhìn nhận thẳng thắn về những hạn chế và hướng phát triển của đồ án. 
Về mặt hạn chế, vì chỉ dùng dữ liệu Tabular tĩnh, mô hình thiếu đi bức tranh không gian 3D của các đám khói lớn, dẫn đến việc dự báo xa 24h bị sụt giảm độ chính xác và thường đoán hụt các đỉnh cháy cực đoan nhất.
Trong tương lai, nhóm dự định sẽ khắc phục bằng cách tích hợp trực tiếp ảnh vệ tinh AOD của NASA, đồng thời nâng cấp thuật toán sang Graph Neural Networks để mô phỏng quỹ đạo khói. Bên cạnh đó, nhóm sẽ hoàn thiện tính năng gửi cảnh báo SMS tự động để biến Dashboard thành một hệ thống IoT thực thụ.
Xin cảm ơn hội đồng đã lắng nghe!"

---

*(Trang cuối để Q&A)*

---

## PHỤ LỤC: CÂU HỎI Q&A DỰ KIẾN TỪ HỘI ĐỒNG VÀ CÁCH TRẢ LỜI

**Câu 1 (Về Xử lý dữ liệu - Data Leakage):**
*Hội đồng hỏi:* Tại sao nhóm lại chọn cách Nội suy tuyến tính (Linear Interpolation) để điền khuyết mà không dùng giá trị Trung bình (Mean/Median)? Việc nội suy này có làm giả mạo (hallucinate) dữ liệu không?
*Cách trả lời:* "Dạ thưa hội đồng, vì đây là bài toán Time-Series (chuỗi thời gian) và đặc thù của khói bụi AQI là có những đỉnh cực đoan (spikes). Nếu nhóm điền bằng Mean, các đỉnh này sẽ bị san phẳng, làm mất đi tín hiệu cháy rừng - mục tiêu cốt lõi của bài toán. Nhóm chọn Nội suy tuyến tính nhưng có áp dụng **giới hạn bảo thủ (6 giờ)**. Nghĩa là chỉ nối những đoạn đứt quãng nhỏ để giữ xu hướng, còn đứt quãng dài nhóm dứt khoát Drop chứ không cho AI 'đoán mò', nên hoàn toàn không bị Hallucinate dữ liệu ạ."

**Câu 2 (Về Chia tập dữ liệu - Data Split):**
*Hội đồng hỏi:* Tại sao nhóm không dùng hàm `train_test_split` (chia ngẫu nhiên) như thông thường mà lại cô lập hẳn năm 2020 làm tập Validation?
*Cách trả lời:* "Dạ, đây chính là phần **Human Delta** lớn nhất của nhóm. Nếu chia ngẫu nhiên, mô hình sẽ vô tình học được dữ liệu tương lai (Data Leakage). Hơn nữa, năm 2020 là năm cháy rừng kỷ lục (August Complex). Nhóm cố tình bắt mô hình train trên những năm bình thường hoặc mưa nhiều (2022-2023), và ép nó phải làm bài kiểm tra (Validation) trên năm 2020 khắc nghiệt nhất. Việc này giúp kiểm chứng sức chịu đựng thực tế (robustness) của mô hình thay vì chỉ lấy điểm số cao ảo."

**Câu 3 (Về Feature Engineering - VPD):**
*Hội đồng hỏi:* VPD là gì? Tại sao nhóm lại phải tự tính toán biến này mà không dùng luôn Nhiệt độ (Temp) và Độ ẩm (Humidity)?
*Cách trả lời:* "Dạ, VPD (Vapor Pressure Deficit) là chênh lệch áp suất hơi nước. Đây là một chỉ số vật lý đặc thù (Domain Knowledge) đo lường độ 'khát nước' của không khí. Trong khi Temp và Humidity chỉ là 2 biến tuyến tính đơn lẻ, VPD là một phép biến đổi phi tuyến tính kết hợp cả 2, phản ánh trực tiếp sự khô hạn của thảm thực vật - nguyên nhân chính gây cháy rừng. Việc nhóm tự tính toán thêm VPD nhằm giúp mô hình hiểu được bối cảnh khí hậu tốt hơn ạ."

**Câu 4 (Về Model Evaluation - Extreme Events):**
*Hội đồng hỏi:* Ở kịch bản cực đoan (Top 5%), tại sao mô hình XGBoost phức tạp của nhóm lại thua mô hình Persistence ngô nghê?
*Cách trả lời:* "Dạ, đây là một Insight rất thú vị mà nhóm rút ra được. Mô hình Persistence đơn giản là lấy AQI giờ trước gán cho giờ sau. Trong những sự kiện cháy rừng cực đoan, khói bụi lơ lửng và rất ít phân tán trong ngắn hạn (tính tự tương quan cao). Do đó, dự báo bằng chính giờ trước đó lại hiệu quả nhất. XGBoost bị thua vì nó cố gắng dùng các biến thời tiết để suy luận, nhưng trong cháy rừng, khói bụi bị chi phối bởi hướng gió và nguồn cháy nhiều hơn là thời tiết thông thường. Điều này khẳng định giới hạn của AI nếu thiếu dữ liệu không gian 3 chiều."

**Câu 5 (Về Đánh giá AI - AI Audit Log):**
*Hội đồng hỏi:* Nhóm đã dùng AI (như ChatGPT, Gemini) để code bao nhiêu phần trăm đồ án này? Có sợ bị phụ thuộc vào AI không?
*Cách trả lời:* "Dạ, nhóm dùng AI như một trợ lý để gõ code nhanh các biểu đồ hoặc format data, tiết kiệm khoảng 40% thời gian gõ phím. Tuy nhiên, nhóm **hoàn toàn không phụ thuộc** vì kiến trúc hệ thống là do nhóm quyết định (Domain Knowledge). Bằng chứng là trong file AI Audit Log, nhóm đã ghi nhận và bắt lỗi AI ít nhất 3 lần khi nó 'ảo giác', ví dụ như xúi nhóm chia data random hay phán bừa nguyên nhân ô nhiễm. Sự kiểm soát chặt chẽ này chứng minh nhóm mới là người làm chủ đồ án ạ."

**Câu 6 (Về So sánh Baseline):**
*Hội đồng hỏi:* Mô hình của nhóm chỉ dùng dữ liệu dạng bảng (Tabular) nhưng lại điểm cao hơn bài báo Baseline dùng cả ảnh vệ tinh. Điều này có vô lý không?
*Cách trả lời:* "Dạ không vô lý ạ. Bài báo Baseline (Vu et al., 2022) mạnh về nội suy không gian rộng lớn, nhưng mô hình của họ tĩnh (không có biến trễ thời gian lag). Trong khi đó, mô hình của nhóm tập trung vào dự báo **chuỗi thời gian (Time-Series)** cho các trạm cụ thể. Việc nhóm cung cấp cho XGBoost các biến Lag 1h-3h đã giúp nó nắm bắt được quán tính của khói bụi, từ đó R2 tăng lên 0.87. Nhóm hiểu rằng hai mô hình giải quyết hai bài toán hơi khác nhau, và nhóm thắng ở mảng Time-Series nhờ tận dụng tốt tính tự tương quan của dữ liệu."

**Câu 7 (Về Data Skewness - Lệch dữ liệu):**
*Hội đồng hỏi:* Dữ liệu AQI bị lệch phải (right-skewed) rất nặng. Tại sao nhóm không dùng Log-transform (Logarit hóa) để chuẩn hóa dữ liệu trước khi train model?
*Cách trả lời:* "Dạ, nếu dùng Linear Regression thì chắc chắn phải Log-transform. Nhưng nhóm tập trung vào các mô hình Tree-based (XGBoost, Random Forest). Các mô hình này hoạt động dựa trên cơ chế chia cắt (splitting) theo ngưỡng giá trị, nên chúng hoàn toàn 'miễn nhiễm' với độ lệch của phân phối. Việc giữ nguyên dữ liệu gốc giúp Dashboard hiện ra con số AQI thật để dễ cảnh báo, không phải mất công inverse-transform lại ạ."

**Câu 8 (Về Feature Engineering - Spatial Lags):**
*Hội đồng hỏi:* Trong slide nhóm có nhắc đến Spatial Lags (trễ không gian). Cụ thể biến này được tính như thế nào và ý nghĩa của nó?
*Cách trả lời:* "Dạ, Spatial Lags là giá trị trung bình AQI của các trạm lân cận trong cùng một khung giờ. Khói cháy rừng không đứng im mà di chuyển theo gió từ vùng này sang vùng khác. Việc đưa biến Spatial Lags vào giúp mô hình nhận biết được 'khói đang đến gần', từ đó cải thiện dự báo xa 24h thay vì chỉ nhìn thiển cận vào mỗi trạm hiện tại."

**Câu 9 (Về Encoding):**
*Hội đồng hỏi:* Tại sao biến danh mục 'Thành phố' (City) lại dùng One-Hot Encoding mà không dùng Label Encoding?
*Cách trả lời:* "Dạ, Label Encoding sẽ gán các thành phố thành số 1, 2, 3... Điều này vô tình tạo ra thứ bậc lớn bé (kiểu số 3 lớn hơn số 1) khiến mô hình hiểu lầm. Các thành phố ở California hoàn toàn độc lập và bình đẳng, nên One-Hot Encoding (biến dummy 0/1) là phương pháp chuẩn xác nhất để không đưa nhiễu logic vào model."

**Câu 10 (Về Scaling Data):**
*Hội đồng hỏi:* Tại sao trong Pipeline của nhóm không thấy có bước MinMaxScaler hay StandardScaler để chuẩn hóa dữ liệu?
*Cách trả lời:* "Dạ, giống như với câu hỏi về Log-transform, việc Scaling là bắt buộc với các thuật toán dựa trên khoảng cách (như KNN, SVM, Neural Network). Nhưng mô hình chủ lực của nhóm là XGBoost. Nó dùng Cây quyết định (Decision Trees) chỉ quan tâm đến việc 'lớn hơn hay nhỏ hơn một ngưỡng', nên hoàn toàn không cần thiết phải Scaling, giúp Pipeline chạy nhanh gọn hơn."

**Câu 11 (Về Validation - TimeSeriesSplit):**
*Hội đồng hỏi:* Trong bước GridSearchCV, tại sao nhóm dùng TimeSeriesSplit mà không dùng K-Fold Cross Validation thông thường?
*Cách trả lời:* "Dạ K-Fold thông thường sẽ xáo trộn (shuffle) dữ liệu. Trong bài toán Time-Series, nếu làm vậy mô hình sẽ lấy dữ liệu tương lai để dự đoán quá khứ. TimeSeriesSplit khắc phục điều này bằng cách tạo ra các nếp gấp (folds) trượt theo thời gian, đảm bảo mô hình luôn train trên quá khứ và test trên tương lai gần, không bao giờ bị rò rỉ ranh giới thời gian."

**Câu 12 (Về Success Metrics):**
*Hội đồng hỏi:* Nhóm dùng cả RMSE và MAE. Sự khác biệt giữa 2 chỉ số này là gì, và trong bài toán cháy rừng này cái nào quan trọng hơn?
*Cách trả lời:* "Dạ MAE (Mean Absolute Error) tính sai số tuyệt đối trung bình, mọi sai số đều bị phạt ngang nhau. Trong khi đó RMSE (Root Mean Square Error) bình phương sai số lên, nên nó 'phạt' rất nặng những dự báo lệch nhiều. Với cháy rừng, việc dự báo trượt một đỉnh AQI cực đoan là rất nguy hiểm. Do đó nhóm ưu tiên tối ưu RMSE hơn để buộc mô hình bám sát các đỉnh gai này."

**Câu 13 (Về XGBoost vs LightGBM):**
*Hội đồng hỏi:* XGBoost và LightGBM đều cho R2 xấp xỉ nhau (~0.87). Tại sao nhóm chọn XGBoost làm mô hình triển khai cuối cùng?
*Cách trả lời:* "Dạ, dù tốc độ train của LightGBM nhanh hơn, nhưng XGBoost (với thuật toán histogram-based) tỏ ra nhỉnh hơn một chút ở chỉ số RMSE (9.47 so với 9.48). XGBoost kiểm soát việc mọc lá sâu (tree pruning) chặt chẽ hơn, giúp nó bớt nhạy cảm với hiện tượng Overfitting khi đối mặt với dữ liệu nhiễu cao như bụi PM2.5."

**Câu 14 (Về Dashboard Architecture):**
*Hội đồng hỏi:* Tại sao nhóm lại chọn Streamlit để xây Dashboard thay vì Dash (Plotly) như nhiều dự án khác thường làm?
*Cách trả lời:* "Dạ, Dash rất mạnh về vẽ biểu đồ tương tác, nhưng lại khá cồng kềnh khi muốn nhúng một mô hình Machine Learning lớn chạy Real-time Inference. Streamlit được thiết kế Native cho Data Science, cho phép nhóm load file `.pkl` của XGBoost trực tiếp vào bộ nhớ đệm (cache), tốc độ dự báo trực tiếp mượt mà hơn và code Python backend rất tinh gọn."

**Câu 15 (Về Air Stagnation Alert):**
*Hội đồng hỏi:* Cảnh báo 'Air Stagnation' (Không khí ứ đọng) trên Dashboard hoạt động dựa trên logic nào? Có phải là bốc thuốc không?
*Cách trả lời:* "Dạ không phải bốc thuốc ạ. Cảnh báo này dựa trên nguyên lý: nếu mô hình dự báo AQI duy trì ở mức Cao/Nguy hiểm liên tục trong nhiều giờ mà tốc độ gió (Wind Speed) cực thấp và nhiệt độ cao, hệ thống sẽ trigger cảnh báo đỏ. Đây là ứng dụng thực tế để nhận diện việc khói bụi bị mắc kẹt lại trong thung lũng, gây ngạt cho người dân."

**Câu 16 (Về Chronological Boundary Leakage Guard):**
*Hội đồng hỏi:* Khoảng đệm 4 giờ (đối với Nowcasting) và 215 giờ (đối với Forecasting) trong việc chia data có tác dụng gì và con số 215h đến từ đâu?
*Cách trả lời:* "Dạ thưa, vì nhóm tạo ra các biến trễ (Lag) lùi về quá khứ (ví dụ Lag_3h là lấy data 3 giờ trước). Ở đúng ranh giới chia cắt giữa tập Train và Test, nếu không xóa trắng một khoảng đệm, biến Lag của tập Test sẽ chọc ngược vào tập Train. Con số 215 giờ ở kịch bản 24h được tính toán chính xác bằng tổng của khoảng trễ quá khứ xa nhất (191 giờ spatial & day-prior lag) cộng với 24 giờ đệm của horizon forecast. Việc xóa khoảng đệm này giúp cắt đứt hoàn toàn dây dưa rò rỉ, đảm bảo tập Test độc lập 100% ạ."

**Câu 17 (Về Business Value):**
*Hội đồng hỏi:* Đồ án này có giá trị ứng dụng thực tế gì hay chỉ là làm cho xong môn học?
*Cách trả lời:* "Dạ đồ án có tính ứng dụng rất cao. Bằng chứng là Dashboard của nhóm có thể deploy chạy Real-time. Nó có thể giúp chính quyền địa phương hoặc các bệnh viện tại California dự báo trước 24 giờ về rủi ro bùng phát hen suyễn do khói cháy rừng để chuẩn bị vật tư y tế. Còn với người dân, nó là một app cảnh báo để họ quyết định có nên ra ngoài hay bật máy lọc không khí."

**Câu 18 (Về Hạn chế):**
*Hội đồng hỏi:* Hãy trung thực, đâu là điểm yếu lớn nhất của mô hình các bạn hiện tại?
*Cách trả lời:* "Dạ điểm yếu lớn nhất là hiệu năng dự báo xa 24 giờ (Forecasting) sụt giảm so với dự báo gần 1-3 giờ (Nowcasting). Nguyên nhân gốc rễ là vì mô hình Tabular bị phụ thuộc quá nhiều vào bối cảnh thời tiết (VPD, Temp) mà thiếu đi con mắt nhìn từ trên cao. Nhóm chưa có dữ liệu Ảnh vệ tinh để thấy được quỹ đạo lan truyền của đám khói khổng lồ. Đây là giới hạn về dữ liệu chứ không hẳn là do thuật toán."

**Câu 19 (Về Giữ lại Outliers):**
*Hội đồng hỏi:* Hầu hết các bài toán ML đều khuyên xóa Outlier. Tại sao nhóm lại kiên quyết giữ lại các giá trị AQI cực đoan lên tới 300-400?
*Cách trả lời:* "Dạ vì Outlier chính là mục tiêu sống còn của đồ án này. Đồ án dự báo 'sự kiện cực đoan và cháy rừng', nếu nhóm xóa Outlier đi (nghĩa là xóa các đợt cháy) thì mô hình chỉ biết dự báo không khí trong lành hằng ngày. Làm Data Science không phải là mù quáng làm mượt dữ liệu, mà phải hiểu ý nghĩa vật lý đằng sau dữ liệu đó ạ."

**Câu 20 (Về Hướng phát triển):**
*Hội đồng hỏi:* Nếu có thêm 3 tháng để làm tiếp dự án này, nhóm sẽ cải tiến điều gì?
*Cách trả lời:* "Dạ nếu có thêm thời gian, nhóm sẽ làm 2 việc. Một là gọi API lấy thẳng dữ liệu Vệ tinh đo độ dày khói (AOD) của NASA/NOAA để bù đắp điểm yếu dự báo xa. Hai là triển khai tính năng tự động gửi SMS/Email cảnh báo khi phát hiện nguy cơ Air Stagnation trên Dashboard, biến nó thành một sản phẩm IoT hoàn chỉnh hơn ạ."

**Câu 21 (Về Ablation Study và vai trò của VPD):**
*Hội đồng hỏi:* Theo báo cáo, khi loại bỏ biến VPD thì R2 chỉ thay đổi rất nhỏ (khoảng 0.001 đến 0.002). Tại sao nhóm vẫn quyết định giữ biến này trong mô hình và gọi nó là "biến bối cảnh"?
*Cách trả lời:* "Dạ thưa, việc tính toán và thêm biến VPD là một quyết định dựa trên Domain Knowledge nhằm cung cấp cho mô hình thước đo về độ khô hạn của không khí - yếu tố cực kỳ quan trọng trong cháy rừng. Mặc dù kết quả Ablation study minh bạch cho thấy điểm số R2 trung bình toàn cục không tăng vọt, nhưng nhóm vẫn giữ lại vì nó giúp mô hình có thêm dữ kiện vật lý trong các phân cảnh cực đoan. Qua đó nhóm cũng rút ra bài học là không nói quá (oversimplify) về sức mạnh của một biến số, mà nhìn nhận nó đúng với vai trò là 'biến bối cảnh' (context feature) ạ."

**Câu 22 (Về Chỉ số Relative Prediction Accuracy trên Dashboard):**
*Hội đồng hỏi:* Chỉ số "Relative Prediction Accuracy = 100% - wMAPE" trên Dashboard được tính như thế nào? Tại sao mô hình Climatology có $R^2$ âm ($-0.0687$) nhưng Relative Accuracy vẫn đạt $67.70\%$? Cách tính này có mâu thuẫn hay đáng tin cậy không?
*Cách trả lời:* "Dạ thưa hội đồng, hai chỉ số này đánh giá hai khía cạnh hoàn toàn khác nhau nên không mâu thuẫn ạ:
1. **$R^2$ Score (Hệ số xác định):** Là thước đo học thuật chuẩn được nhóm báo cáo chính thức trong bài báo (Section 5), dùng để đánh giá khả năng mô hình giải thích sự biến thiên (variance) của AQI. $R^2 < 0$ ở Climatology cho thấy mô hình baseline này không bắt được các biến động cực đoan.
2. **Relative Accuracy ($100\% - \text{wMAPE}$):** Là chỉ số vận hành (operational metric) nhóm thiết kế thêm trên giao diện Web App cho người dùng phổ thông. Nhóm dùng $\text{wMAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i}$ (Weighted MAPE) thay vì MAPE thông thường để tránh hiện tượng bùng nổ phân số khi AQI tiệm cận 0. Con số $67.70\%$ chỉ phản ánh rằng tổng sai số tuyệt đối của Climatology chiếm $32.30\%$ quy mô AQI thực tế.
Tóm lại, trong bài báo nghiên cứu nhóm hoàn toàn tuân thủ chuẩn mực bằng bộ ba MAE, RMSE, $R^2$, còn Relative Accuracy là chỉ số trực quan giao diện giúp cán bộ quản lý dễ ước lượng tỷ lệ sai số theo phần trăm ạ."

