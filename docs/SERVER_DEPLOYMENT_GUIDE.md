# Windows Server Deployment Guide (Python + Tailscale)

Hướng dẫn từng bước kéo mã nguồn mới nhất, nạp dữ liệu lịch sử và khởi chạy ứng dụng Streamlit trên Windows Server 2019 sử dụng Python Native và Tailscale Funnel.

---

## 📋 Thư mục làm việc mặc định trên Server
`C:\AQICali`

---

## 🚀 Các bước thực thi trên Windows Server

### 1. Cập nhật mã nguồn & xử lý thay đổi cục bộ
Mở **PowerShell** tại `C:\AQICali`:
```powershell
cd C:\AQICali
git stash push -m "server-local-changes"
git pull origin main
```

---

### 2. Cập nhật các thư viện phụ thuộc (Dependencies)
```powershell
cd C:\AQICali
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

### 3. Nạp dữ liệu dự báo lịch sử cho Wildfire Bands (SQLite Backfill)
```powershell
cd C:\AQICali
.\.venv\Scripts\python.exe scr\backfill_historical_model_predictions.py
```

---

### 4. Khởi chạy Streamlit Dashboard
```powershell
cd C:\AQICali
.\.venv\Scripts\python.exe -m streamlit run app\ui.py `
  --server.address 0.0.0.0 `
  --server.port 8501 `
  --server.headless true
```
> **Lưu ý:** Dùng `--server.address 0.0.0.0` để ứng dụng tiếp nhận đồng thời cả kết nối qua **Tailscale Funnel (HTTPS)** lẫn kết nối **IP nội bộ Tailscale (`100.x.y.z:8501`)**.

---

### 5. Kích hoạt Tailscale Funnel (Chia sẻ liên kết HTTPS Public)
Mở một cửa sổ **PowerShell Administrator mới**:
```powershell
& "C:\Program Files\Tailscale\tailscale.exe" funnel 8501
```

Sau khi chạy lệnh trên, ứng dụng sẽ cung cấp đường link HTTPS public an toàn (ví dụ: `https://<server-name>.tailnet.ts.net`) để thuyết trình hoặc chia sẻ cho các bên liên quan.
