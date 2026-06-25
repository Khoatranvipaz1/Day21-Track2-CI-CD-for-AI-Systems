# Báo Cáo Lab MLOps - Day 21: CI/CD for AI Systems

**Họ tên:** Trần Văn Khoa  
**MSSV:** 2A202600827  
**Course:** AIInAction - VinUni  
**Ngày nộp:** 25/06/2026

---

## Bước 1 — Thí Nghiệm MLflow Cục Bộ

### Bộ siêu tham số đã thử nghiệm

| Run | model_type | n_estimators | max_depth | Accuracy | F1 |
|---|---|---|---|---|---|
| 1 | random_forest | 100 | 5 | 0.564 | 0.558 |
| 2 | random_forest | 200 | 10 | 0.644 | 0.638 |
| 3 | gradient_boosting | 200 | 5 | 0.656 | 0.651 |
| 4 | random_forest | 300 | 20 | **0.756** | **0.755** |

### Bộ tham số tốt nhất (params.yaml)

```yaml
model_type: random_forest
n_estimators: 300
max_depth: 20
min_samples_split: 2
```

**Kết quả cuối:** Accuracy = **0.756** | F1 = **0.755**

**Lý do chọn:** Random Forest với 300 cây và max_depth=20 cho độ chính xác cao nhất trên bài toán phân loại chất lượng rượu vang 3 lớp (thấp/trung bình/cao). Kết hợp với dữ liệu phase1+phase2 (5996 mẫu) để vượt ngưỡng eval gate 0.70.

---

## Bước 2 — Pipeline CI/CD Tự Động

### Kiến trúc hệ thống

- **Cloud Provider:** AWS (S3 + EC2)
- **VM:** t3.micro, Ubuntu 26.04, IP: `54.167.39.70`
- **Storage:** S3 bucket `mlops-lab-21-2026`
- **API:** FastAPI trên cổng 8000, quản lý bởi systemd (`mlops-serve`)
- **CI/CD:** GitHub Actions với 4 jobs tuần tự

### Luồng pipeline (mlops.yml)

```
Push code/data → Unit Test → Train → Eval (gate ≥ 0.70) → Deploy
```

| Job | Thao tác | Thời gian |
|---|---|---|
| Unit Test | pytest tests/ | ~1m 11s |
| Train | dvc pull → train.py → upload S3 | ~1m 3s |
| Eval | rollback check + accuracy ≥ 0.70 | ~11s |
| Deploy | SSH restart mlops-serve + health check | ~18s |

### Kết quả API

```bash
curl http://54.167.39.70:8000/health
# {"status":"ok"}

curl -X POST http://54.167.39.70:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [7.4,0.70,0.00,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,0]}'
# {"prediction":0,"label":"thap"}
```

---

## Bước 3 — Huấn Luyện Liên Tục Với Dữ Liệu Mới

### Quy trình

1. Kỹ sư dữ liệu chạy `add_new_data.py` → data tăng từ 2998 → 5996 mẫu
2. `dvc add data/train_phase1.csv` → cập nhật file `.dvc`
3. `dvc push` → đẩy data mới lên S3
4. `git commit data/train_phase1.csv.dvc -m "data: add new training data phase2 (2998 -> 5996 samples)"`
5. `git push` → **pipeline tự động trigger**, không cần tác động thủ công

**Commit kích hoạt Bước 3:** `e4e4b65`  
**Commit message:** `data: add new training data phase2 (2998 -> 5996 samples)`  
**Kết quả:** 4 jobs xanh, accuracy 0.756 ≥ 0.70, model mới deploy lên VM

---

## Các Tính Năng Bonus

| # | Tính năng | Trạng thái | Mô tả |
|---|---|---|---|
| 1 | MLflow Remote Tracking | ✅ | `MLFLOW_TRACKING_URI` cấu hình qua GitHub Secrets |
| 2 | Multi-model Support | ✅ | `params.yaml` hỗ trợ random_forest / gradient_boosting / logistic_regression |
| 3 | CI Artifacts | ✅ | `outputs/metrics.json` + `outputs/report.txt` upload lên GitHub Actions |
| 4 | Rollback Protection | ✅ | So sánh accuracy mới vs deployed; hủy deploy nếu giảm |
| 5 | Label Distribution Check | ✅ | Cảnh báo nếu class nào < 10% tổng mẫu |

---

## Khó Khăn và Cách Giải Quyết

| Vấn đề | Giải pháp |
|---|---|
| Python 3.14 không tương thích scikit-learn 1.4.2 | Tạo venv bằng Python 3.10 |
| AWS credentials bị lỗi parse trong shell (ký tự đặc biệt) | Ghi secret ra `/tmp/creds.json`, dùng Python đọc và export env vars |
| GitHub Actions không trigger (branch `main` vs `master`) | Thêm `master` vào `branches:` trong workflow |
| Accuracy < 0.70 với 2998 mẫu (max ~0.678) | Kết hợp phase1 + phase2 → 5996 mẫu → accuracy 0.756 |
| Ubuntu 26.04 chặn pip3 system-wide | Tạo virtualenv `~/mlops-venv` trên VM |

---

## Screenshots

| File | Nội dung |
|---|---|
| `screenshots/buoc1-mlflow-ui.png` | MLflow UI — 4 lần thử nghiệm |
| `screenshots/buoc2-github-actions-green.png` | GitHub Actions — 4 jobs xanh |
| `screenshots/buoc2-curl-health-predict.png` | Kết quả curl /health và /predict |
| `screenshots/buoc2-s3-console.png` | S3 bucket với 3 folders: dvc/, models/, outputs/ |
| `screenshots/buoc3-auto-trigger.png` | Pipeline tự trigger từ data commit |
