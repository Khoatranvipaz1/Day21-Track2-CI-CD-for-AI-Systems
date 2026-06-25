# Báo Cáo Lab MLOps - Day 21: CI/CD for AI Systems

**Họ tên:** Trần Văn Khoa  
**MSSV:** 2A202600827  
**Course:** AIInAction - VinUni

---

## 1. Bộ Siêu Tham Số Đã Chọn

| Tham số | Giá trị |
|---|---|
| model_type | random_forest |
| n_estimators | 300 |
| max_depth | 20 |
| min_samples_split | 2 |

**Kết quả:** Accuracy = 0.756 | F1 = 0.755

**Lý do chọn:** Sau khi thử nghiệm 4 bộ tham số khác nhau trên MLflow:
- random_forest (n=100, depth=5): accuracy 0.564
- random_forest (n=200, depth=10): accuracy 0.644
- gradient_boosting (n=200, depth=5): accuracy 0.656
- random_forest (n=300, depth=20) + data phase2: accuracy **0.756** ✓

Bộ tham số cuối cùng kết hợp với toàn bộ dữ liệu (5996 mẫu = phase1 + phase2) cho kết quả tốt nhất và vượt ngưỡng eval gate 0.70.

---

## 2. Khó Khăn Gặp Phải và Cách Giải Quyết

**Vấn đề 1: Python 3.14 không tương thích với scikit-learn 1.4.2**  
→ Giải quyết: Dùng Python 3.10 (có sẵn trên máy) để tạo virtual environment.

**Vấn đề 2: AWS credentials bị lỗi khi parse trong GitHub Actions**  
→ Giải quyết: Thay vì parse JSON inline trong shell, ghi credentials ra file tạm `/tmp/creds.json` rồi dùng Python đọc và export env vars.

**Vấn đề 3: Accuracy không đạt ngưỡng 0.70 với data phase1 (2998 mẫu)**  
→ Giải quyết: Kết hợp data phase1 + phase2 (tổng 5996 mẫu), accuracy tăng lên 0.756.

**Vấn đề 4: GitHub Actions không trigger do workflow chỉ lắng nghe branch `main`**  
→ Giải quyết: Thêm `master` vào danh sách branches trong `mlops.yml`.

---

## 3. Kiến Trúc Hệ Thống

- **Cloud Provider:** AWS (S3 + EC2)
- **VM:** t3.micro, Ubuntu 26.04, IP: 54.167.39.70
- **Storage:** S3 bucket `mlops-lab-21-2026`
- **API:** FastAPI chạy trên cổng 8000 qua systemd service
- **CI/CD:** GitHub Actions với 4 jobs: Unit Test → Train → Eval → Deploy
