# Luyện tập 1: Dự đoán & Phân loại cấp bão (Storm Intensity Classification)

## 1. Giới thiệu bài toán
- **Mục tiêu**: Phân loại cấp độ bão dựa trên ảnh vệ tinh (5 cấp độ bão).
- **Phương pháp**: Mô hình CNN ResNet tùy chỉnh (Custom ResNet / CNN), tối ưu Focal Loss để giải quyết mất cân bằng lớp giữa các cấp bão hiếm gặp.

## 2. Cấu trúc thư mục
```
practice-1/
├── data/
│   ├── csv/                 # Chứa public_cv.csv, private_cv.csv phân tách mẫu
│   ├── train/               # Ảnh huấn luyện và nhãn annotations.csv
│   ├── public_test/         # Tập ảnh public test
│   └── private_test/        # Tập ảnh private test
├── notebooks/
│   └── baseline_cv.ipynb    # Huấn luyện mô hình CNN phân loại bão
├── outputs/
│   └── log/                 # Lịch sử training (training_history.csv)
├── submissions/             # Các file nộp bài (public_cv.csv, private_cv.csv, QuangManhAI.zip)
└── docs/                    # Báo cáo kỹ thuật (.pdf)
```

## 3. Cách chạy
Mở và thực thi luồng huấn luyện trong notebook [baseline_cv.ipynb](file:///Users/quangmanh/Project/Olympic-AI-Practice-1/practice-1/notebooks/baseline_cv.ipynb).
