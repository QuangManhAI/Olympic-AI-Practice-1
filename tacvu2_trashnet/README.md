# Tác vụ 2: Phân loại rác thải TrashNet (TrashNet Image Classification)

## 1. Giới thiệu bài toán
- **Mục tiêu**: Phân loại hình ảnh rác thành 6 nhóm danh mục (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`).
- **Metric đánh giá**: Macro F1-Score.
- **Kiến trúc mô hình**: `EfficientNet-B2` kết hợp `Focal Loss` giải quyết mất cân bằng lớp và kỹ thuật Data Augmentation (Flip, Rotation, Resize).

## 2. Cấu trúc thư mục
```
tacvu2_trashnet/
├── notebooks/
│   ├── baseline_TACVU2.ipynb      # Pipeline baseline phân loại rác với EfficientNet
│   ├── finetune.ipynb             # Thử nghiệm fine-tuning và tuning threshold
│   └── eval_competitive.ipynb     # Đánh giá so sánh và phân tích dự đoán
├── scripts/
│   ├── train_full_30epochs.py     # Huấn luyện mô hình 30 epochs trên toàn bộ dữ liệu train
│   └── train_unleashed.py         # Huấn luyện mở rộng tối ưu hyperparameters
├── checkpoints/
│   ├── best_model_eff_b2.pth      # Trọng số tốt nhất của EfficientNet-B2
│   ├── checkpoints/               # Checkpoints từng epoch
│   └── val_probs.npy, val_targets.npy
└── submissions/
    ├── submission.csv             # File kết quả dự đoán trên tập test
    ├── submission.zip             # Gói nộp bài định dạng quy định của BTC
    └── QuangManhAI_tacvu2.zip     # Gói nộp dự phòng
```

## 3. Cách chạy
- Dữ liệu đầu vào: nằm ở `data/TACVU2/train`, `data/TACVU2/public_test`, `data/TACVU2/private_test`.
- Huấn luyện qua script:
  ```bash
  python tacvu2_trashnet/scripts/train_full_30epochs.py
  ```
- Hoặc mở notebook trong [notebooks/](file:///Users/quangmanh/Project/Olympic-AI-Practice-1/tacvu2_trashnet/notebooks) để chạy trực tiếp từng bước.
