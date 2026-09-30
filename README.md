# Olympic AI - Tổng hợp Bài thi & Luyện tập

Kho lưu trữ chứa toàn bộ mã nguồn, tài liệu và kết quả dự đoán của 3 bài thi/tác vụ trong khuôn khổ Olympic AI:

1. **Luyện tập 1 (`practice-1/`)**: Dự đoán & Phân loại cấp bão (Storm Intensity Classification - CV).
2. **Tác vụ 1 (`tacvu1_absa/`)**: Khai phá ý kiến đa miền - Trích xuất bộ ba khía cạnh, danh mục, cảm xúc (MEMD-ABSA / ACSTE - NLP).
3. **Tác vụ 2 (`tacvu2_trashnet/`)**: Phân loại rác thải 6 lớp (TrashNet Classification - CV).
4. **Tác vụ 3 (`tacvu3_deepFake/`)**: Phát hiện mạo danh / Deepfake khuôn mặt theo cặp (Pairwise Classification - CV).
5. **Tác vụ 4 (`tacvu4_table_extraction/`)**: Trích xuất bảng từ ảnh tài liệu tiếng Việt sang Markdown (Table Structure Recognition + OCR).

---

## 📁 Cấu trúc thư mục

```text
Olympic-AI-Practice-1/
├── practice-1/                          # BÀI THI 1: Phân loại cấp bão (CV)
│   ├── data/                            # Ảnh vệ tinh và nhãn bão
│   ├── notebooks/                       # baseline_cv.ipynb
│   ├── outputs/                         # Log & lịch sử huấn luyện
│   ├── submissions/                     # public_cv.csv, private_cv.csv, QuangManhAI.zip
│   ├── docs/                            # Báo cáo kỹ thuật (.pdf)
│   └── README.md
│
├── tacvu1_absa/                         # BÀI THI 2 (Tác vụ 1): MEMD-ABSA (NLP)
│   ├── notebooks/
│   │   ├── generate_result_qwen.ipynb   # Pipeline suy luận Qwen3
│   │   └── generate_result_t5.ipynb     # Pipeline suy luận T5-Base
│   ├── scripts/
│   │   ├── train_qwen.py                # Huấn luyện SFT LoRA Qwen3
│   │   ├── train_boost_laptop_clothing.py # Huấn luyện boost miền Laptop & Clothing
│   │   ├── train_boost_qwen_v2.py       # Tích hợp mô hình & tối ưu dự đoán
│   │   ├── train_t5_base.py             # Huấn luyện T5-Base
│   │   ├── train_t5_acste.py            # Huấn luyện T5-ACSTE baseline
│   │   ├── evaluation_script.py         # Script tính Micro-F1 của BTC
│   │   └── merge_domain_submission.py   # Ghép kết quả các domain
│   ├── submissions/                     # submission.json, submission_*.zip (F1 ~56.54%)
│   └── README.md
│
├── tacvu2_trashnet/                     # BÀI THI 3 (Tác vụ 2): Phân loại rác TrashNet (CV)
│   ├── notebooks/
│   │   ├── baseline_TACVU2.ipynb        # Pipeline baseline EfficientNet
│   │   ├── finetune.ipynb               # Fine-tuning và threshold tuning
│   │   └── eval_competitive.ipynb       # Đánh giá so sánh mô hình
│   ├── scripts/
│   │   ├── train_full_30epochs.py       # Huấn luyện 30 epochs trên 100% dữ liệu
│   │   └── train_unleashed.py           # Huấn luyện tối ưu hiệu năng
│   ├── checkpoints/                     # best_model_eff_b2.pth và các checkpoints
│   ├── submissions/                     # submission.csv, submission.zip
│   └── README.md
│
├── tacvu3_deepFake/                     # BÀI THI 4 (Ca 1 - Tác vụ 1): Deepfake Face Detection (CV)
│   └── ca1/TACVU1/
│       ├── baseline_TACVU1_dense_net.ipynb # Pipeline DenseNet-121 (Private score 94.00%)
│       └── data/                        # train, public_test, private_test
│
├── tacvu4_table_extraction/             # BÀI THI 5 (Ca 1 - Tác vụ 2): Trích xuất bảng sang Markdown
│   ├── baseline_TACVU2.ipynb            # Notebook baseline độc lập
│   └── data/                            # training_set, public_test, private_test
│
├── data/                                # Dữ liệu dùng chung (Raw Data)
├── models/                              # Pretrained weights (qwen3-0.6B, t5-base, table_microsoft)
├── outputs/                             # Checkpoints sinh ra khi chạy huấn luyện
├── requirements.txt                     # Danh sách thư viện Python cần thiết
└── README.md                            # Tài liệu tổng quan (file này)
```

---

## ⚙️ Cài đặt môi trường

```bash
# Kích hoạt virtual environment
source .venv/bin/activate

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

---

## 🚀 Hướng dẫn thực thi từng bài

### 1. Luyện tập 1: Phân loại cấp bão
- Mở notebook: [`practice-1/notebooks/baseline_cv.ipynb`](practice-1/notebooks/baseline_cv.ipynb)
- Xem chi tiết tại [`practice-1/README.md`](practice-1/README.md).

### 2. Tác vụ 1: MEMD-ABSA (NLP)
- **Suy luận tạo kết quả `submission.json` với Qwen**:
  ```bash
  python tacvu1_absa/scripts/train_qwen.py \
    --predict_only \
    --checkpoint_path outputs/qwen_all/checkpoint-best \
    --domain all \
    --predict_split PublicTest \
    --output_dir tacvu1_absa/submissions
  ```
- **Huấn luyện boost các miền Laptop & Clothing**:
  ```bash
  python tacvu1_absa/scripts/train_boost_laptop_clothing.py --dataset_root data/TACVU1
  ```
- Xem chi tiết tại [`tacvu1_absa/README.md`](tacvu1_absa/README.md).

### 3. Tác vụ 2: Phân loại rác TrashNet (CV)
- **Huấn luyện EfficientNet-B2**:
  ```bash
  python tacvu2_trashnet/scripts/train_full_30epochs.py
  ```
- Xem chi tiết tại [`tacvu2_trashnet/README.md`](tacvu2_trashnet/README.md).
