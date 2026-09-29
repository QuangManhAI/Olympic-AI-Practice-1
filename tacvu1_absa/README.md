# Tác vụ 1: Khai phá ý kiến đa miền (MEMD-ABSA / ACSTE)

## 1. Giới thiệu bài toán
- **Mục tiêu**: Trích xuất bộ ba khía cạnh - danh mục - cảm xúc (Aspect-Category-Sentiment Triplet Extraction) qua 5 miền dữ liệu: Restaurant, Laptop, Hotel, Books, Clothing.
- **Metric đánh giá**: Micro-F1 (Exact Match Triples).
- **Mô hình tiếp cận**:
  - `Qwen3-0.6B` (Causal LM LoRA SFT) kết hợp Domain Conditioning & Loss Masking.
  - `T5-Base` (Seq2Seq LM LoRA SFT / ACSTE format).

## 2. Cấu trúc thư mục
```
tacvu1_absa/
├── notebooks/
│   ├── generate_result_qwen.ipynb   # Luồng suy luận Qwen3 sinh submission.json
│   └── generate_result_t5.ipynb     # Luồng suy luận T5-Base sinh submission.json
├── scripts/
│   ├── train_qwen.py                # Script train và eval mô hình Qwen3
│   ├── train_boost_laptop_clothing.py # Huấn luyện boost riêng miền Laptop & Clothing
│   ├── train_boost_qwen_v2.py       # Ghép checkpoint và đa miền
│   ├── train_t5_base.py             # Script train T5-Base
│   ├── train_t5_acste.py            # Script train T5-ACSTE baseline
│   ├── evaluation_script.py         # Script tính Micro-F1 theo chuẩn BTC
│   └── merge_domain_submission.py   # Ghép kết quả dự đoán từng domain
├── submissions/
│   ├── submission.json              # File kết quả JSON nộp bài
│   ├── submission_55.04.json        # Checkpoint điểm F1 55.04%
│   ├── submission_qwen_56.54.json   # Checkpoint điểm F1 56.54%
│   └── submission_*.zip             # Gói nén nộp bài
└── README.md
```

## 3. Hướng dẫn chạy
- **Dữ liệu**: Nằm tại `data/TACVU1/`.
- **Mô hình Pretrained**: Nằm tại `models/qwen3-0.6B` và `models/t5-base`.
- **Outputs / Checkpoints**: Nằm tại `outputs/qwen_all`, `outputs/t5_base`, ...

### Chạy suy luận tạo submission với Qwen:
```bash
python tacvu1_absa/scripts/train_qwen.py \
  --predict_only \
  --checkpoint_path outputs/qwen_all/checkpoint-best \
  --domain all \
  --predict_split PublicTest \
  --output_dir tacvu1_absa/submissions
```

### Chạy huấn luyện boost miền Laptop & Clothing:
```bash
python tacvu1_absa/scripts/train_boost_laptop_clothing.py --dataset_root data/TACVU1
```
