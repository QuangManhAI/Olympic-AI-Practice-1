# OLYMPIC AI 2026 - TÁC VỤ 1: TRÍCH XUẤT BỘ BA KHÍA CẠNH (ACSTE)

## 1. Cấu trúc thư mục nộp bài (submission.zip)

```text
submission.zip
├── submission.json          # File kết quả dự đoán trên tập PublicTest (hoặc PrivateTest)
├── generate_result.ipynb    # Notebook tái lập toàn bộ quy trình suy luận từ đầu đến cuối
├── train_qwen.py            # Script huấn luyện (SFT LoRA) và suy luận tự động
├── evaluation_script.py     # Script đánh giá Micro-F1 chính thức của BTC
└── README.md                # Tài liệu mô tả giải pháp và hướng dẫn chạy
```

---

## 2. Thông tin mô hình & Giải pháp

* **Kiến trúc mô hình:** `Qwen3-0.6B` (Causal Language Model, 28 layers, hidden_size 1024).
* **Phương pháp huấn luyện:** Supervised Fine-Tuning (SFT) kết hợp **LoRA (Low-Rank Adaptation)** trên toàn bộ 7 linear projection layers (`q, k, v, o, gate, up, down`).
* **Kỹ thuật tối ưu:**
  * **Domain Conditioning:** Đưa thông tin domain trực tiếp vào ChatML prompt để mô hình tập trung không gian nhãn chính xác.
  * **Loss Masking:** Chỉ tính Cross-Entropy loss trên phần nhãn trả lời `aspect | category | sentiment`, triệt tiêu phần suy luận thừa (`<think>`).
  * **Gradient Accumulation:** Tích lũy gradient (effective batch size = 16) tối ưu hóa hội tụ.
* **Kết quả thực nghiệm trên tập Dev (Micro-F1):**
  * Restaurant: **61.39%** (tăng +6.03% so với baseline 55.36%)
  * Laptop: **46.18%** (tăng +5.90% so với baseline 40.28%)
  * Hotel: **72.67%** (tăng +11.40% so với baseline 61.27%)
  * Books: **52.84%** (tăng +10.53% so với baseline 42.31%)
  * Clothing: **40.73%** (tương đương baseline 40.90%)
  * 👉 **Trung bình Micro-F1 trên cả 5 domains:** **54.76%** (vượt trội hơn baseline 48.02% gần +7% F1).

---

## 3. Hướng dẫn chạy lại luồng

### A. Chạy bằng Jupyter Notebook:
Mở và chạy toàn bộ các cells trong `generate_result.ipynb`.

### B. Chạy bằng dòng lệnh (CLI):
```bash
# Suy luận trực tiếp tạo submission.json:
python train_qwen.py \
  --predict_only \
  --checkpoint_path outputs/qwen_all/checkpoint-best \
  --domain all \
  --predict_split PublicTest \
  --output_dir .
```
