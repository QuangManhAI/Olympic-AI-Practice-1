import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
FIELD_SEP = " | "
TRIPLE_JOIN = " ; "
EMPTY_TARGET = "<empty>"

def quadruple_to_triple_parts(q: dict) -> tuple[str, str, str]: 
    terms = q['aspect'].get("term") or []
    if not terms or terms == ["NULL"]:
        aspect = "NULL"
    else:
        aspect = " ".join(str(t).strip() for t in terms)

    category = (q.get("category") or "").strip()
    sentiment = (q.get("sentiment") or "").strip().upper()
    return aspect, category, sentiment

def target_text_from_quadruples(quadruples: list) -> str: 
    parts = []
    for q in quadruples:
        a, c, s = quadruple_to_triple_parts(q)
        parts.append(f"{a}{FIELD_SEP}{c}{FIELD_SEP}{s}")
    return TRIPLE_JOIN.join(parts) if parts else EMPTY_TARGET

def build_prompt(raw_words: str, domain: str) -> str:
    system_msg = (
            "You are an expert in Aspect-Based Sentiment Analysis (ABSA). "
            "Extract all aspect-category-sentiment triples from the review sentence in the specified domain. "
            "Output format: aspect | category | sentiment. "
            "Separate multiple triples with ' ; '. If no triple, output <empty>. "
            "Output ONLY the triples without explanation or thinking tokens."
        )
    user_msg = f"Domain: {domain}\nReview: {raw_words}"

    prompt = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt

def prepare_training_sample(prompt_text: str, 
                            answer_text: str, 
                            tokenizer, 
                            max_length: int = 512):

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    answer_text_full = f"{answer_text}<|im_end|>\n"
    answer_ids = tokenizer.encode(answer_text_full, add_special_tokens=False)

    input_ids = prompt_ids + answer_ids

    labels = [-100] * len(prompt_ids) + answer_ids

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids)
    }

class ACSTEDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index]

class DataCollatorForCausalLM:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            batch_input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(item["attention_mask"] + [0] * pad_len)
            batch_labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

def parse_generated_triples(text: str) -> list[dict]:
    """
    Chuyển text do Qwen sinh ra thành list các dict triples.
    """
    if not text:
        return []
        
    t = text.strip()
    # Nếu rỗng hoặc sinh ra token trống -> trả về mảng rỗng []
    if not t or t.lower() in ("<empty>", "none", "n/a", "empty"):
        return []

    triples = []
    # Tách từng triple ngăn cách bởi dấu ';'
    for chunk in re.split(r"\s*;\s*", t):
        chunk = chunk.strip()
        if not chunk:
            continue
            
        parts = chunk.split(FIELD_SEP, 2)
        if len(parts) != 3:
            continue

        aspect, category, sentiment = (p.strip() for p in parts)
        aspect_clean = "NULL" if aspect.lower() == "null" or not aspect else aspect
        sentiment_clean = sentiment.upper() if sentiment.upper() in ("POS", "NEG", "NEU") else "POS"

        triples.append({
            "aspect": aspect_clean,
            "category": category,
            "sentiment": sentiment_clean,
        })
        
    return triples

def generate_predictions_for_domain(model, tokenizer, items: list[dict], domain: str, device: str) -> list[dict]:
    """
    Chạy inference trên toàn bộ các câu của 1 domain và trả về format nộp bài.
    """
    model.eval()
    results = []

    print(f"Đang sinh kết quả cho domain [{domain}] ({len(items)} câu)...")
    for item in tqdm(items):
        raw_words = item["raw_words"]
        
        # 1. Tạo prompt ChatML cho câu này
        prompt = build_prompt(raw_words, domain)
        
        # 2. Tokenize prompt và chuyển lên thiết bị (GPU/MPS/CPU)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]
        
        # 3. Model sinh tiếp sau prompt (không dùng sampling để kết quả ổn định nhất)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
        # 4. CHÚ Ý QUAN TRỌNG: Cắt bỏ phần prompt, chỉ decode phần token mới sinh ra
        generated_tokens = outputs[0][input_len:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # 5. Parse text thành danh sách triples
        triples = parse_generated_triples(response_text)
        
        results.append({
            "raw_words": raw_words,
            "triples": triples
        })
        
    return results


ALL_DOMAINS = ["Restaurant", "Laptop", "Hotel", "Books", "Clothing"]


def load_merged_split(dataset_root: str, split: str) -> dict[str, list]:
    """
    Đọc dữ liệu theo cấu trúc của ban tổ chức.
    """
    root = Path(dataset_root)
    if split in ("Train", "Dev"):
        path = root / "train" / f"{split}.json"
    elif split == "PublicTest":
        path = root / "public_test" / f"{split}.json"
    elif split == "PrivateTest":
        path = root / "private_test" / f"{split}.json"
        if not path.exists():
            path = root / f"{split}.json"
    else:
        raise ValueError(f"Split không hợp lệ: {split}")

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return data


def train_model(
    model,
    tokenizer,
    train_samples: list[dict],
    eval_samples: list[dict],
    output_dir: str = "outputs/qwen3-0.6B",
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    use_lora: bool = True,
    seed: int = 42,
):
    """
    Huấn luyện mô hình Qwen3 bằng Hugging Face Trainer (hỗ trợ LoRA).
    """
    print(f"\n=== BẮT ĐẦU HUẤN LUYỆN (SFT QWEN3-0.6B) ===")
    print(f"Số mẫu Train: {len(train_samples)} | Số mẫu Eval (Dev): {len(eval_samples)}")
    print(f"Epochs: {num_train_epochs} | Batch size hiệu dụng: {per_device_train_batch_size * gradient_accumulation_steps}")

    # Vô hiệu hóa cache trong lúc training
    model.config.use_cache = False

    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        print(">> Đang kích hoạt LoRA (PEFT)...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_dataset = ACSTEDataset(train_samples)
    eval_dataset = ACSTEDataset(eval_samples) if eval_samples else None
    data_collator = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate if use_lora else 2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        greater_is_better=False,
        seed=seed,
        fp16=False,
        bf16=False,
        gradient_checkpointing=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    best_checkpoint_dir = os.path.join(output_dir, "checkpoint-best")
    trainer.save_model(best_checkpoint_dir)
    tokenizer.save_pretrained(best_checkpoint_dir)
    print(f"\n Đã lưu checkpoint tốt nhất tại: {best_checkpoint_dir}")

    # Nếu dùng LoRA, gộp adapter vào model gốc để suy luận nhanh nhất
    if use_lora:
        print(">> Đang hợp nhất LoRA adapter vào mô hình gốc để suy luận...")
        model = model.merge_and_unload()

    return model


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện và Suy luận Qwen3-0.6B cho tác vụ ACSTE")
    parser.add_argument("--model_path", type=str, default="models/qwen3-0.6B", help="Đường dẫn thư mục trọng số Qwen3")
    parser.add_argument("--dataset", type=str, default="data/TACVU1", help="Thư mục gốc chứa data/TACVU1")
    parser.add_argument("--domain", type=str, default="all", help="'all' hoặc tên 1 domain (Restaurant, Laptop, ...)")
    parser.add_argument("--train_all_domains", action="store_true", help="Gộp chung cả 5 domain để train")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen3-0.6B", help="Thư mục lưu checkpoint và kết quả")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Số epochs huấn luyện")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device (khuyên dùng 2 trên Mac)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Số bước tích lũy gradient (khuyên dùng 8)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Tốc độ học (2e-4 cho LoRA, 2e-5 cho FFT)")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Sử dụng LoRA (mặc định: True)")
    parser.add_argument("--no_lora", dest="use_lora", action="store_false", help="Tắt LoRA, chuyển sang Full Fine-tune")
    parser.add_argument("--predict_split", type=str, default="Dev", choices=["Dev", "PublicTest", "PrivateTest"])
    parser.add_argument("--predict_only", action="store_true", help="Chỉ chạy suy luận, không train")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Đường dẫn checkpoint khi chạy --predict_only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Thiết lập seed
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device.upper()}")

    # Xác định các domain cần xử lý
    domains_to_process = ALL_DOMAINS if (args.domain == "all" or args.train_all_domains) else [args.domain]

    # Nạp Tokenizer
    model_source = args.checkpoint_path if args.predict_only and args.checkpoint_path else args.model_path
    print(f"Nạp Tokenizer từ: {model_source}")
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1. QUÁ TRÌNH HUẤN LUYỆN
    if not args.predict_only:
        print(f"Nạp mô hình Qwen3 để huấn luyện từ: {args.model_path}")
        # Dùng float32 hoặc bfloat16 tùy thiết bị
        model_dtype = torch.float32 if device == "cpu" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            local_files_only=True,
            dtype=model_dtype,
        )

        train_raw = load_merged_split(args.dataset, "Train")
        dev_raw = load_merged_split(args.dataset, "Dev")

        train_samples = []
        eval_samples = []

        for d in domains_to_process:
            for item in train_raw[d]:
                target_str = target_text_from_quadruples(item["quadruples"])
                prompt_str = build_prompt(item["raw_words"], d)
                sample = prepare_training_sample(prompt_str, target_str, tokenizer)
                train_samples.append(sample)

            for item in dev_raw[d]:
                target_str = target_text_from_quadruples(item["quadruples"])
                prompt_str = build_prompt(item["raw_words"], d)
                sample = prepare_training_sample(prompt_str, target_str, tokenizer)
                eval_samples.append(sample)

        # Chạy training
        model = train_model(
            model=model,
            tokenizer=tokenizer,
            train_samples=train_samples,
            eval_samples=eval_samples,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            use_lora=args.use_lora,
            seed=args.seed,
        )
    else:
        print(f"Chế độ PREDICT-ONLY: Nạp mô hình từ {model_source}")
        model_dtype = torch.float32 if device == "cpu" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            local_files_only=True,
            dtype=model_dtype,
        )

    # Đưa model lên device để suy luận
    model.to(device)

    # 2. QUÁ TRÌNH SUY LUẬN (INFERENCE)
    print(f"\n=== BẮT ĐẦU DỰ ĐOÁN TRÊN TẬP [{args.predict_split}] ===")
    test_raw = load_merged_split(args.dataset, args.predict_split)
    os.makedirs(args.output_dir, exist_ok=True)

    all_predictions = {}
    for d in domains_to_process:
        items = test_raw[d]
        pred_items = generate_predictions_for_domain(model, tokenizer, items, d, device)
        all_predictions[d] = pred_items

    # Xuất file nộp bài
    if args.domain == "all" or args.train_all_domains:
        submission_path = os.path.join(args.output_dir, f"predictions_{args.predict_split}.json")
        with open(submission_path, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, indent=2, ensure_ascii=False)
        print(f"\n Đã ghi kết quả của cả 5 domain ra file: {submission_path}")
    else:
        submission_path = os.path.join(args.output_dir, f"predictions_{args.predict_split}_{args.domain}.json")
        with open(submission_path, "w", encoding="utf-8") as f:
            json.dump(all_predictions[args.domain], f, indent=2, ensure_ascii=False)
        print(f"\n Đã ghi kết quả của domain [{args.domain}] ra file: {submission_path}")

    # Nếu dự đoán trên tập Dev, in luôn câu lệnh đánh giá F1 score
    if args.predict_split == "Dev":
        print("\nĐể xem điểm Micro-F1 trên tập Dev, hãy chạy lệnh:")
        gold_path = os.path.join(args.dataset, "train", "Dev.json")
        domain_flag = "all" if (args.domain == "all" or args.train_all_domains) else args.domain
        print(f"python data/TACVU1/baseline_TACVU1/evaluation_script.py --submission {submission_path} --domain {domain_flag} --gold_merged {gold_path}")


if __name__ == "__main__":
    main()


