import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

FIELD_SEP = " | "
TRIPLE_JOIN = " ; "
EMPTY_TARGET = "<empty>"
TARGET_DOMAINS = ["Laptop", "Clothing"]


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

    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def prepare_training_sample(prompt_text: str, answer_text: str, tokenizer, max_length: int = 256):
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
        "attention_mask": [1] * len(input_ids),
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
    if not text:
        return []
    t = text.strip()
    if not t or t.lower() in ("<empty>", "none", "n/a", "empty"):
        return []

    triples = []
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
    model.eval()
    results = []
    print(f"Đang sinh kết quả chuyên sâu cho domain [{domain}] ({len(items)} câu)...")
    for item in tqdm(items):
        raw_words = item["raw_words"]
        prompt = build_prompt(raw_words, domain)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_len:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        triples = parse_generated_triples(response_text)

        results.append({
            "raw_words": raw_words,
            "triples": triples,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Boost Laptop & Clothing with Train+Dev data")
    parser.add_argument("--base_model", type=str, default="models/qwen3-0.6B")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/qwen_all/checkpoint-best")
    parser.add_argument("--dataset_root", type=str, default="data/TACVU1")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_boost_laptop_clothing")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device.upper()}")

    # 1. Nạp Model & Tokenizer kế thừa từ checkpoint 55.04
    print(f"Nạp Tokenizer từ: {args.checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Nạp Base model từ {args.base_model} và tích hợp LoRA weights từ {args.checkpoint_dir}...")
    model_dtype = torch.float32 if device == "cpu" else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=True, dtype=model_dtype)
    peft_model = PeftModel.from_pretrained(base_model, args.checkpoint_dir, local_files_only=True)
    model = peft_model.merge_and_unload()
    print(">> Đã hợp nhất mô hình nền tảng thành công!")

    # 2. Chuẩn bị dữ liệu Train + Dev cho Laptop và Clothing
    root = Path(args.dataset_root)
    with open(root / "train/Train.json") as f:
        train_raw = json.load(f)
    with open(root / "train/Dev.json") as f:
        dev_raw = json.load(f)

    train_samples = []
    for d in TARGET_DOMAINS:
        # Gộp cả Train và Dev
        all_items = train_raw[d] + dev_raw[d]
        print(f"Domain [{d}]: Gom {len(train_raw[d])} câu (Train) + {len(dev_raw[d])} câu (Dev) = {len(all_items)} câu")
        for item in all_items:
            target_str = target_text_from_quadruples(item["quadruples"])
            prompt_str = build_prompt(item["raw_words"], d)
            sample = prepare_training_sample(prompt_str, target_str, tokenizer)
            train_samples.append(sample)

    print(f">> Tổng số mẫu huấn luyện chuyên biệt: {len(train_samples)}")

    # Gắn LoRA mới để tinh chỉnh tiếp
    model.config.use_cache = False
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
    data_collator = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        seed=42,
        fp16=False,
        bf16=False,
        gradient_checkpointing=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("\n=== BẮT ĐẦU FINE-TUNE CHUYÊN BIỆT CHO LAPTOP & CLOTHING ===")
    trainer.train()

    best_dir = os.path.join(args.output_dir, "checkpoint-best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    model = model.merge_and_unload()
    model.to(device)

    # 3. Chạy suy luận PrivateTest cho Laptop và Clothing
    priv_file = root / "private_test/PrivateTest.json"
    with open(priv_file) as f:
        priv_data = json.load(f)

    boosted_predictions = {}
    for d in TARGET_DOMAINS:
        items = priv_data[d]
        pred_items = generate_predictions_for_domain(model, tokenizer, items, d, device)
        boosted_predictions[d] = pred_items

    # 4. Hợp nhất vào submission.json hiện có (giữ Hotel, Restaurant, Books đỉnh cao)
    current_sub_path = "submission_tacvu1/submission.json"
    with open(current_sub_path) as f:
        current_submission = json.load(f)

    # Thay thế Laptop và Clothing bằng kết quả mới
    for d in TARGET_DOMAINS:
        current_submission[d] = boosted_predictions[d]
        print(f">> Đã cập nhật kết quả mới cho domain [{d}] ({len(boosted_predictions[d])} câu)")

    # Lưu lại submission.json
    with open(current_sub_path, "w", encoding="utf-8") as f:
        json.dump(current_submission, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu submission.json mới tại: {current_sub_path}")

    # Đóng gói submission.zip
    os.system("cd submission_tacvu1 && zip -FS -r ../submission.zip submission.json generate_result.ipynb train_qwen.py evaluation_script.py README.md && cd ..")
    print("\n🎉 ĐÃ CẬP NHẬT THÀNH CÔNG submission.zip!")


if __name__ == "__main__":
    main()
