#!/usr/bin/env python3
"""
train_boost_qwen_v2.py: Kế thừa và huấn luyện tiếp từ Checkpoint Qwen 56.54 điểm
(Đã tích hợp Stage 1 [qwen_all] + Stage 2 [qwen_boost_laptop_clothing]).

Các điểm nổi bật:
  1. Hợp nhất tuần tự 2 checkpoint thành mô hình nền tảng mạnh mẽ nhất (56.54 điểm).
  2. Gắn LoRA Adapter mới (r=16, alpha=32) để tiếp tục tinh chỉnh thêm 3-5 epochs.
  3. Batch size 4, gradient_accumulation_steps 4 (effective bs = 16) tối ưu tốc độ x2 trên Apple Silicon (MPS).
  4. Hỗ trợ huấn luyện toàn bộ 5 domain hoặc chuyên sâu cho [Restaurant, Hotel, Books] (Train + Dev).
  5. Tự động dự đoán PrivateTest, áp dụng bộ lọc hậu xử lý (Post-Processing) nắn chỉnh span và category.
  6. Đóng gói đầy đủ submission.zip (kèm generate_result.ipynb, main.ipynb) chuẩn quy chế chấm của BTC.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

ALL_DOMAINS = ["Restaurant", "Laptop", "Hotel", "Books", "Clothing"]
FIELD_SEP = " | "
TRIPLE_JOIN = " ; "
EMPTY_TARGET = "<empty>"
VALID_SENTIMENTS = {"POS", "NEG", "NEU"}


def quadruple_to_triple_parts(q: dict) -> tuple[str, str, str]:
    terms = q.get("aspect", {}).get("term", [])
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
    if not text or text.strip().lower() in ("<empty>", "none", "n/a", "empty"):
        return []

    triples = []
    for chunk in re.split(r"\s*;\s*", text.strip()):
        parts = chunk.split(FIELD_SEP)
        if len(parts) != 3:
            parts = [p.strip() for p in chunk.split("|")]
        if len(parts) != 3:
            continue

        aspect, category, sentiment = (p.strip() for p in parts)
        aspect_clean = "NULL" if aspect.lower() == "null" or not aspect else aspect
        snt = sentiment.upper()
        if snt not in VALID_SENTIMENTS:
            snt = "POS" if "POS" in snt else ("NEG" if "NEG" in snt else "NEU")

        triples.append({"aspect": aspect_clean, "category": category, "sentiment": snt})
    return triples


def post_process_triples(triples: list[dict], raw_words: str, known_cats: set[str] = None) -> list[dict]:
    seen = set()
    cleaned = []
    raw_lower = raw_words.lower()

    for t in triples:
        asp = t.get("aspect", "NULL").strip()
        cat = t.get("category", "").strip()
        snt = t.get("sentiment", "POS").strip().upper()

        if snt not in VALID_SENTIMENTS:
            snt = "POS"

        # 1. Nắn chỉnh aspect về exact substring
        if asp and asp != "NULL":
            if asp not in raw_words:
                idx = raw_lower.find(asp.lower())
                if idx != -1:
                    asp = raw_words[idx : idx + len(asp)]
                else:
                    cl = asp.strip(".,!?:;\"'()[]{}")
                    idx2 = raw_lower.find(cl.lower())
                    if idx2 != -1:
                        asp = raw_words[idx2 : idx2 + len(cl)]
                    elif cl in raw_words:
                        asp = cl
                    else:
                        asp = "NULL"

        # 2. Chuẩn hóa category
        if known_cats:
            if cat not in known_cats:
                matched = False
                for kc in known_cats:
                    if kc.lower() == cat.lower():
                        cat = kc
                        matched = True
                        break
                if not matched:
                    gen_cands = [kc for kc in known_cats if kc.endswith("#GENERAL")]
                    if gen_cands:
                        cat = gen_cands[0]

        key = (asp, cat, snt)
        if key not in seen:
            seen.add(key)
            cleaned.append({"aspect": asp, "category": cat, "sentiment": snt})
    return cleaned


def generate_domain_predictions(model, tokenizer, items: list[dict], domain: str, device: str, known_cats: set[str]) -> list[dict]:
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

        gen_tokens = outputs[0][input_len:]
        resp_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        raw_t = parse_generated_triples(resp_text)
        clean_t = post_process_triples(raw_t, raw_words, known_cats)
        results.append({"raw_words": raw_words, "triples": clean_t})
    return results


def main():
    parser = argparse.ArgumentParser(description="Boost Qwen 56.54 with further epochs")
    parser.add_argument("--base_model", type=str, default="models/qwen3-0.6B")
    parser.add_argument("--ck1_path", type=str, default="outputs/qwen_all/checkpoint-best")
    parser.add_argument("--ck2_path", type=str, default="outputs/qwen_boost_laptop_clothing/checkpoint-best")
    parser.add_argument("--dataset_root", type=str, default="data/TACVU1")
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_boost_v2")
    parser.add_argument("--domains", type=str, default="all", help="'all' hoặc danh sách domain ngăn bởi dấu phẩy, ví dụ 'Restaurant,Hotel,Books'")
    parser.add_argument("--epochs", type=int, default=3, help="Số epochs huấn luyện tiếp")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (4 tối ưu tốc độ trên MPS)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation (eff bs = 4*4=16)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (1e-4 cho LoRA)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"==================================================")
    print(f"🚀 TIẾP TỤC HUẤN LUYỆN QWEN (KẾ THỪA MỐC 56.54 ĐIỂM)")
    print(f"   Thiết bị: {device.upper()}")
    print(f"   Số epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size} (accum={args.gradient_accumulation_steps}, eff_bs={args.batch_size*args.gradient_accumulation_steps})")
    print(f"   Domains: {args.domains}")
    print(f"==================================================")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Nạp và hợp nhất tuần tự Stage 1 và Stage 2
    tokenizer = AutoTokenizer.from_pretrained(args.ck2_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f">> Nạp Base model từ {args.base_model}...")
    model_dtype = torch.float32 if device == "cpu" else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=True, dtype=model_dtype)

    print(f">> Hợp nhất Stage 1 từ: {args.ck1_path}...")
    base_model = PeftModel.from_pretrained(base_model, args.ck1_path, local_files_only=True).merge_and_unload()

    print(f">> Hợp nhất Stage 2 từ: {args.ck2_path} (mốc 56.54)...")
    base_model = PeftModel.from_pretrained(base_model, args.ck2_path, local_files_only=True).merge_and_unload()

    if hasattr(base_model, "peft_config"):
        delattr(base_model, "peft_config")

    print("✅ Đã hợp nhất hoàn hảo trọng số 56.54 vào base model!")

    # 2. Chuẩn bị dữ liệu Train + Dev
    root = Path(args.dataset_root)
    with open(root / "train/Train.json") as f:
        train_raw = json.load(f)
    with open(root / "train/Dev.json") as f:
        dev_raw = json.load(f)

    if args.domains.strip().lower() == "all":
        target_domains = ALL_DOMAINS
    else:
        target_domains = [d.strip() for d in args.domains.split(",") if d.strip() in ALL_DOMAINS]

    known_categories = {d: set() for d in ALL_DOMAINS}
    for d in ALL_DOMAINS:
        for item in train_raw.get(d, []) + dev_raw.get(d, []):
            for q in item.get("quadruples", []):
                cat = (q.get("category") or "").strip()
                if cat:
                    known_categories[d].add(cat)

    train_samples = []
    for d in target_domains:
        all_items = train_raw[d] + dev_raw[d]
        print(f"Domain [{d}]: Gom {len(train_raw[d])} câu (Train) + {len(dev_raw[d])} câu (Dev) = {len(all_items)} câu")
        for item in all_items:
            tgt_str = target_text_from_quadruples(item["quadruples"])
            pmt_str = build_prompt(item["raw_words"], d)
            sample = prepare_training_sample(pmt_str, tgt_str, tokenizer)
            train_samples.append(sample)

    print(f">> Tổng số mẫu huấn luyện đợt này: {len(train_samples)}")

    # 3. Gắn LoRA Adapter mới
    base_model.config.use_cache = False
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ACSTEDataset(train_samples)
    data_collator = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=False,
        bf16=False,
        report_to="none",
        dataloader_pin_memory=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print(f"\n=== BẮT ĐẦU HUẤN LUYỆN ({args.epochs} EPOCHS) ===")
    trainer.train()

    best_checkpoint_dir = out_dir / "checkpoint-best"
    print(f"\n>> Đang lưu checkpoint tốt nhất vào: {best_checkpoint_dir}")
    trainer.model.save_pretrained(best_checkpoint_dir)
    tokenizer.save_pretrained(best_checkpoint_dir)

    # 4. Sinh kết quả PrivateTest & Tích hợp vào Submission
    print("\n=== ĐANG SINH KẾT QUẢ DỰ ĐOÁN PRIVATE TEST ===")
    with open(root / "private_test/PrivateTest.json") as f:
        private_test_raw = json.load(f)

    # Nạp base submission hiện tại (56.54)
    base_sub_path = Path("submission_tacvu1/submission_qwen_56.54.json")
    if base_sub_path.exists():
        with open(base_sub_path) as f:
            submission_data = json.load(f)
    else:
        submission_data = {}

    for d in target_domains:
        test_items = private_test_raw[d]
        d_preds = generate_domain_predictions(
            model=model,
            tokenizer=tokenizer,
            items=test_items,
            domain=d,
            device=device,
            known_cats=known_categories[d],
        )
        submission_data[d] = d_preds
        print(f">> Cập nhật domain [{d}]: {len(d_preds)} câu")

    # Lưu submission.json
    final_sub_path = out_dir / "submission.json"
    with open(final_sub_path, "w", encoding="utf-8") as f:
        json.dump(submission_data, f, ensure_ascii=False, indent=2)
    print(f">> Đã lưu submission.json tại: {final_sub_path}")

    # Đồng bộ vào submission_tacvu1
    tacvu1_sub = Path("submission_tacvu1/submission.json")
    with open(tacvu1_sub, "w", encoding="utf-8") as f:
        json.dump(submission_data, f, ensure_ascii=False, indent=2)

    # 5. Đóng gói submission.zip
    zip_path = Path("submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(final_sub_path, arcname="submission.json")
        z.write("generate_result.ipynb", arcname="generate_result.ipynb")
        z.write("main.ipynb", arcname="main.ipynb")
        z.write("train_boost_qwen_v2.py", arcname="train_qwen.py")
        z.write("submission_tacvu1/README.md", arcname="README.md")
        z.write("submission_tacvu1/evaluation_script.py", arcname="evaluation_script.py")

    print(f"\n📦 ĐÃ ĐÓNG GÓI THÀNH CÔNG: {zip_path.resolve()} ({zip_path.stat().st_size // 1024} KB)")
    print("✅ TOÀN BỘ TIẾN TRÌNH HOÀN TẤT!")


if __name__ == "__main__":
    main()
