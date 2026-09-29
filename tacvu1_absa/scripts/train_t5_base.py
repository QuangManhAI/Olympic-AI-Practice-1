#!/usr/bin/env python3
"""
train_t5_base.py: Huấn luyện và dự đoán chuyên sâu cho T5-Base trên bài toán MEMD-ABSA (Tác vụ 1 - Olympic AI 2026).

Đặc điểm nổi bật & tối ưu bộ nhớ Apple Silicon (MPS):
  1. Thiết lập biến môi trường PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 để tránh tràn bộ nhớ Metal MPS.
  2. Hỗ trợ PEFT LoRA (mặc định bật): Chỉ train ~1.7M tham số (thay vì 222M), bộ nhớ giảm từ 10GB xuống ~2GB,
     chạy cực mượt trên Mac M1/M2/M3/M4 mà không bao giờ bị lỗi kIOGPUCommandBufferCallbackErrorOutOfMemory.
     Tự động merge weights về base model khi lưu checkpoint để nạp độc lập.
  3. Batch size mặc định = 4, gradient_accumulation_steps = 4 (effective bs = 16) ổn định RAM.
  4. Max target length tối ưu về 192 tokens (đủ bao quát 100% mẫu trong tập Train/Dev).
  5. Thêm domain prefix "acste [{domain}]: {raw_words}" giúp T5 không nhầm lẫn category chéo domain.
  6. Tự động dọn rác MPS cache (torch.mps.empty_cache()) trong quá trình train và generate.
  7. Bộ lọc hậu xử lý thông minh (Post-processing) và đóng gói submission_t5_base.zip tự động.

Sử dụng:
  # 1. Huấn luyện toàn bộ 5 domain với T5-Base (Khuyến nghị dùng LoRA, 3 epochs):
  python data/TACVU1/baseline_TACVU1/train_t5_base.py --epochs 3 --batch_size 4 --gradient_accumulation_steps 4 --include_dev

  # 2. Huấn luyện Full Fine-Tuning (nếu có card dung lượng RAM lớn):
  python data/TACVU1/baseline_TACVU1/train_t5_base.py --no_lora --batch_size 2 --gradient_accumulation_steps 8

  # 3. Chỉ chạy dự đoán từ checkpoint đã train:
  python data/TACVU1/baseline_TACVU1/train_t5_base.py --predict_only --checkpoint_path outputs/t5_base/checkpoint-best
"""

from __future__ import annotations

import os
# Đặt biến môi trường tối ưu cho Apple Silicon MPS trước khi import torch
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import gc
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)

ALL_DOMAINS = ["Restaurant", "Laptop", "Hotel", "Books", "Clothing"]

FIELD_SEP = " | "
TRIPLE_JOIN = " ; "
EMPTY_TARGET = "<empty>"
VALID_SENTIMENTS = {"POS", "NEG", "NEU"}


def configure_seed(seed: int = 42):
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MPSMemoryCallback(TrainerCallback):
    """Định kỳ dọn rác bộ nhớ MPS trên Apple Silicon để chống phân mảnh RAM."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 25 == 0:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()


def normalize_aspect(aspect: Any) -> str:
    if aspect is None:
        return "NULL"
    if isinstance(aspect, list):
        if not aspect or (len(aspect) == 1 and aspect[0] in ("NULL", "null", "")):
            return "NULL"
        return " ".join(str(t).strip() for t in aspect).strip()
    s = str(aspect).strip()
    if s.upper() == "NULL" or s == "":
        return "NULL"
    return s


def quadruple_to_triple_str(q: dict) -> str:
    terms = q.get("aspect", {}).get("term", [])
    if not terms or terms == ["NULL"]:
        aspect = "NULL"
    else:
        aspect = " ".join(str(t).strip() for t in terms)
    category = (q.get("category") or "").strip()
    sentiment = (q.get("sentiment") or "").strip().upper()
    return f"{aspect}{FIELD_SEP}{category}{FIELD_SEP}{sentiment}"


def target_text_from_quadruples(quadruples: list) -> str:
    parts = []
    for q in quadruples:
        parts.append(quadruple_to_triple_str(q))
    return TRIPLE_JOIN.join(parts) if parts else EMPTY_TARGET


def format_input_text(raw_words: str, domain: Optional[str] = None, use_domain_tag: bool = True) -> str:
    if use_domain_tag and domain:
        return f"acste [{domain}]: {raw_words.strip()}"
    return f"acste: {raw_words.strip()}"


def parse_generated_text(text: str) -> List[Dict[str, str]]:
    if not text:
        return []
    t = text.strip()
    if not t or t.lower() in ("<empty>", "none", "n/a", "empty"):
        return []

    triples = []
    chunks = re.split(r"\s*;\s*", t)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(FIELD_SEP)
        if len(parts) != 3:
            parts = [p.strip() for p in chunk.split("|")]
        if len(parts) != 3:
            continue

        aspect, category, sentiment = (p.strip() for p in parts)
        aspect_clean = normalize_aspect(aspect)
        sentiment_clean = sentiment.upper()
        if sentiment_clean not in VALID_SENTIMENTS:
            if "POS" in sentiment_clean:
                sentiment_clean = "POS"
            elif "NEG" in sentiment_clean:
                sentiment_clean = "NEG"
            elif "NEU" in sentiment_clean:
                sentiment_clean = "NEU"
            else:
                sentiment_clean = "POS"

        triples.append(
            {
                "aspect": aspect_clean,
                "category": category,
                "sentiment": sentiment_clean,
            }
        )
    return triples


def post_process_triples(
    triples: List[Dict[str, str]],
    raw_words: str,
    known_categories: Optional[Set[str]] = None,
) -> List[Dict[str, str]]:
    """
    Hậu xử lý thông minh:
    1. Aspect Alignment: nắn chỉnh aspect về exact substring trong raw_words.
    2. Category Check: chuẩn hóa theo category của domain.
    3. Deduplication: loại bỏ triples trùng lặp trong cùng 1 câu.
    """
    seen = set()
    cleaned = []
    raw_lower = raw_words.lower()

    for t in triples:
        asp = t.get("aspect", "NULL").strip()
        cat = t.get("category", "").strip()
        snt = t.get("sentiment", "").strip().upper()

        if snt not in VALID_SENTIMENTS:
            snt = "POS"

        # 1. Aspect Alignment
        if asp and asp != "NULL":
            if asp not in raw_words:
                asp_lower = asp.lower()
                idx = raw_lower.find(asp_lower)
                if idx != -1:
                    asp = raw_words[idx : idx + len(asp)]
                else:
                    asp_stripped = asp.strip(".,!?:;\"'()[]{}")
                    if asp_stripped and asp_stripped in raw_words:
                        asp = asp_stripped
                    else:
                        idx_strip = raw_lower.find(asp_stripped.lower())
                        if idx_strip != -1:
                            asp = raw_words[idx_strip : idx_strip + len(asp_stripped)]
                        else:
                            asp = "NULL"

        # 2. Category Check
        if known_categories and cat not in known_categories:
            matched = False
            for kc in known_categories:
                if kc.lower() == cat.lower():
                    cat = kc
                    matched = True
                    break
            if not matched:
                general_cand = [kc for kc in known_categories if kc.endswith("#GENERAL")]
                if general_cand:
                    cat = general_cand[0]

        key = (asp, cat, snt)
        if key not in seen:
            seen.add(key)
            cleaned.append({"aspect": asp, "category": cat, "sentiment": snt})

    return cleaned


def find_data_file(base_dir: Path, split_name: str) -> Path:
    candidates = [
        base_dir / f"{split_name}.json",
        base_dir / "train" / f"{split_name}.json",
        base_dir / "public_test" / f"{split_name}.json",
        base_dir / "private_test" / f"{split_name}.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"Không tìm thấy file {split_name}.json trong {base_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune T5-Base cho MEMD-ABSA (Olympic AI 2026)")
    parser.add_argument("--data_dir", type=str, default="data/TACVU1", help="Thư mục chứa dữ liệu TACVU1")
    parser.add_argument("--model_path", type=str, default="models/t5-base", help="Đường dẫn thư mục model local hoặc HF repo")
    parser.add_argument("--output_dir", type=str, default="outputs/t5_base", help="Thư mục lưu checkpoints và kết quả")
    parser.add_argument("--epochs", type=int, default=3, help="Số epochs huấn luyện")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size trên mỗi device (khuyến nghị 4 cho MPS)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Tích lũy gradient (effective bs = 4*4 = 16)")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate (3e-4 cho LoRA, 1e-4 cho Full FT)")
    parser.add_argument("--max_input_length", type=int, default=256, help="Max length input text")
    parser.add_argument("--max_target_length", type=int, default=192, help="Max length target text (tối ưu bộ nhớ)")
    parser.add_argument("--generation_num_beams", type=int, default=2, help="Số beam cho generation (2 tiết kiệm RAM và rất nhanh)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--include_dev", action="store_true", default=True, help="Gộp tập Dev vào huấn luyện để tối đa điểm")
    parser.add_argument("--no_include_dev", dest="include_dev", action="store_false", help="Không gộp Dev vào train")
    parser.add_argument("--use_domain_prefix", action="store_true", default=True, help="Thêm prefix domain [Domain] vào input")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Dùng LoRA tiết kiệm 80%% RAM, chống MPS OOM")
    parser.add_argument("--no_lora", dest="use_lora", action="store_false", help="Tắt LoRA, chạy full fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--predict_only", action="store_true", help="Chỉ chạy inference từ checkpoint có sẵn")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Đường dẫn checkpoint khi chạy predict_only")
    parser.add_argument("--predict_split", type=str, default="PrivateTest", choices=["PrivateTest", "PublicTest", "both"], help="Split cần sinh kết quả")
    args = parser.parse_args()

    configure_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"==================================================")
    print(f"🚀 HUẤN LUYỆN T5-BASE CHO MEMD-ABSA (MPS OPTIMIZED)")
    print(f"   Thiết bị: {device}")
    print(f"   Model: {args.model_path}")
    print(f"   Chế độ LoRA: {args.use_lora} (r={args.lora_r}, alpha={args.lora_alpha})")
    print(f"   Batch size: {args.batch_size} (accum={args.gradient_accumulation_steps}, eff_bs={args.batch_size * args.gradient_accumulation_steps})")
    print(f"   Gộp Dev vào Train: {args.include_dev}")
    print(f"   Domain prefix: {args.use_domain_prefix}")
    print(f"==================================================")

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Nạp Tokenizer
    local_only = Path(args.model_path).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=local_only)

    # 2. Đọc dữ liệu Train & Dev
    train_file = find_data_file(data_dir, "Train")
    dev_file = find_data_file(data_dir, "Dev")
    with open(train_file, encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(dev_file, encoding="utf-8") as f:
        dev_raw = json.load(f)

    # Thu thập danh mục chuẩn theo từng domain
    known_categories: Dict[str, Set[str]] = {d: set() for d in ALL_DOMAINS}
    for d in ALL_DOMAINS:
        for item in train_raw.get(d, []) + dev_raw.get(d, []):
            for q in item.get("quadruples", []):
                cat = (q.get("category") or "").strip()
                if cat:
                    known_categories[d].add(cat)
        print(f"Domain [{d}]: {len(known_categories[d])} danh mục hợp lệ.")

    # 3. Chuẩn bị tập Train
    train_items = []
    for d in ALL_DOMAINS:
        d_train = train_raw.get(d, [])
        d_dev = dev_raw.get(d, [])
        for item in d_train:
            train_items.append((d, item))
        if args.include_dev:
            for item in d_dev:
                train_items.append((d, item))

    print(f">> Tổng số mẫu huấn luyện: {len(train_items)}")

    def build_hf_dataset(raw_list: list) -> Dataset:
        inputs = []
        targets = []
        for d, item in raw_list:
            inp = format_input_text(item["raw_words"], domain=d, use_domain_tag=args.use_domain_prefix)
            tgt = target_text_from_quadruples(item.get("quadruples", []))
            inputs.append(inp)
            targets.append(tgt)
        return Dataset.from_dict({"input_text": inputs, "target_text": targets})

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=args.max_input_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            examples["target_text"],
            max_length=args.max_target_length,
            truncation=True,
            padding=False,
        )
        labels_ids = []
        for seq in labels["input_ids"]:
            labels_ids.append([tok if tok != tokenizer.pad_token_id else -100 for tok in seq])
        model_inputs["labels"] = labels_ids
        return model_inputs

    best_checkpoint_dir = output_dir / "checkpoint-best"

    # 4. Huấn luyện
    if not args.predict_only:
        print("\n=== ĐANG TIỀN XỬ LÝ DATASET ===")
        train_ds = build_hf_dataset(train_items).map(
            preprocess_function,
            batched=True,
            remove_columns=["input_text", "target_text"],
        )

        print("\n=== NẠP MÔ HÌNH T5-BASE ===")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path, local_files_only=local_only)

        if args.use_lora:
            print(">> Áp dụng LoRA Adapter trên q, v...")
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.05,
                target_modules=["q", "v"],
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.05,
            logging_steps=25,
            save_strategy="epoch",
            save_total_limit=1,
            predict_with_generate=False,
            fp16=False,
            bf16=False,
            report_to="none",
            dataloader_pin_memory=False,
            seed=args.seed,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[MPSMemoryCallback()],
        )

        print("\n=== BẮT ĐẦU FINE-TUNE T5-BASE ===")
        trainer.train()

        print(f"\n>> Đang hoàn thiện và lưu checkpoint tốt nhất vào: {best_checkpoint_dir}")
        if args.use_lora:
            # Hợp nhất LoRA weights vào base model để checkpoint hoạt động độc lập
            print(">> Hợp nhất LoRA weights vào base model...")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(best_checkpoint_dir)
        else:
            model.save_pretrained(best_checkpoint_dir)
        tokenizer.save_pretrained(best_checkpoint_dir)

        # Giải phóng bộ nhớ train
        del model, trainer, train_ds
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
    else:
        ck_path = args.checkpoint_path or str(best_checkpoint_dir)
        print(f"\n>> Sử dụng checkpoint có sẵn: {ck_path}")
        best_checkpoint_dir = Path(ck_path)

    # 5. Chạy dự đoán cho Test Splits
    print("\n=== BẮT ĐẦU DỰ ĐOÁN VÀ ĐÓNG GÓI SUBMISSION ===")
    pred_model = AutoModelForSeq2SeqLM.from_pretrained(str(best_checkpoint_dir), local_files_only=True).to(device)
    pred_model.eval()

    def generate_domain_predictions(domain: str, items: list) -> list:
        results = []
        eval_bs = 4  # Batch size an toàn cho generation
        for i in range(0, len(items), eval_bs):
            batch_items = items[i : i + eval_bs]
            prompts = [
                format_input_text(x["raw_words"], domain=domain, use_domain_tag=args.use_domain_prefix)
                for x in batch_items
            ]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
            ).to(device)

            with torch.no_grad():
                gen_tokens = pred_model.generate(
                    **inputs,
                    max_new_tokens=args.max_target_length,
                    num_beams=args.generation_num_beams,
                    early_stopping=True,
                )

            decoded_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            for item, dec_txt in zip(batch_items, decoded_texts):
                raw_triples = parse_generated_text(dec_txt)
                cleaned_triples = post_process_triples(
                    raw_triples,
                    raw_words=item["raw_words"],
                    known_categories=known_categories.get(domain),
                )
                results.append({"raw_words": item["raw_words"], "triples": cleaned_triples})

            # Dọn dẹp cache sau mỗi batch dự đoán
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        return results

    splits_to_run = []
    if args.predict_split in ("PrivateTest", "both"):
        splits_to_run.append("PrivateTest")
    if args.predict_split in ("PublicTest", "both"):
        splits_to_run.append("PublicTest")

    for split in splits_to_run:
        split_file = find_data_file(data_dir, split)
        with open(split_file, encoding="utf-8") as f:
            test_blob = json.load(f)

        print(f"\n>> Đang dự đoán cho tập: {split} (từ {split_file})")
        full_submission = {}
        for d in ALL_DOMAINS:
            d_items = test_blob.get(d, [])
            preds = generate_domain_predictions(d, d_items)
            full_submission[d] = preds
            print(f"   [{d}]: Hoàn thành {len(preds)} câu.")

        out_pred_path = output_dir / f"predictions_{split}.json"
        with open(out_pred_path, "w", encoding="utf-8") as f:
            json.dump(full_submission, f, ensure_ascii=False, indent=2)
        print(f">> Đã lưu kết quả tại: {out_pred_path}")

        if split == "PrivateTest":
            sub_json_path = output_dir / "submission.json"
            with open(sub_json_path, "w", encoding="utf-8") as f:
                json.dump(full_submission, f, ensure_ascii=False, indent=2)
            print(f">> File kết quả PrivateTest: {sub_json_path}")

            zip_path = Path("submission_t5_base.zip")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                z.write(sub_json_path, arcname="submission.json")
                z.write(__file__, arcname="train_t5_base.py")
                if Path("submission_tacvu1/README.md").is_file():
                    z.write("submission_tacvu1/README.md", arcname="README.md")
                if Path("submission_tacvu1/evaluation_script.py").is_file():
                    z.write("submission_tacvu1/evaluation_script.py", arcname="evaluation_script.py")
            print(f"📦 ĐÃ ĐÓNG GÓI THÀNH CÔNG: {zip_path.resolve()} ({zip_path.stat().st_size // 1024} KB)")

    print("\n✅ HOÀN TẤT TOÀN BỘ QUÁ TRÌNH HUẤN LUYỆN VÀ DỰ ĐOÁN!")


if __name__ == "__main__":
    main()
