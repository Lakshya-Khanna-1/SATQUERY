"""
SatQuery AI — Easy Test Set Evaluation & Benchmarking Script (All 25 Samples / 91 QA Pairs)
SIH 2026 (PS 26167)

Evaluates Specialist A on easy_test_set
"""

import os
import sys
import json
import re
import logging
from typing import List, Dict, Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalEasyTestSet")

TEST_SET_DIR = r"C:\Users\Admitrator\Desktop\easy-test-pull\easy_test_set"
JSON_PATH = os.path.join(TEST_SET_DIR, "questions_answers.json")
IMAGES_DIR = os.path.join(TEST_SET_DIR, "images")


def clean_question(q_raw: str) -> str:
    q = re.sub(r'<image>\s*\[vqa\]\s*', '', q_raw).strip()
    q = re.sub(r'\.?\s*(A short answer to the question is|Short answer:)*$', '', q, flags=re.IGNORECASE).strip()
    return q


def run_evaluation(
    base_model_path: str = "models/base/specialist_a_base",
    adapter_path: str = "models/adapters/specialist_a"
) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("EVALUATING SPECIALIST A ON EASY-TEST-PULL DATASET (ALL 25 SAMPLES)")
    logger.info(f"Base Model: {base_model_path}")
    logger.info(f"Adapter   : {adapter_path}")
    logger.info("=" * 70)
    
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"Questions JSON not found at {JSON_PATH}")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading processor and base model on {device} (4-bit NF4)...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    if os.path.exists(adapter_path):
        logger.info(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        logger.warning(f"No adapter found at {adapter_path}. Evaluating base model zero-shot.")
        
    model.eval()

    total_questions = 0
    correct_questions = 0
    sample_reports = []

    for i, sample in enumerate(samples):
        img_filename = sample["image_file"]
        img_path = os.path.join(IMAGES_DIR, img_filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(TEST_SET_DIR, img_filename)
        if not os.path.exists(img_path):
            logger.warning(f"Image missing: {img_filename}")
            continue

        pil_img = Image.open(img_path).convert("RGB")
        convs = sample.get("conversations", [])

        sample_eval = {
            "sample_index": i,
            "image_file": img_filename,
            "qa_results": []
        }

        logger.info(f"\n--- Sample {i:02d} ({img_filename}) ---")
        q_counter = 0

        for j in range(len(convs) - 1):
            if convs[j]["from"] == "human" and "[vqa]" in convs[j]["value"]:
                q_counter += 1
                question = clean_question(convs[j]["value"])
                ground_truth = convs[j+1]["value"].strip().lower()

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": f"Answer concisely in 1-5 words: {question}"}
                        ]
                    }
                ]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=32)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    pred = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

                pred_clean = pred.lower()
                is_match = (pred_clean == ground_truth) or (ground_truth in pred_clean) or (pred_clean in ground_truth)

                total_questions += 1
                if is_match:
                    correct_questions += 1

                status_mark = "✓ MATCH" if is_match else "✗ MISMATCH"
                logger.info(f" Q{q_counter}: '{question}'")
                logger.info(f"   Ground Truth : {ground_truth}")
                logger.info(f"   Prediction   : {pred} [{status_mark}]")

                sample_eval["qa_results"].append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": pred,
                    "is_match": is_match
                })

        sample_reports.append(sample_eval)

    accuracy = (correct_questions / max(1, total_questions)) * 100
    summary = {
        "total_samples": len(samples),
        "total_questions": total_questions,
        "correct_questions": correct_questions,
        "accuracy_pct": round(accuracy, 2),
        "sample_reports": sample_reports
    }

    logger.info("\n" + "=" * 70)
    logger.info(f"EASY-TEST-PULL EVALUATION SUMMARY:")
    logger.info(f" Accuracy: {accuracy:.2f}% ({correct_questions}/{total_questions})")
    logger.info("=" * 70)

    return summary


if __name__ == "__main__":
    res = run_evaluation()
    os.makedirs("reports", exist_ok=True)
    with open("reports/easy_test_pull_eval_report.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    logger.info("Report written to reports/easy_test_pull_eval_report.json")
