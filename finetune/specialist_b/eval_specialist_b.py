"""
SatQuery AI — Specialist B Evaluation Script

Evaluates Specialist B on CDVQA Test/Test2 splits, reporting:
1. Overall Change-VQA Accuracy
2. Accuracy breakdown by CDVQA question type:
   - Change detection ("Has there been a change?")
   - Increase / Decrease ("Did vegetation increase or decrease?")
   - Extent / Area estimation ("What is the extent of new buildings?")
   - Max / Min change ("Which land class changed the most?")
   - Rate of change / Proportion
"""

import os
import sys
import argparse
import logging
from collections import defaultdict
from typing import List, Dict, Any

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.loaders.cdvqa_loader import CDVQADataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalSpecialistB")


def classify_question_type(query: str) -> str:
    q = query.lower()
    if "increase" in q or "decrease" in q or "more" in q or "less" in q:
        return "increase_decrease"
    elif "most" in q or "least" in q or "largest" in q or "smallest" in q or "max" in q or "min" in q:
        return "max_min_change"
    elif "area" in q or "extent" in q or "how much" in q or "percentage" in q or "percent" in q:
        return "extent_rate"
    elif "change" in q or "happen" in q or "appear" in q or "disappear" in q:
        return "change_detection"
    else:
        return "general_change_vqa"


def evaluate_specialist_b(
    base_model_path: str = "models/base/specialist_b_base",
    adapter_path: str = "models/adapters/specialist_b",
    split: str = "test",
    max_eval_samples: int = 50
):
    logger.info("=" * 60)
    logger.info("EVALUATING SPECIALIST B — BI-TEMPORAL CHANGE-VQA")
    logger.info(f"Base: {base_model_path} | Adapter: {adapter_path} | Split: {split}")
    logger.info("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Processor & Model
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if os.path.exists(adapter_path):
        logger.info(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        logger.warning(f"No adapter found at {adapter_path}. Evaluating zero-shot base model.")
        
    model.eval()
    
    # 2. Load CDVQA Evaluation Set
    cdvqa_test = CDVQADataset(split=split, max_samples=max_eval_samples)
    logger.info(f"Evaluating on {len(cdvqa_test)} CDVQA {split} samples...")
    
    type_correct = defaultdict(int)
    type_total = defaultdict(int)
    results = []
    
    for i in range(len(cdvqa_test)):
        sample = cdvqa_test[i]
        images = sample["images"]
        query = sample["query"]
        ground_truth = sample["answer"].strip().lower()
        q_type = classify_question_type(query)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image 1 (Time 1 / Pre-change):"},
                    {"type": "image", "image": images[0]},
                    {"type": "text", "text": "Image 2 (Time 2 / Post-change):"},
                    {"type": "image", "image": images[1]},
                    {"type": "text", "text": f"Answer concisely in 1-3 words: {query}"}
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
            pred = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip().lower()
            
        is_match = (pred == ground_truth) or (ground_truth in pred) or (pred in ground_truth)
        type_total[q_type] += 1
        if is_match:
            type_correct[q_type] += 1
            
        results.append({
            "type": q_type,
            "query": query,
            "ground_truth": ground_truth,
            "prediction": pred,
            "correct": is_match
        })
        
    total_samples = len(results)
    overall_correct = sum(type_correct.values())
    overall_acc = (overall_correct / max(1, total_samples)) * 100
    
    logger.info("\n" + "=" * 60)
    logger.info("SPECIALIST B EVALUATION BREAKDOWN:")
    logger.info(f" - OVERALL ACCURACY: {overall_acc:.2f}% ({overall_correct}/{total_samples})")
    logger.info("-" * 60)
    for q_type, count in type_total.items():
        acc = (type_correct[q_type] / count) * 100
        logger.info(f"   * {q_type:<22}: {acc:6.2f}% ({type_correct[q_type]}/{count})")
    logger.info("=" * 60)
    
    return {
        "overall_accuracy": overall_acc,
        "type_breakdown": {qt: (type_correct[qt]/type_total[qt])*100 for qt in type_total},
        "sample_predictions": results[:5]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specialist B Evaluation CLI")
    parser.add_argument("--base_model", default="models/base/specialist_b_base")
    parser.add_argument("--adapter", default="models/adapters/specialist_b")
    parser.add_argument("--split", default="val", choices=["val", "test", "test2"])
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    
    evaluate_specialist_b(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        split=args.split,
        max_eval_samples=args.samples
    )
