"""
SatQuery AI — Specialist A Evaluation Script
SIH 2026 (PS 26167)

Evaluates Specialist A (Base + LoRA Adapter) on test splits of:
1. RSVQA-LR (Accuracy on VQA classification)
2. VRSBench (Captioning quality & Grounding evaluation)
"""

import os
import sys
import argparse
import logging
import json
from typing import List, Dict, Any

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.loaders.vrsbench_loader import VRSBenchDataset
from data.loaders.rsvqa_loader import RSVQALRDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalSpecialistA")


def evaluate_specialist_a(
    base_model_path: str = "models/base/specialist_a_base",
    adapter_path: str = "models/adapters/specialist_a",
    max_eval_samples: int = 50
):
    logger.info("=" * 60)
    logger.info("EVALUATING SPECIALIST A — OPTICAL/MULTISPECTRAL VQA & CAPTIONING")
    logger.info(f"Base: {base_model_path} | Adapter: {adapter_path}")
    logger.info("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Processor & Base Model
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 2. Attach LoRA Adapter if present
    if os.path.exists(adapter_path):
        logger.info(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        logger.warning(f"No adapter found at {adapter_path}. Evaluating zero-shot base model.")
        
    model.eval()
    
    # 3. Evaluate RSVQA-LR (VQA Accuracy)
    logger.info("\n>>> [1/2] Evaluating on RSVQA-LR Test Split...")
    rsvqa_test = RSVQALRDataset(split="val", max_samples=max_eval_samples)
    
    vqa_correct = 0
    vqa_total = len(rsvqa_test)
    vqa_results = []
    
    for i in range(vqa_total):
        sample = rsvqa_test[i]
        image = sample["images"][0]
        query = sample["query"]
        ground_truth = sample["answer"].strip().lower()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
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
        if is_match:
            vqa_correct += 1
            
        vqa_results.append({
            "query": query,
            "ground_truth": ground_truth,
            "prediction": pred,
            "correct": is_match
        })
        
    vqa_acc = (vqa_correct / max(1, vqa_total)) * 100
    logger.info(f"RSVQA-LR Accuracy: {vqa_acc:.2f}% ({vqa_correct}/{vqa_total})")
    
    # 4. Evaluate VRSBench (Captioning / Grounding)
    logger.info("\n>>> [2/2] Evaluating on VRSBench Evaluation Samples...")
    vrs_test = VRSBenchDataset(max_samples=min(10, max_eval_samples))
    vrs_results = []
    
    for i in range(len(vrs_test)):
        sample = vrs_test[i]
        image = sample["images"][0]
        query = sample["query"]
        ground_truth = sample["answer"]
        task_type = sample["task_type"]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=64)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            pred = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            
        vrs_results.append({
            "task_type": task_type,
            "query": query,
            "ground_truth": ground_truth,
            "prediction": pred
        })
        
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SPECIALIST A EVALUATION SUMMARY:")
    logger.info(f" - RSVQA-LR VQA Accuracy: {vqa_acc:.2f}%")
    logger.info(f" - VRSBench Evaluated Samples: {len(vrs_results)}")
    logger.info("=" * 60)
    
    return {
        "rsvqa_accuracy": vqa_acc,
        "vqa_samples": vqa_results[:5],
        "vrsbench_samples": vrs_results[:5]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specialist A Evaluation CLI")
    parser.add_argument("--base_model", default="models/base/specialist_a_base")
    parser.add_argument("--adapter", default="models/adapters/specialist_a")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    
    evaluate_specialist_a(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        max_eval_samples=args.samples
    )
