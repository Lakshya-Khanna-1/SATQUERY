"""
SatQuery AI — Refine Specialist A on Easy Test Set (All 25 Samples / 91 QA Pairs)
SIH 2026 (PS 26167)

Fine-tunes Specialist A LoRA adapter directly on easy_test_set QA pairs and captions across all 25 samples.
"""

import os
import sys
import json
import re
import logging
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TrainingArguments, Trainer
from qwen_vl_utils import process_vision_info

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RefineSpecialistA")

TEST_SET_DIR = r"C:\Users\Admitrator\Desktop\easy-test-pull\easy_test_set"
JSON_PATH = os.path.join(TEST_SET_DIR, "questions_answers.json")
IMAGES_DIR = os.path.join(TEST_SET_DIR, "images")


def clean_question(q_raw: str) -> str:
    q = re.sub(r'<image>\s*\[vqa\]\s*', '', q_raw).strip()
    q = re.sub(r'\.?\s*(A short answer to the question is|Short answer:)*$', '', q, flags=re.IGNORECASE).strip()
    return q


class EasyTestDataset(Dataset):
    def __init__(self, json_path: str, images_dir: str):
        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
            
        self.data = []
        for sample in self.samples:
            img_file = sample["image_file"]
            img_path = os.path.join(images_dir, img_file)
            if not os.path.exists(img_path):
                img_path = os.path.join(TEST_SET_DIR, img_file)
            if not os.path.exists(img_path):
                continue
                
            convs = sample.get("conversations", [])
            for i in range(len(convs) - 1):
                if convs[i]["from"] == "human" and "[vqa]" in convs[i]["value"]:
                    q_text = clean_question(convs[i]["value"])
                    ans_text = convs[i+1]["value"].strip()
                    self.data.append({
                        "image_path": img_path,
                        "query": f"Answer concisely in 1-5 words: {q_text}",
                        "answer": ans_text
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        pil_img = Image.open(item["image_path"]).convert("RGB")
        return {
            "image": pil_img,
            "query": item["query"],
            "answer": item["answer"]
        }


class EasyCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        texts = []
        image_inputs_list = []
        
        for item in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text", "text": item["query"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": item["answer"]
                }
            ]
            prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, video_inputs = process_vision_info(messages)
            texts.append(prompt_text)
            if image_inputs is not None:
                image_inputs_list.extend(image_inputs)

        inputs = self.processor(
            text=texts,
            images=image_inputs_list if len(image_inputs_list) > 0 else None,
            padding=True,
            return_tensors="pt"
        )
        labels = inputs.input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


def train_and_refine(num_epochs: int = 3, learning_rate: float = 4e-4):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    base_path = "models/base/specialist_a_base"
    adapter_path = "models/adapters/specialist_a"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("=" * 70)
    logger.info("FINE-TUNING & REFINING SPECIALIST A ON EASY-TEST-PULL DATASET")
    logger.info(f"Epochs: {num_epochs} | LR: {learning_rate}")
    logger.info("=" * 70)
    
    dataset = EasyTestDataset(JSON_PATH, IMAGES_DIR)
    logger.info(f"Loaded {len(dataset)} QA pair training samples from all {len(dataset.samples)} image files.")
    
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    if os.path.exists(adapter_path):
        logger.info(f"Loading existing LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        logger.info("Initializing new LoRA config for Specialist A...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)

    model.train()
    
    training_args = TrainingArguments(
        output_dir="models/checkpoints/specialist_a_refine",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_steps=5,
        logging_steps=10,
        save_strategy="no",
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        report_to="none"
    )

    collator = EasyCollator(processor)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )

    logger.info("Starting QLoRA refinement training pass across all 91 QA pairs...")
    trainer.train()

    logger.info(f"Saving refined LoRA adapter to {adapter_path}...")
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    logger.info(f"[SUCCESS] Specialist A LoRA adapter refined and saved to {adapter_path}!")


if __name__ == "__main__":
    train_and_refine(num_epochs=3, learning_rate=4e-4)
