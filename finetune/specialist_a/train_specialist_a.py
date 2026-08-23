"""
SatQuery AI — Specialist A QLoRA Fine-Tuning Script
SIH 2026 (PS 26167)

Specialist A: Single-Image Optical & Multispectral VQA, Captioning, and Spatial Grounding.
Base Model: Qwen/Qwen2.5-VL-3B-Instruct (4-bit NF4 Quantization)
Datasets: VRSBench (Captioning & Grounding) + RSVQA-LR (Visual Question Answering)
Target Modules: Attention projections (q_proj, k_proj, v_proj, o_proj) + MLP projections (gate_proj, up_proj, down_proj)
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, ConcatDataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from qwen_vl_utils import process_vision_info
from PIL import Image

# Import SatQuery Unified Loaders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.loaders.vrsbench_loader import VRSBenchDataset
from data.loaders.rsvqa_loader import RSVQALRDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainSpecialistA")


class SpecialistACollator:
    """
    Collator that formats multi-modal VQA and Captioning samples into tokenized Qwen2.5-VL batches.
    """
    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    def __call__(self, batch: Any) -> Dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            items = [batch]
        elif isinstance(batch, (list, tuple)):
            items = batch
        else:
            items = [batch]

        messages_list = []
        for item in items:
            images = item.get("images", [])
            query = item.get("query", "")
            answer = item.get("answer", "")
            
            # Format conversational prompt
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": query})
            
            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            messages_list.append(messages)

        # Prepare inputs for Qwen2.5-VL
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages_list
        ]
        image_inputs, video_inputs = process_vision_info(messages_list)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Labels for causal LM training (masking prompt tokens if desired, or full sequence loss)
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        
        return inputs


def setup_model_and_lora(base_model_path: str, lora_r: int = 16, lora_alpha: int = 32):
    logger.info(f"Loading base model in 4-bit NF4 from {base_model_path}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    return peft_model, processor


def train_specialist_a(
    base_model_path: str = "models/base/specialist_a_base",
    output_dir: str = "models/adapters/specialist_a",
    max_samples: int = 500,
    epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4
):
    os.makedirs(output_dir, exist_ok=True)
    logger.info("=" * 60)
    logger.info("TRAINING SPECIALIST A — OPTICAL/MULTISPECTRAL VQA & CAPTIONING")
    logger.info("=" * 60)
    
    # 1. Load Datasets
    logger.info("Loading VRSBench and RSVQA-LR unified datasets...")
    vrsbench_ds = VRSBenchDataset(max_samples=max_samples // 2)
    rsvqa_ds = RSVQALRDataset(split="train", max_samples=max_samples // 2)
    
    combined_dataset = ConcatDataset([vrsbench_ds, rsvqa_ds])
    logger.info(f"Total training samples: {len(combined_dataset)} (VRSBench: {len(vrsbench_ds)}, RSVQA: {len(rsvqa_ds)})")
    
    # 2. Setup Model & Processor
    model, processor = setup_model_and_lora(base_model_path)
    collator = SpecialistACollator(processor)
    
    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=2,
        logging_steps=1,
        save_strategy="no",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=0
    )
    
    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        data_collator=collator
    )
    
    logger.info("Starting QLoRA fine-tuning...")
    trainer.train()
    
    # 5. Save Adapter and Processor
    logger.info(f"Saving Specialist A LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info(f"[SUCCESS] Specialist A adapter ready at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specialist A QLoRA Training CLI")
    parser.add_argument("--base_model", default="models/base/specialist_a_base", help="Path to base model")
    parser.add_argument("--output_dir", default="models/adapters/specialist_a", help="Output adapter dir")
    parser.add_argument("--samples", type=int, default=100, help="Max training samples")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()
    
    train_specialist_a(
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        max_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr
    )
