"""
SatQuery AI — Base Model Vision-Language Inference Verification
SIH 2026 (PS 26167)

Tests both local copies of Qwen/Qwen2.5-VL-3B-Instruct:
1. Specialist A Base (models/base/specialist_a_base)
2. Specialist B Base (models/base/specialist_b_base)

Loads each model on GPU (if CUDA available, with torch.bfloat16 or 4-bit) / CPU
and produces a vision-language caption for a test satellite image.
"""

import os
import sys
import logging
import argparse
from PIL import Image
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestBaseInference")

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_BASE_DIR = os.path.join(ROOT_DIR, "models", "base")
DIR_SPECIALIST_A = os.path.join(MODELS_BASE_DIR, "specialist_a_base")
DIR_SPECIALIST_B = os.path.join(MODELS_BASE_DIR, "specialist_b_base")


def get_test_image() -> Image.Image:
    """
    Finds an available satellite test image from raw data dirs or creates one.
    """
    candidates = [
        os.path.join(ROOT_DIR, "data", "raw", "vrsbench_samples", "vrsbench_sample_0.png"),
        os.path.join(ROOT_DIR, "data", "raw", "rsvqa_lr", "Images_LR", "0.tif"),
        os.path.join(ROOT_DIR, "data", "raw", "second", "im1", "00001.png"),
    ]
    for c in candidates:
        if os.path.exists(c):
            logger.info(f"Using test satellite image from: {c}")
            return Image.open(c).convert("RGB")
            
    # Synthetic fallback
    logger.info("Creating synthetic RGB satellite patch for inference test...")
    import numpy as np
    arr = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    arr[60:180, 60:180] = [80, 180, 90] # vegetation patch
    return Image.fromarray(arr, "RGB")


def run_inference_on_model(model_path: str, model_name: str, test_image: Image.Image) -> str:
    """
    Loads Qwen2.5-VL-3B from local path and runs image captioning inference.
    """
    logger.info(f"\n============================================================")
    logger.info(f"Loading and testing {model_name} from: {model_path}")
    logger.info(f"============================================================")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    logger.info(f"Loading model on device='{device}', dtype={torch_dtype}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if device == "cpu":
        model = model.to("cpu")
        
    logger.info("Loading AutoProcessor...")
    processor = AutoProcessor.from_pretrained(model_path)
    
    prompt_text = "Describe the main land cover and objects visible in this remote sensing satellite image in 2 concise sentences."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": test_image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    logger.info("Preparing model inputs and vision tokens...")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    logger.info("Generating response...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )
        
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    logger.info(f"\n--- {model_name} INFERENCE RESULT ---")
    logger.info(f"Prompt: {prompt_text}")
    logger.info(f"Generated Output: {output_text.strip()}")
    logger.info("------------------------------------\n")
    
    # Free memory
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return output_text.strip()


def test_both_base_models() -> bool:
    test_img = get_test_image()
    
    # 1. Test Specialist A Base
    out_a = run_inference_on_model(DIR_SPECIALIST_A, "Specialist A (VQA/Grounding)", test_img)
    assert len(out_a) > 0, "Specialist A base model produced empty output!"
    
    # 2. Test Specialist B Base
    out_b = run_inference_on_model(DIR_SPECIALIST_B, "Specialist B (Change-VQA)", test_img)
    assert len(out_b) > 0, "Specialist B base model produced empty output!"
    
    logger.info("=" * 60)
    logger.info("✓ BOTH BASE MODEL COPIES LOADED AND PASSED INFERENCE TEST CLEANLY!")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Base Model Inference")
    parser.add_argument("--model", choices=["specialist_a", "specialist_b", "all"], default="all")
    args = parser.parse_args()
    
    try:
        img = get_test_image()
        if args.model == "specialist_a":
            run_inference_on_model(DIR_SPECIALIST_A, "Specialist A Base", img)
        elif args.model == "specialist_b":
            run_inference_on_model(DIR_SPECIALIST_B, "Specialist B Base", img)
        else:
            test_both_base_models()
        sys.exit(0)
    except Exception as exc:
        logger.exception(f"Base model test failed: {exc}")
        sys.exit(1)
