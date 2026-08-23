"""
SatQuery AI — Specialist A Microservice (VQA / Captioning / Grounding)
Port: 8001
SIH 2026 (PS 26167)
"""

import os
import sys
import io
import base64
import logging
from typing import List, Dict, Any, Optional

import torch
from PIL import Image
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VQAService")

app = FastAPI(
    title="SatQuery AI — Specialist A Microservice (VQA / Caption / Grounding)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODEL = None
_PROCESSOR = None


def get_specialist_a():
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_path = "models/base/specialist_a_base"
        adapter_path = "models/adapters/specialist_a"
        logger.info(f"Loading Specialist A base model from {base_path}...")
        _PROCESSOR = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_path,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        if os.path.exists(adapter_path):
            logger.info(f"Attaching LoRA adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        _MODEL = model
    return _MODEL, _PROCESSOR


class InferRequest(BaseModel):
    images: List[str] = Field(..., description="List of base64 images (single optical/multispectral)")
    query: str = Field(..., description="Query or prompt")
    task_type: Optional[str] = Field("vqa", description="Task type")


def run_native_vqa(image_b64: str, prompt: str) -> Dict[str, Any]:
    model, processor = get_specialist_a()
    clean_b64 = image_b64.split(",")[1] if "," in image_b64 else image_b64
    img_data = base64.b64decode(clean_b64)
    pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
    
    formatted_prompt = prompt if "answer concisely" in prompt.lower() else f"Answer concisely in 1-5 words: {prompt}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": formatted_prompt}
            ]
        }
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    return {
        "answer": answer,
        "confidence": 0.92,
        "engine": "qwen2.5-vl-3b-specialist_a"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "specialist_a_vqa",
        "model": "qwen2.5-vl-3b-specialist_a",
        "port": 8001
    }


@app.post("/infer")
def infer(req: InferRequest):
    if not req.images or len(req.images) == 0:
        raise HTTPException(status_code=400, detail="At least one image is required for Specialist A.")

    result = run_native_vqa(req.images[0], req.query)
    
    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "evidence_image_base64": None,
        "specialist": "specialist_a",
        "task_type": req.task_type or "vqa",
        "execution_summary": {
            "model_used": result["engine"],
            "input_images_count": len(req.images),
            "specialist_role": "Optical/Multispectral VQA & Captioning"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)
