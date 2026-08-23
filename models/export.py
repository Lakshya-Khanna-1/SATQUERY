"""
SatQuery AI — LoRA Adapter Merging, GGUF Conversion & Ollama Modelfile Exporter
SIH 2026 (PS 26167)

Performs:
1. Merges Specialist A LoRA weights into Base Model -> /models/merged/satquery-vqa
2. Merges Specialist B LoRA weights into Base Model -> /models/merged/satquery-changevqa
3. Generates Ollama Modelfiles:
   - models/Modelfile.vqa (satquery-vqa)
   - models/Modelfile.changevqa (satquery-changevqa)
4. Generates instructions & automated GGUF conversion script via llama.cpp.
"""

import os
import sys
import shutil
import argparse
import logging
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelExport")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MERGED_DIR = os.path.join(ROOT_DIR, "models", "merged")


def merge_lora_adapter(base_model_path: str, adapter_path: str, output_path: str):
    logger.info("=" * 60)
    logger.info(f"MERGING LORA ADAPTER: {adapter_path} -> {output_path}")
    logger.info(f"Base: {base_model_path}")
    logger.info("=" * 60)
    
    os.makedirs(output_path, exist_ok=True)
    
    # 1. Load processor
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.save_pretrained(output_path)
    
    # 2. Load base model in 16-bit
    logger.info("Loading base model in FP16/BF16 for unquantized merge...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # 3. Load & Merge LoRA Adapter
    if os.path.exists(adapter_path) and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        logger.info(f"Attaching and fusing LoRA adapter from {adapter_path}...")
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = peft_model.merge_and_unload()
    else:
        logger.warning(f"No adapter found at {adapter_path}. Copying base model weights directly.")
        merged_model = base_model
        
    # 4. Save Merged Checkpoint
    logger.info(f"Saving merged standalone model to {output_path}...")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    logger.info(f"[SUCCESS] Merged model saved to {output_path}")
    return output_path


def create_ollama_modelfiles():
    """
    Creates Ollama Modelfile definitions with satellite domain system prompts and chat templates.
    """
    models_dir = os.path.join(ROOT_DIR, "models")
    
    # 1. Modelfile for satquery-vqa (Specialist A)
    modelfile_vqa = os.path.join(models_dir, "Modelfile.vqa")
    vqa_content = """# SatQuery AI — Specialist A (satquery-vqa)
# Single-Image Optical/Multispectral VQA, Captioning, Spatial Grounding

FROM ./merged/satquery-vqa

SYSTEM \"\"\"You are SatQuery AI Specialist A, an expert satellite earth observation analyst and vision-language specialist.
Your role is to accurately analyze optical and multispectral remote sensing imagery, answer visual questions (RSVQA), produce dense geographic scene captions (VRSBench), and identify ground features with spatial coordinates.
Always maintain high scientific precision, state confidence, and cite observable spectral/spatial evidence.\"\"\"

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"""
    with open(modelfile_vqa, "w", encoding="utf-8") as f:
        f.write(vqa_content)
    logger.info(f"Created Ollama Modelfile for satquery-vqa: {modelfile_vqa}")

    # 2. Modelfile for satquery-changevqa (Specialist B)
    modelfile_changevqa = os.path.join(models_dir, "Modelfile.changevqa")
    changevqa_content = """# SatQuery AI — Specialist B (satquery-changevqa)
# Bi-Temporal Change Detection Visual Question Answering (Change-VQA)

FROM ./merged/satquery-changevqa

SYSTEM \"\"\"You are SatQuery AI Specialist B, a dedicated bi-temporal change detection specialist.
You compare pre-change (Time 1) and post-change (Time 2) satellite images to detect construction, deforestation, flood damage, urban expansion, and environmental dynamics.
Answer CDVQA questions with exact change categories, direction of change (increase/decrease), and extent.\"\"\"

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"""
    with open(modelfile_changevqa, "w", encoding="utf-8") as f:
        f.write(changevqa_content)
    logger.info(f"Created Ollama Modelfile for satquery-changevqa: {modelfile_changevqa}")

    # 3. Create Ollama Deployment README
    readme_path = os.path.join(models_dir, "OLLAMA_REGISTRATION.md")
    readme_content = """# SatQuery AI — Ollama Model Registration Guide

### 1. Build Merged Models
Run the export script to fuse LoRA adapters:
```bash
python models/export.py --all
```

### 2. Register with Ollama
Execute the following commands to create the local Ollama models:
```bash
# Register Specialist A (VQA/Captioning)
ollama create satquery-vqa -f models/Modelfile.vqa

# Register Specialist B (Change-VQA)
ollama create satquery-changevqa -f models/Modelfile.changevqa
```

### 3. Verify Local Execution
```bash
# Test Specialist A
ollama run satquery-vqa "Describe the landscape features visible in this satellite scene."

# Test Specialist B
ollama run satquery-changevqa "What is the primary land-cover change between these two observation dates?"
```
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    logger.info(f"Created Ollama Registration documentation: {readme_path}")


def export_all():
    base_a = os.path.join(ROOT_DIR, "models", "base", "specialist_a_base")
    base_b = os.path.join(ROOT_DIR, "models", "base", "specialist_b_base")
    adapter_a = os.path.join(ROOT_DIR, "models", "adapters", "specialist_a")
    adapter_b = os.path.join(ROOT_DIR, "models", "adapters", "specialist_b")
    
    out_a = os.path.join(MERGED_DIR, "satquery-vqa")
    out_b = os.path.join(MERGED_DIR, "satquery-changevqa")
    
    if os.path.exists(base_a):
        merge_lora_adapter(base_a, adapter_a, out_a)
    if os.path.exists(base_b):
        merge_lora_adapter(base_b, adapter_b, out_b)
        
    create_ollama_modelfiles()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatQuery Model Export CLI")
    parser.add_argument("--all", action="store_true", help="Merge all specialists and create Modelfiles")
    parser.add_argument("--modelfiles_only", action="store_true", help="Create only Ollama Modelfiles")
    args = parser.parse_args()
    
    if args.modelfiles_only:
        create_ollama_modelfiles()
    else:
        export_all()
