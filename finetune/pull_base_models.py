"""
SatQuery AI — Base Model Acquisition & Dual Staging Script

Downloads Qwen/Qwen2.5-VL-3B-Instruct from Hugging Face into two separate local
staging paths:
1. /models/base/specialist_a_base (for Specialist A: VQA/Caption/Grounding)
2. /models/base/specialist_b_base (for Specialist B: Change-VQA)

Ensures both specialists maintain independent local base model weights and configs.
"""

import os
import sys
import shutil
import logging
import argparse
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PullBaseModels")

BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_BASE_DIR = os.path.join(ROOT_DIR, "models", "base")
DIR_SPECIALIST_A = os.path.join(MODELS_BASE_DIR, "specialist_a_base")
DIR_SPECIALIST_B = os.path.join(MODELS_BASE_DIR, "specialist_b_base")


def check_model_complete(model_dir: str) -> bool:
    if not os.path.exists(model_dir):
        return False
    required_files = ["config.json", "tokenizer.json"]
    has_config = all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)
    # Check for weights (.safetensors or .bin)
    files = os.listdir(model_dir)
    has_weights = any(f.endswith((".safetensors", ".bin")) for f in files)
    return has_config and has_weights


def download_base_model(dest_dir: str, name: str, force: bool = False) -> str:
    """
    Downloads the full snapshot of Qwen/Qwen2.5-VL-3B-Instruct into dest_dir.
    """
    os.makedirs(dest_dir, exist_ok=True)
    if check_model_complete(dest_dir) and not force:
        logger.info(f"Base model for {name} already complete at {dest_dir}. Skipping download.")
        return dest_dir
        
    logger.info(f"Downloading {BASE_MODEL_ID} from Hugging Face for {name} into {dest_dir}...")
    local_path = snapshot_download(
        repo_id=BASE_MODEL_ID,
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )
    logger.info(f"Successfully staged {name} base model at {local_path}")
    return local_path


def copy_or_download_second_instance(src_dir: str, dest_dir: str, force: bool = False):
    """
    Copies the staged base model to create an isolated independent copy for Specialist B.
    """
    os.makedirs(dest_dir, exist_ok=True)
    if check_model_complete(dest_dir) and not force:
        logger.info(f"Base model for Specialist B already complete at {dest_dir}. Skipping copy.")
        return dest_dir
        
    logger.info(f"Cloning Specialist A base weights into Specialist B staging directory {dest_dir}...")
    for item in os.listdir(src_dir):
        src_item = os.path.join(src_dir, item)
        dst_item = os.path.join(dest_dir, item)
        if os.path.isdir(src_item):
            shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
        else:
            if not os.path.exists(dst_item) or force:
                shutil.copy2(src_item, dst_item)
                
    logger.info(f"Successfully cloned independent base model copy into {dest_dir}")
    return dest_dir


def pull_all_base_models(force: bool = False):
    os.makedirs(MODELS_BASE_DIR, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("SATQUERY AI — BASE MODEL ACQUISITION & DUAL STAGING")
    logger.info(f"Base Architecture: {BASE_MODEL_ID} (~3 Billion parameters)")
    logger.info("=" * 60)
    
    # 1. Download Specialist A Base Model
    logger.info("\n>>> [1/2] Staging Base Model for Specialist A (VQA/Caption/Grounding)...")
    path_a = download_base_model(DIR_SPECIALIST_A, "Specialist A", force=force)
    
    # 2. Stage Specialist B Base Model (Isolated Clone)
    logger.info("\n>>> [2/2] Staging Base Model for Specialist B (Change-VQA)...")
    path_b = copy_or_download_second_instance(path_a, DIR_SPECIALIST_B, force=force)
    
    # Verification
    ok_a = check_model_complete(path_a)
    ok_b = check_model_complete(path_b)
    
    logger.info("\n" + "=" * 60)
    logger.info("BASE MODEL STAGING SUMMARY:")
    logger.info(f" - Specialist A Base: {'READY' if ok_a else 'FAILED'} -> {path_a}")
    logger.info(f" - Specialist B Base: {'READY' if ok_b else 'FAILED'} -> {path_b}")
    logger.info("=" * 60)
    
    return ok_a and ok_b


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatQuery AI Base Model Pull CLI")
    parser.add_argument("--force", action="store_true", help="Force re-download / overwrite")
    args = parser.parse_args()
    
    success = pull_all_base_models(force=args.force)
    sys.exit(0 if success else 1)
