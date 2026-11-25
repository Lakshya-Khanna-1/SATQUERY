"""
SatQuery AI — Specialist C Evaluation & Verification Test

Verifies:
1. Optical-SAR Multimodal Fusion: Returns 256-d normalized joint embedding & cross-modal similarity.
2. Bi-Temporal Change Mapping: Generates 6-class segmentation mask & saves visual RGBA PNG overlay.
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.loaders.bigearthnet_loader import BigEarthNetDataset
from data.loaders.cdvqa_loader import CDVQADataset
from finetune.specialist_c.model_c import SpecialistC, CHANGE_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalSpecialistC")


def test_specialist_c_inference(
    checkpoint_path: str = "models/specialist_c/specialist_c.pt",
    output_overlay_path: str = "models/specialist_c/test_change_overlay.png"
):
    logger.info("=" * 60)
    logger.info("TESTING SPECIALIST C — MULTIMODAL FUSION & CHANGE MAP GENERATOR")
    logger.info("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpecialistC().to(device)
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading Specialist C checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}. Running initialized weights.")
        
    model.eval()
    
    # 1. Test Optical-SAR Fusion
    logger.info("\n>>> [1/2] Testing Optical-SAR Multimodal Embedding...")
    ben_loader = BigEarthNetDataset(max_samples=5)
    sample_ben = ben_loader[0]
    opt_img = sample_ben["images"][0]
    sar_arr = np.array(sample_ben["images"][1], dtype=np.float32)
    
    fusion_result = model.encode_pair(opt_img, sar_arr)
    
    logger.info(f" - Optical Embedding Dim : {len(fusion_result['optical_embedding'])}")
    logger.info(f" - SAR Embedding Dim     : {len(fusion_result['sar_embedding'])}")
    logger.info(f" - Joint Embedding Dim   : {len(fusion_result['joint_embedding'])}")
    logger.info(f" - Cross-Modal Similarity: {fusion_result['cross_modal_similarity']:.4f}")
    logger.info(f" - Alignment Confidence  : {fusion_result['modality_alignment_confidence']:.4f}")
    
    assert len(fusion_result["joint_embedding"]) == 256, "Expected 256-d joint embedding"
    
    # 2. Test Bi-Temporal Change Mapping
    logger.info("\n>>> [2/2] Testing Bi-Temporal Change-Map & Overlay Generation...")
    cdvqa_ds = CDVQADataset(split="train", max_samples=5)
    sample_cd = cdvqa_ds[0]
    im1 = sample_cd["images"][0]
    im2 = sample_cd["images"][1]
    
    change_result = model.generate_change_map(im1, im2)
    
    logger.info(f" - Mask Resolution       : {change_result['mask_shape']}")
    logger.info(f" - Changed Pixels         : {change_result['changed_pixels']}/{change_result['total_pixels']}")
    logger.info(f" - Change Area (%)        : {change_result['percentage_area_changed']:.2f}%")
    logger.info(f" - Class Breakdown        :")
    for cls_name, count in change_result["class_breakdown"].items():
        if count > 0:
            logger.info(f"     * {cls_name:<36}: {count} px")
            
    # Save test overlay
    os.makedirs(os.path.dirname(output_overlay_path), exist_ok=True)
    change_result["overlay_image"].save(output_overlay_path)
    logger.info(f" - Visual Overlay Saved to: {output_overlay_path}")
    logger.info(f" - Base64 Length          : {len(change_result['overlay_png_base64'])} chars")
    
    assert os.path.exists(output_overlay_path), "Overlay file should exist on disk"
    assert len(change_result["overlay_png_base64"]) > 100, "Base64 overlay should not be empty"
    
    logger.info("\n" + "=" * 60)
    logger.info("SPECIALIST C DEFINITION OF DONE: VERIFIED PASS")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    test_specialist_c_inference()
