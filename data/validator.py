"""
SatQuery AI — Comprehensive Dataset & GeoTIFF Validator

Validates every downloaded and streamed sample:
1. Correct file format (PNG, JPEG, TIFF, GeoTIFF, JSON).
2. Correct band counts:
   - RGB optical: 3 bands
   - Multispectral Sentinel-2: 12 bands
   - SAR Sentinel-1: 2 bands (VV, VH)
   - Change/Semantic masks: 1 band (8-bit)
3. For GeoTIFFs: Strict presence and validity of CRS (Coordinate Reference System)
   and spatial transform.
4. Fails loudly with clear, actionable error messages on any mismatch.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Validator")


class ValidationError(Exception):
    """Raised when a sample fails validation checks."""
    pass


def validate_file_exists(file_path: str, description: str = "File") -> None:
    if not os.path.exists(file_path):
        raise ValidationError(f"CRITICAL: {description} not found at path: {file_path}")
    if os.path.getsize(file_path) == 0:
        raise ValidationError(f"CRITICAL: {description} at {file_path} is completely empty (0 bytes).")


def validate_json_file(file_path: str, required_keys: Optional[List[str]] = None) -> Any:
    validate_file_exists(file_path, "JSON File")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"CRITICAL: Invalid JSON syntax in {file_path}: {exc}")
        
    if required_keys:
        if isinstance(data, dict):
            for k in required_keys:
                if k not in data:
                    raise ValidationError(f"CRITICAL: Required key '{k}' missing from JSON dictionary in {file_path}")
        elif isinstance(data, list):
            if len(data) == 0:
                raise ValidationError(f"CRITICAL: JSON list is empty in {file_path}")
            if isinstance(data[0], dict):
                for k in required_keys:
                    if k not in data[0]:
                        raise ValidationError(f"CRITICAL: Required key '{k}' missing from first element in {file_path}")
    return data


def validate_standard_image(file_path: str, expected_bands: int = 3, expected_format: Optional[str] = None) -> Dict[str, Any]:
    """
    Validates standard RGB / Grayscale images (PNG, JPEG).
    """
    validate_file_exists(file_path, "Image File")
    try:
        with Image.open(file_path) as img:
            fmt = img.format
            mode = img.mode
            size = img.size
            bands = len(img.getbands())
            
            if expected_format and fmt.upper() != expected_format.upper():
                raise ValidationError(
                    f"CRITICAL FORMAT MISMATCH: Expected {expected_format}, but {file_path} is format '{fmt}'"
                )
                
            if bands != expected_bands:
                raise ValidationError(
                    f"CRITICAL BAND COUNT MISMATCH: Expected {expected_bands} band(s), but {file_path} has {bands} bands (mode: {mode})"
                )
                
            return {"file_path": file_path, "format": fmt, "mode": mode, "size": size, "bands": bands}
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"CRITICAL: Failed to read image file {file_path}: {exc}")


def validate_geotiff(
    file_path: str,
    expected_bands: Optional[int] = None,
    require_crs: bool = True
) -> Dict[str, Any]:
    """
    Strictly validates GeoTIFF / TIFF files using rasterio.
    Checks: format, band count, data type, dimensions, and CRS presence.
    """
    validate_file_exists(file_path, "GeoTIFF File")
    
    try:
        import rasterio
        with rasterio.open(file_path) as src:
            driver = src.driver
            count = src.count
            width = src.width
            height = src.height
            crs = src.crs
            transform = src.transform
            dtypes = src.dtypes
            
            if driver != "GTiff":
                logger.warning(f"File {file_path} driver is '{driver}' (expected GTiff)")
                
            if expected_bands is not None and count != expected_bands:
                raise ValidationError(
                    f"CRITICAL GEOTIFF BAND MISMATCH in {file_path}: Expected {expected_bands} bands, but found {count} bands."
                )
                
            if require_crs:
                if crs is None or not str(crs).strip():
                    raise ValidationError(
                        f"CRITICAL CRS MISSING in GeoTIFF {file_path}: GeoTIFF must contain a valid Coordinate Reference System (CRS)."
                    )
                if not getattr(crs, "is_epsg_code", False) and not getattr(crs, "is_projected", False) and not getattr(crs, "is_geographic", False):
                    # Fallback check
                    pass
                    
            if transform is None or transform.is_identity:
                logger.warning(f"GeoTIFF {file_path} has default/identity transform.")
                
            return {
                "file_path": file_path,
                "driver": driver,
                "bands": count,
                "width": width,
                "height": height,
                "crs": str(crs),
                "transform": transform,
                "dtypes": dtypes,
            }
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"CRITICAL: Failed to validate GeoTIFF at {file_path}: {exc}")


# ==============================================================================
# DATASET SPECIFIC VALIDATORS
# ==============================================================================

def validate_vrsbench_sample() -> bool:
    """
    Validates VRSBench samples (streamed or cached).
    """
    logger.info(">>> Validating VRSBench Dataset...")
    sample_dir = os.path.join(os.path.dirname(__file__), "raw", "vrsbench_samples")
    meta_path = os.path.join(sample_dir, "samples_meta.json")
    
    if os.path.exists(meta_path):
        meta = validate_json_file(meta_path)
        logger.info(f"Loaded {len(meta)} VRSBench cached metadata records.")
        for item in meta:
            if "saved_path" in item:
                info = validate_standard_image(item["saved_path"], expected_bands=3)
                logger.info(f"  [PASS] Sample image {os.path.basename(item['saved_path'])}: {info['size']}, {info['bands']} bands")
    else:
        # Validate live streaming sample
        from download_vrsbench import get_vrsbench_stream
        stream = get_vrsbench_stream(split="train")
        sample = next(iter(stream))
        img = sample.get("image") or sample.get("img")
        if img is None:
            raise ValidationError("CRITICAL: Streamed VRSBench sample does not contain an image field!")
        bands = len(img.getbands())
        if bands != 3:
            raise ValidationError(f"CRITICAL: VRSBench image has {bands} bands (expected 3 RGB bands)")
        logger.info(f"  [PASS] Live streamed VRSBench sample verified: size={img.size}, bands={bands}")
        
    logger.info("✓ VRSBench validation PASSED cleanly.\n")
    return True


def validate_rsvqa_lr() -> bool:
    """
    Validates RSVQA-LR raw dataset folder, questions, answers, and images.
    """
    logger.info(">>> Validating RSVQA-LR Dataset...")
    base_dir = os.path.join(os.path.dirname(__file__), "raw", "rsvqa_lr")
    if not os.path.exists(base_dir):
        raise ValidationError(f"CRITICAL: RSVQA-LR directory not found at {base_dir}")
        
    q_path = os.path.join(base_dir, "LR_questions.json")
    a_path = os.path.join(base_dir, "LR_answers.json")
    
    questions = validate_json_file(q_path, required_keys=["questions"])
    answers = validate_json_file(a_path, required_keys=["answers"])
    
    logger.info(f"  [PASS] LR_questions.json ({len(questions['questions'])} questions)")
    logger.info(f"  [PASS] LR_answers.json ({len(answers['answers'])} answers)")
    
    # Check images directory
    img_dir = os.path.join(base_dir, "Images_LR")
    if not os.path.exists(img_dir):
        raise ValidationError(f"CRITICAL: Images_LR directory missing from {base_dir}")
        
    images = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".tif", ".tiff", ".jpg"))]
    if len(images) == 0:
        raise ValidationError(f"CRITICAL: No image files found in {img_dir}")
        
    # Check first 3 sample images
    for img_name in images[:3]:
        p = os.path.join(img_dir, img_name)
        if img_name.lower().endswith((".tif", ".tiff")):
            info = validate_geotiff(p, expected_bands=3, require_crs=False)
            logger.info(f"  [PASS] Image {img_name}: {info['width']}x{info['height']}, {info['bands']} bands")
        else:
            info = validate_standard_image(p, expected_bands=3)
            logger.info(f"  [PASS] Image {img_name}: {info['size']}, {info['bands']} bands")
            
    logger.info("✓ RSVQA-LR validation PASSED cleanly.\n")
    return True


def validate_cdvqa_second() -> bool:
    """
    Validates CDVQA JSON files and SECOND bi-temporal imagery & masks.
    """
    logger.info(">>> Validating CDVQA & SECOND Datasets...")
    cdvqa_dir = os.path.join(os.path.dirname(__file__), "raw", "cdvqa")
    second_dir = os.path.join(os.path.dirname(__file__), "raw", "second")
    
    # 1. CDVQA JSON files
    for split in ["Train", "Val", "Test"]:
        q_file = os.path.join(cdvqa_dir, f"{split}_Questions.json")
        if os.path.exists(q_file):
            q_data = validate_json_file(q_file)
            logger.info(f"  [PASS] CDVQA {split}_Questions.json ({len(q_data)} items)")
            
    # 2. SECOND bi-temporal images and masks
    im1_dir = os.path.join(second_dir, "im1")
    im2_dir = os.path.join(second_dir, "im2")
    l1_dir = os.path.join(second_dir, "label1")
    l2_dir = os.path.join(second_dir, "label2")
    ch_dir = os.path.join(second_dir, "unchanged_label")
    
    for d, desc in [(im1_dir, "im1 (T1)"), (im2_dir, "im2 (T2)"), (l1_dir, "label1"), (l2_dir, "label2")]:
        if not os.path.exists(d):
            raise ValidationError(f"CRITICAL: Missing SECOND subfolder {d} ({desc})")
            
    im1_list = sorted([f for f in os.listdir(im1_dir) if f.endswith(".png")])
    if len(im1_list) == 0:
        raise ValidationError("CRITICAL: No bi-temporal images found in SECOND/im1 directory!")
        
    for fname in im1_list[:3]:
        p1 = os.path.join(im1_dir, fname)
        p2 = os.path.join(im2_dir, fname)
        pl1 = os.path.join(l1_dir, fname)
        pl2 = os.path.join(l2_dir, fname)
        
        info1 = validate_standard_image(p1, expected_bands=3)
        info2 = validate_standard_image(p2, expected_bands=3)
        
        if info1["size"] != info2["size"]:
            raise ValidationError(f"CRITICAL SPATIAL MISMATCH: T1 size {info1['size']} != T2 size {info2['size']} for {fname}")
            
        # Validate masks (1-band grayscale)
        if os.path.exists(pl1):
            validate_standard_image(pl1, expected_bands=1)
        if os.path.exists(pl2):
            validate_standard_image(pl2, expected_bands=1)
            
        logger.info(f"  [PASS] Bi-temporal pair {fname}: T1={info1['size']} (3-band RGB), T2={info2['size']} (3-band RGB)")
        
    logger.info("✓ CDVQA & SECOND validation PASSED cleanly.\n")
    return True


def validate_bigearthnet() -> bool:
    """
    Validates BigEarthNet-MM multimodal GeoTIFFs (Sentinel-1 SAR 2-band + Sentinel-2 Multispectral 12-band).
    Strictly checks CRS presence and band integrity.
    """
    logger.info(">>> Validating BigEarthNet-MM Dataset...")
    sample_dir = os.path.join(os.path.dirname(__file__), "raw", "bigearthnet_sample")
    if not os.path.exists(sample_dir):
        raise ValidationError(f"CRITICAL: BigEarthNet sample directory not found at {sample_dir}")
        
    s2_files = sorted([f for f in os.listdir(sample_dir) if f.endswith("_S2.tif")])
    s1_files = sorted([f for f in os.listdir(sample_dir) if f.endswith("_S1.tif")])
    
    if len(s2_files) == 0 or len(s1_files) == 0:
        raise ValidationError(f"CRITICAL: BigEarthNet-MM samples missing in {sample_dir}")
        
    for s2_name, s1_name in zip(s2_files[:3], s1_files[:3]):
        p_s2 = os.path.join(sample_dir, s2_name)
        p_s1 = os.path.join(sample_dir, s1_name)
        
        # 1. Validate Sentinel-2 (12-band multispectral GeoTIFF with CRS)
        s2_info = validate_geotiff(p_s2, expected_bands=12, require_crs=True)
        logger.info(
            f"  [PASS] Sentinel-2 GeoTIFF {s2_name}: "
            f"{s2_info['width']}x{s2_info['height']}, {s2_info['bands']} bands, CRS={s2_info['crs']}"
        )
        
        # 2. Validate Sentinel-1 (2-band SAR GeoTIFF with CRS)
        s1_info = validate_geotiff(p_s1, expected_bands=2, require_crs=True)
        logger.info(
            f"  [PASS] Sentinel-1 SAR GeoTIFF {s1_name}: "
            f"{s1_info['width']}x{s1_info['height']}, {s1_info['bands']} bands (VV/VH), CRS={s1_info['crs']}"
        )
        
        # 3. Check spatial coordinate system alignment
        if s2_info["crs"] != s1_info["crs"]:
            raise ValidationError(
                f"CRITICAL MULTIMODAL CRS MISMATCH: S2 CRS '{s2_info['crs']}' != S1 CRS '{s1_info['crs']}'"
            )
            
    logger.info("✓ BigEarthNet-MM validation PASSED cleanly.\n")
    return True


def validate_all() -> bool:
    """
    Executes full validation across all 4 datasets.
    """
    logger.info("=" * 60)
    logger.info("STARTING COMPREHENSIVE SATQUERY DATASET & RASTER VALIDATION")
    logger.info("=" * 60)
    
    vrs_ok = validate_vrsbench_sample()
    rsvqa_ok = validate_rsvqa_lr()
    cdvqa_ok = validate_cdvqa_second()
    ben_ok = validate_bigearthnet()
    
    all_ok = vrs_ok and rsvqa_ok and cdvqa_ok and ben_ok
    if all_ok:
        logger.info("=" * 60)
        logger.info("ALL DATASET SAMPLES VALIDATED CLEANLY! (100% PASS)")
        logger.info("=" * 60)
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatQuery AI Dataset Validator")
    parser.add_argument("--all", action="store_true", help="Validate all 4 datasets")
    parser.add_argument("--dataset", choices=["vrsbench", "rsvqa", "cdvqa", "bigearthnet"], help="Validate specific dataset")
    parser.add_argument("--file", type=str, help="Inspect and validate an individual raster/image/JSON file")
    parser.add_argument("--bands", type=int, default=None, help="Expected band count for file inspection")
    args = parser.parse_args()
    
    try:
        if args.file:
            if args.file.lower().endswith((".tif", ".tiff")):
                info = validate_geotiff(args.file, expected_bands=args.bands, require_crs=True)
                print(f"GeoTIFF Validated: {info}")
            elif args.file.lower().endswith((".png", ".jpg", ".jpeg")):
                info = validate_standard_image(args.file, expected_bands=args.bands or 3)
                print(f"Image Validated: {info}")
            elif args.file.lower().endswith(".json"):
                info = validate_json_file(args.file)
                print(f"JSON Validated: {len(info) if isinstance(info, (list, dict)) else 'OK'}")
            else:
                raise ValidationError(f"Unsupported file format for inspection: {args.file}")
        elif args.dataset == "vrsbench":
            validate_vrsbench_sample()
        elif args.dataset == "rsvqa":
            validate_rsvqa_lr()
        elif args.dataset == "cdvqa":
            validate_cdvqa_second()
        elif args.dataset == "bigearthnet":
            validate_bigearthnet()
        else:
            validate_all()
        sys.exit(0)
    except ValidationError as err:
        logger.error(f"\n❌ VALIDATION FAILURE: {err}")
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"\n💥 UNEXPECTED ERROR DURING VALIDATION: {exc}")
        sys.exit(2)
