"""
SatQuery AI — Grounded-SAM 2 Remote-Sensing Feature Segmenter
SIH 2026 (PS 26167)

High-precision remote sensing instance segmentation pipeline:
1. Local Satellite Object Grounding (YOLOv8-OBB trained on DOTA aerial dataset):
   - Fast, local CUDA execution (~15ms)
   - Zero external network dependency, immune to Gemini 429 rate limits
   - Detects planes, helicopters, large vehicles (buses, trucks), small vehicles (cars),
     ships, harbors, storage tanks, bridges, sports fields, and swimming pools.
2. Zero-Shot Vision-Language Grounding (VLM Gemini / Qwen):
   - Handles open-vocabulary targets and complex natural landcover features.
3. Meta SAM 2.1 Hiera (Segment Anything 2):
   - Converts prompt bounding boxes into sub-pixel polygon masks with clean neon cyan rendering.
"""

import os
import io
import json
import base64
import logging
import urllib.request
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger("GroundedSAM2")

_SAM2_MODEL = None
_SAM2_DEVICE = None
_OBB_MODEL = None


def get_gemini_api_keys() -> List[str]:
    """Retrieves available Gemini API keys from environment or .env file."""
    keys = []
    for var in ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_1"]:
        k = os.environ.get(var)
        if k and k.strip() and not k.startswith("YOUR_"):
            keys.append(k.strip())
    
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY=") or line.startswith("GEMINI_API_KEY_2=") or line.startswith("GEMINI_API_KEY_1="):
                        val = line.split("=", 1)[1].strip().strip('"\'')
                        if val and not val.startswith("YOUR_") and val not in keys:
                            keys.append(val)
        except Exception as e:
            logger.warning(f"Error reading .env for grounding keys: {e}")
    return keys


def get_sam2_model():
    """Lazy-loads and caches Meta SAM 2.1 model instance."""
    global _SAM2_MODEL, _SAM2_DEVICE
    if _SAM2_MODEL is not None:
        return _SAM2_MODEL

    try:
        import torch
        from ultralytics import SAM
        
        _SAM2_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        weights_candidates = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "sam", "sam2.1_t.pt"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sam2.1_t.pt"),
            "sam2.1_t.pt"
        ]
        chosen_weights = "sam2.1_t.pt"
        for candidate in weights_candidates:
            if os.path.exists(candidate):
                chosen_weights = candidate
                break

        logger.info(f"Loading Meta SAM 2.1 Hiera ({chosen_weights}) on {_SAM2_DEVICE}...")
        _SAM2_MODEL = SAM(chosen_weights)
        logger.info("Meta SAM 2.1 Hiera successfully initialized and ready.")
        return _SAM2_MODEL
    except Exception as e:
        logger.error(f"Failed to load SAM 2 model: {e}", exc_info=True)
        return None


def get_obb_model():
    """Lazy-loads local YOLOv8-OBB model trained on DOTA aerial remote sensing dataset."""
    global _OBB_MODEL
    if _OBB_MODEL is not None:
        return _OBB_MODEL

    try:
        from ultralytics import YOLO
        
        candidates = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "sam", "yolov8n-obb.pt"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "yolov8n-obb.pt"),
            "yolov8n-obb.pt"
        ]
        chosen = "yolov8n-obb.pt"
        for c in candidates:
            if os.path.exists(c):
                chosen = c
                break

        logger.info(f"Loading local satellite OBB detector ({chosen})...")
        _OBB_MODEL = YOLO(chosen)
        logger.info("Local satellite OBB detector initialized.")
        return _OBB_MODEL
    except Exception as e:
        logger.warning(f"Could not load local satellite OBB detector: {e}")
        return None


def safe_b64_decode(image_b64: str) -> bytes:
    """Safely decodes base64 string handling padding or header tags."""
    if not image_b64 or not isinstance(image_b64, str):
        raise ValueError("Image payload is empty or not a string.")
    clean_b64 = image_b64.split(",")[1] if "," in image_b64 else image_b64
    clean_b64 = "".join(clean_b64.split())
    rem = len(clean_b64) % 4
    if rem == 1:
        clean_b64 = clean_b64[:-1]
    elif rem > 1:
        clean_b64 += "=" * (4 - rem)
    try:
        raw_bytes = base64.b64decode(clean_b64)
        with Image.open(io.BytesIO(raw_bytes)) as test_img:
            test_img.verify()
        return raw_bytes
    except Exception as e:
        logger.error(f"Invalid base64 satellite image payload: {e}")
        raise ValueError(f"Invalid base64 satellite image data: {e}")


def determine_feature_category(query: str) -> Tuple[str, str]:
    """Extracts target feature label and grounding prompt target from user query."""
    q_lower = query.lower()

    if any(k in q_lower for k in ["aircraft", "airplane", "plane", "jet", "aviation"]):
        return "Aircraft & Aviation Assets", "airplanes, aircraft, or jets"
    elif any(k in q_lower for k in ["vehicle", "car", "truck", "automobile", "suv", "van", "parking", "transport"]):
        return "Vehicles & Transport Objects", "vehicles, cars, trucks, vans, airplanes, or transport assets"
    elif any(k in q_lower for k in ["ship", "boat", "vessel", "maritime"]):
        return "Maritime Vessels & Ships", "ships, boats, or marine vessels"
    elif any(k in q_lower for k in ["building", "structure", "house", "roof", "facility", "built-up", "urban"]):
        return "Built-Up Structures & Buildings", "buildings, built-up structures, architectural facilities, or houses"
    elif any(k in q_lower for k in ["water", "river", "lake", "ocean", "sea", "pond", "canal", "reservoir", "hydrology"]):
        return "Water Bodies & Surface Hydrology", "lakes, rivers, ponds, reservoirs, or water bodies"
    elif any(k in q_lower for k in ["vegetation", "forest", "tree", "crop", "agriculture", "plant", "canopy", "farmland"]):
        return "Vegetation Canopy & Agricultural Cover", "vegetation, trees, forests, agriculture, or green canopy"
    elif any(k in q_lower for k in ["runway", "airstrip", "taxiway"]):
        return "Aviation Runways & Tarmac", "airport runways, taxiways, or tarmac"
    elif any(k in q_lower for k in ["road", "highway", "street"]):
        return "Road Transport Corridors", "roads, highways, or paved transportation corridors"
    elif any(k in q_lower for k in ["tank", "oil", "fuel", "gas"]):
        return "Industrial Storage Tanks", "cylindrical storage tanks or fuel reservoirs"
    elif any(k in q_lower for k in ["pool", "swimming"]):
        return "Swimming Pools", "swimming pools or recreational water basins"
    else:
        words = [w for w in q_lower.replace("?", "").replace(".", "").split() if w not in [
            "what", "where", "is", "are", "there", "show", "me", "highlight", "segment", "locate", "the", "in", "this", "image", "satellite"
        ]]
        target = " ".join(words) if words else "target features"
        return f"Identified {target.title()}", target


def ground_with_local_obb(pil_img: Image.Image, query: str, feature_label: str) -> Tuple[List[List[int]], str, int]:
    """
    Executes local YOLOv8-OBB aerial remote sensing model on CUDA.
    Returns: (pixel_boxes, dynamic_label, count)
    """
    detector = get_obb_model()
    if detector is None:
        return [], feature_label, 0

    q_lower = query.lower()
    w, h = pil_img.size

    # Class mappings for DOTA remote sensing classes
    if any(k in q_lower for k in ["aircraft", "airplane", "plane", "jet", "aviation"]):
        target_dota = ["plane", "helicopter"]
        primary_name = "aircraft"
    elif any(k in q_lower for k in ["ship", "boat", "vessel"]):
        target_dota = ["ship", "harbor"]
        primary_name = "vessels"
    elif any(k in q_lower for k in ["tank", "storage"]):
        target_dota = ["storage tank"]
        primary_name = "storage tanks"
    elif any(k in q_lower for k in ["bridge"]):
        target_dota = ["bridge"]
        primary_name = "bridges"
    elif any(k in q_lower for k in ["pool", "swimming"]):
        target_dota = ["swimming pool"]
        primary_name = "swimming pools"
    elif any(k in q_lower for k in ["car", "automobile", "parking"]):
        target_dota = ["small vehicle", "large vehicle"]
        primary_name = "vehicles"
    elif any(k in q_lower for k in ["truck", "bus", "van"]):
        target_dota = ["large vehicle"]
        primary_name = "transport vehicles"
    elif any(k in q_lower for k in ["vehicle", "transport"]):
        # Broad transport: includes both ground vehicles and planes/helicopters
        target_dota = ["large vehicle", "small vehicle", "plane", "helicopter"]
        primary_name = "vehicles & aviation assets"
    else:
        target_dota = []
        primary_name = feature_label

    if not target_dota:
        return [], feature_label, 0

    try:
        res = detector(pil_img, conf=0.15, verbose=False)
        obb_detections = res[0].obb if (res and len(res) > 0 and hasattr(res[0], "obb")) else None
        
        if obb_detections is None or len(obb_detections) == 0:
            return [], feature_label, 0

        boxes = []
        breakdown = {}
        for b in obb_detections:
            cls_id = int(b.cls[0].item())
            cls_name = detector.names[cls_id]
            if cls_name in target_dota:
                xyxy = [int(v) for v in b.xyxy[0].tolist()]
                # Clamp coordinates to image boundaries
                x1 = max(0, min(w - 1, xyxy[0]))
                y1 = max(0, min(h - 1, xyxy[1]))
                x2 = max(x1 + 2, min(w, xyxy[2]))
                y2 = max(y1 + 2, min(h, xyxy[3]))
                boxes.append([x1, y1, x2, y2])
                breakdown[cls_name] = breakdown.get(cls_name, 0) + 1

        if not boxes:
            return [], feature_label, 0

        # Construct detailed breakdown string
        parts = []
        for k, count in breakdown.items():
            name = "aircraft" if k == "plane" else ("transport vehicle" if k == "large vehicle" else ("car" if k == "small vehicle" else k))
            parts.append(f"{count} {name}" if (count == 1 or name == "aircraft") else f"{count} {name}s")
        
        breakdown_str = f" ({', '.join(parts)})" if len(parts) > 1 else ""
        dynamic_label = f"{primary_name}{breakdown_str}"
        logger.info(f"Local Satellite OBB detected {len(boxes)} objects: {breakdown}")
        return boxes, dynamic_label, len(boxes)

    except Exception as e:
        logger.warning(f"Local satellite OBB inference error: {e}")
        return [], feature_label, 0


def ground_objects_with_vlm(pil_img: Image.Image, target_desc: str) -> List[List[int]]:
    """
    Uses zero-shot Vision-Language Model grounding to detect 2D bounding boxes.
    Returns pixel coordinates: [[x1, y1, x2, y2], ...]
    """
    w, h = pil_img.size
    keys = get_gemini_api_keys()

    if not keys:
        return []

    grounding_img = pil_img.copy()
    if max(w, h) > 512:
        grounding_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
    
    buf = io.BytesIO()
    grounding_img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = f"""Detect and locate all instances of {target_desc} in this satellite/aerial image.
Do not highlight empty background, pavement, or unrelated objects.
Return ONLY a valid JSON list of bounding boxes with normalized coordinates [0, 1000]:
[
  {{"box_2d": [ymin, xmin, ymax, xmax], "label": "{target_desc}"}}
]
where ymin is top (0-1000), xmin is left (0-1000), ymax is bottom (0-1000), xmax is right (0-1000)."""

    payload = {
        "contents": [{
            "parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                {"text": prompt}
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json"
        }
    }

    boxes_2d = []
    for key_idx, k in enumerate(keys[:2]):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={k}"
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as res:
                res_data = json.loads(res.read().decode("utf-8"))
                candidates = res_data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts and "text" in parts[0]:
                        raw_text = parts[0]["text"].strip()
                        boxes_2d = json.loads(raw_text)
                        logger.info(f"VLM Grounding succeeded via Key #{key_idx+1}: Found {len(boxes_2d)} boxes.")
                        break
        except Exception as e:
            logger.warning(f"VLM Grounding key #{key_idx+1} notice: {e}")
        if boxes_2d:
            break

    pixel_boxes = []
    for b in boxes_2d:
        ymin, xmin, ymax, xmax = b.get("box_2d", [0, 0, 0, 0])
        x1 = max(0, int(xmin / 1000.0 * w))
        y1 = max(0, int(ymin / 1000.0 * h))
        x2 = min(w, int(xmax / 1000.0 * w))
        y2 = min(h, int(ymax / 1000.0 * h))
        if x2 > x1 and y2 > y1:
            pixel_boxes.append([x1, y1, x2, y2])

    return pixel_boxes


def segment_target_feature(image_b64: str, query: str) -> Tuple[Optional[str], float, str, int]:
    """
    Executes Grounded-SAM 2 on the satellite image:
    1. Visual Grounding:
       - Primary: Local YOLOv8-OBB aerial remote sensing model (~15ms on GPU, offline).
       - Secondary: Zero-shot VLM grounding (Gemini) for open-vocabulary targets.
    2. Meta SAM 2.1 Hiera: Produces sub-pixel instance segmentation masks.
    3. Renders publication-grade cyan contour overlay (#06b6d4 fill, #00f0ff border).
    
    Returns:
        (highlighted_b64_image, coverage_percentage, target_feature_label, detected_count)
    """
    try:
        img_bytes = safe_b64_decode(image_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(pil_img)
        h, w, _ = img_np.shape
        total_px = h * w

        q_lower = query.lower()

        # Check for explicit grounding/segmentation intent
        explicit_grounding_intent = any(k in q_lower for k in [
            "highlight", "segment", "mask", "locate", "where", "find", "detect", "ground",
            "bounding", "show", "point out", "water", "lake", "river", "building", "structure",
            "vehicle", "car", "truck", "aircraft", "airplane", "plane", "jet", "ship",
            "vegetation", "forest", "crop", "road", "how many", "count"
        ])

        is_general_overview = any(k in q_lower for k in [
            "describe the scene", "give an overview", "caption this", "summary of the area",
            "general description", "what features are here"
        ]) and not any(k in q_lower for k in ["highlight", "segment", "locate", "where", "show", "find"])

        if is_general_overview or not explicit_grounding_intent:
            logger.info("General overview query: Skipping feature segmentation overlay.")
            return None, 0.0, "Scene Overview", 0

        feature_label, target_desc = determine_feature_category(query)
        logger.info(f"Initiating Grounded-SAM 2 for category: '{feature_label}'...")

        # Stage 1: Visual Grounding
        # 1a. Try local aerial satellite detector first (15ms, zero network dependency)
        pixel_boxes, dynamic_label, detected_count = ground_with_local_obb(pil_img, query, feature_label)
        
        # 1b. If local aerial detector returned no boxes or target is open-vocabulary, try VLM grounding
        if not pixel_boxes:
            vlm_boxes = ground_objects_with_vlm(pil_img, target_desc)
            if vlm_boxes:
                pixel_boxes = vlm_boxes
                detected_count = len(pixel_boxes)
                dynamic_label = feature_label

        combined_mask = np.zeros((h, w), dtype=np.uint8)
        sam = get_sam2_model()

        if sam is not None and pixel_boxes:
            # Stage 2: Meta SAM 2.1 Instance Segmentation
            try:
                results = sam(img_np, bboxes=pixel_boxes, verbose=False)
                for r in results:
                    if r.masks is not None:
                        for mask_tensor in r.masks.data:
                            m_np = mask_tensor.cpu().numpy().astype(np.uint8)
                            if m_np.shape != (h, w):
                                m_np = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST)
                            combined_mask = np.maximum(combined_mask, m_np)
            except Exception as e:
                logger.error(f"SAM 2 prediction error: {e}. Falling back to bounding box masks.")
                for bx in pixel_boxes:
                    combined_mask[bx[1]:bx[3], bx[0]:bx[2]] = 255

        elif pixel_boxes:
            for bx in pixel_boxes:
                combined_mask[bx[1]:bx[3], bx[0]:bx[2]] = 255

        else:
            # Broad land-cover spectral fallback for water/vegetation if no boxes found
            if "water" in q_lower or "lake" in q_lower or "river" in q_lower:
                r_ch, g_ch, b_ch = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
                hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                water_cond = (b_ch > r_ch + 15) & (b_ch >= g_ch - 2) & (hsv[:, :, 1] > 30) & (hsv[:, :, 2] < 120)
                combined_mask[water_cond] = 255
            elif "vegetation" in q_lower or "forest" in q_lower or "tree" in q_lower:
                r_ch, g_ch, b_ch = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
                veg_cond = (g_ch > r_ch + 10) & (g_ch > b_ch + 8)
                combined_mask[veg_cond] = 255

        # Clean noise (< 0.05% of image area)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        matching_pixels = int(np.sum(combined_mask > 0))
        coverage_pct = round(float(matching_pixels / total_px) * 100.0, 1)

        # If zero objects/regions were found, return clean None (zero false hallucinations!)
        if coverage_pct < 0.1 and detected_count == 0:
            logger.info(f"Target feature '{feature_label}' not present in image scene (coverage < 0.1%).")
            return None, 0.0, f"No {feature_label}", 0

        # Stage 3: Publication-Grade Rendering (45% Cyan Fill + 2px Neon Cyan Contour)
        overlay_rgb = img_np.copy()
        cyan_rgb = np.array([6, 182, 212], dtype=np.float32) # Hex: #06b6d4
        mask_bool = combined_mask > 0

        # Alpha composite masked region
        overlay_rgb[mask_bool] = (
            overlay_rgb[mask_bool].astype(np.float32) * 0.55 + cyan_rgb * 0.45
        ).astype(np.uint8)

        # Sharp Neon Cyan Contour Boundary (#00f0ff)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_rgb, contours, -1, (0, 240, 255), 2, cv2.LINE_AA)

        # Convert back to optimized JPEG base64
        res_pil = Image.fromarray(overlay_rgb)
        buf = io.BytesIO()
        res_pil.save(buf, format="JPEG", quality=92)
        highlighted_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        logger.info(f"Grounded-SAM 2 complete: {detected_count} objects ({dynamic_label}), {coverage_pct}% coverage.")
        return highlighted_b64, coverage_pct, dynamic_label, detected_count

    except Exception as e:
        logger.error(f"Grounded-SAM 2 pipeline failure: {e}", exc_info=True)
        return None, 0.0, "Error during feature segmentation", 0
