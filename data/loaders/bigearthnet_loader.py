"""
SatQuery AI — BigEarthNet-MM PyTorch Dataset Loader
SIH 2026 (PS 26167)

Specialist C dataset loader for Optical–SAR Multimodal Fusion & Change Mapping.
Pairs Sentinel-2 (Multispectral/Optical) with Sentinel-1 (SAR) GeoTIFFs,
verifying spatial co-registration via pairing.py.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .common import SatQuerySample, ensure_pil_image
try:
    from ..pairing import align_image_pair
except (ImportError, ValueError):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from pairing import align_image_pair

logger = logging.getLogger("BigEarthNetLoader")

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw", "bigearthnet_sample")


class BigEarthNetDataset(Dataset):
    """
    PyTorch Dataset for BigEarthNet-MM Optical + SAR Multimodal Pairs (Specialist C).
    """
    def __init__(
        self,
        max_samples: Optional[int] = None,
        data_dir: str = SAMPLE_DIR,
        return_pil: bool = True,
        transform: Optional[Any] = None
    ):
        self.data_dir = data_dir
        self.return_pil = return_pil
        self.transform = transform
        self.items: List[Dict[str, Any]] = []
        
        self._load_data(max_samples)
        logger.info(f"Initialized BigEarthNetDataset with {len(self.items)} optical-SAR pairs.")

    def _load_data(self, max_samples: Optional[int]):
        meta_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                s2_p = rec.get("s2_path")
                s1_p = rec.get("s1_path")
                if s2_p and s1_p and os.path.exists(s2_p) and os.path.exists(s1_p):
                    self.items.append({
                        "patch_id": rec.get("patch_id"),
                        "s2_path": s2_p,
                        "s1_path": s1_p,
                        "labels": rec.get("labels", ["Vegetation", "Built-up"]),
                        "crs": rec.get("crs", "EPSG:32632"),
                    })
        else:
            # Discover from folder
            if os.path.exists(self.data_dir):
                s2_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith("_S2.tif")])
                for s2_name in s2_files:
                    prefix = s2_name.replace("_S2.tif", "")
                    s1_name = f"{prefix}_S1.tif"
                    s2_p = os.path.join(self.data_dir, s2_name)
                    s1_p = os.path.join(self.data_dir, s1_name)
                    if os.path.exists(s1_p):
                        self.items.append({
                            "patch_id": prefix,
                            "s2_path": s2_p,
                            "s1_path": s1_p,
                            "labels": ["Forest", "Water", "Arable land"],
                            "crs": "EPSG:32632",
                        })
                        
        if max_samples:
            self.items = self.items[:max_samples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> SatQuerySample:
        item = self.items[idx]
        s2_path = item["s2_path"]
        s1_path = item["s1_path"]
        
        # Verify and align spatial co-registration between Optical (S2) and SAR (S1)
        data_s2, data_s1_aligned, was_resampled, align_info = align_image_pair(
            s2_path, s1_path, tolerance_meters=1.0
        )
        
        # Convert to appropriate format for return
        if self.return_pil:
            # Convert S2 multispectral (first 3 channels) to RGB PIL Image
            s2_img = ensure_pil_image(data_s2)
            # Convert SAR (VV / VH) to False Color representation (VV, VH, VV/VH ratio)
            sar_vv = data_s1_aligned[0]
            sar_vh = data_s1_aligned[1] if data_s1_aligned.shape[0] > 1 else data_s1_aligned[0]
            
            # Normalize SAR float dB (-25 to 0) to 0..255
            vv_norm = np.clip((sar_vv + 25.0) / 25.0 * 255.0, 0, 255).astype(np.uint8)
            vh_norm = np.clip((sar_vh + 30.0) / 30.0 * 255.0, 0, 255).astype(np.uint8)
            ratio = np.clip((vv_norm.astype(float) / (vh_norm.astype(float) + 1e-3)) * 64.0, 0, 255).astype(np.uint8)
            
            sar_rgb = np.stack([vv_norm, vh_norm, ratio], axis=-1)
            s1_img = Image.fromarray(sar_rgb)
            images_out = [s2_img, s1_img]
        else:
            # Return raw float tensors
            t_s2 = torch.from_numpy(data_s2.astype(np.float32))
            t_s1 = torch.from_numpy(data_s1_aligned.astype(np.float32))
            images_out = [t_s2, t_s1]
            
        labels_str = ", ".join(item["labels"])
        query_text = "Use the optical and SAR images together to identify built-up and water-covered regions."
        answer_text = f"Identified land cover categories: {labels_str}."
        
        return {
            "images": images_out, # [Optical, SAR] pair
            "query": query_text,
            "answer": answer_text,
            "task_type": "fusion",
            "metadata": {
                "source": "BigEarthNet-MM",
                "patch_id": item["patch_id"],
                "s2_bands": data_s2.shape[0],
                "s1_bands": data_s1_aligned.shape[0],
                "crs": item["crs"],
                "was_resampled": was_resampled,
                "labels": item["labels"],
            }
        }
