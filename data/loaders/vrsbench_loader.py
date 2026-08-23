"""
SatQuery AI — VRSBench PyTorch Dataset Loader
SIH 2026 (PS 26167)

Specialist A dataset loader for single-image VQA, Captioning, and
Region Grounding using VRSBench.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Literal
from PIL import Image
import torch
from torch.utils.data import Dataset
from .common import SatQuerySample, TaskType, ensure_pil_image

logger = logging.getLogger("VRSBenchLoader")

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw", "vrsbench_samples")


class VRSBenchDataset(Dataset):
    """
    PyTorch Dataset for VRSBench supporting single-image VQA, Captioning, and Grounding.
    """
    def __init__(
        self,
        task_filter: Optional[Literal["vqa", "caption", "grounding"]] = None,
        max_samples: Optional[int] = None,
        sample_dir: str = SAMPLE_DIR,
        transform: Optional[Any] = None
    ):
        self.task_filter = task_filter
        self.transform = transform
        self.sample_dir = sample_dir
        self.items: List[Dict[str, Any]] = []
        
        self._load_data(max_samples)
        logger.info(f"Initialized VRSBenchDataset with {len(self.items)} items (filter={task_filter})")

    def _load_data(self, max_samples: Optional[int]):
        meta_file = os.path.join(self.sample_dir, "samples_meta.json")
        eval_vqa_file = os.path.join(self.sample_dir, "VRSBench_EVAL_vqa.json")
        eval_cap_file = os.path.join(self.sample_dir, "VRSBench_EVAL_Cap.json")
        eval_ref_file = os.path.join(self.sample_dir, "VRSBench_EVAL_referring.json")
        
        # 1. Load from samples_meta.json if present
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
                for rec in records:
                    img_path = rec.get("saved_path") or os.path.join(self.sample_dir, f"vrsbench_sample_{rec.get('index', 0)}.png")
                    
                    # Add caption item
                    if rec.get("caption") and (self.task_filter in [None, "caption"]):
                        self.items.append({
                            "image_path": img_path,
                            "query": "Describe the land-cover and objects in this satellite image.",
                            "answer": rec["caption"],
                            "task_type": "caption",
                            "metadata": {"source": "VRSBench", "type": "caption"},
                        })
                        
                    # Add QA item(s)
                    qa_list = rec.get("qa_pairs") or []
                    if isinstance(qa_list, list):
                        for qa in qa_list:
                            if self.task_filter in [None, "vqa"]:
                                self.items.append({
                                    "image_path": img_path,
                                    "query": qa.get("question", "What is visible here?"),
                                    "answer": str(qa.get("answer", "")),
                                    "task_type": "vqa",
                                    "metadata": {"source": "VRSBench", "qa_id": qa.get("ques_id")},
                                })
                                
                    # Add grounding item(s)
                    objs = rec.get("objects") or []
                    if isinstance(objs, list):
                        for obj in objs:
                            if self.task_filter in [None, "grounding"]:
                                sent = obj.get("referring_sentence") or f"Locate the {obj.get('obj_cls', 'object')}"
                                coord = obj.get("obj_coord") or obj.get("obj_corner", [])
                                self.items.append({
                                    "image_path": img_path,
                                    "query": f"Locate and highlight: {sent}",
                                    "answer": f"bbox: {coord}",
                                    "task_type": "grounding",
                                    "metadata": {"source": "VRSBench", "box": coord, "class": obj.get("obj_cls")},
                                })
            except Exception as e:
                logger.warning(f"Error reading samples_meta.json: {e}")
                
        # 2. If items still empty, populate from evaluation JSONs
        if len(self.items) == 0 and os.path.exists(eval_vqa_file):
            with open(eval_vqa_file, "r", encoding="utf-8") as f:
                vqa_entries = json.load(f)
            for idx, entry in enumerate(vqa_entries[:max_samples or 50]):
                img_path = os.path.join(self.sample_dir, f"vrsbench_sample_{idx % 3}.png")
                self.items.append({
                    "image_path": img_path,
                    "query": entry.get("question", "What is in the image?"),
                    "answer": entry.get("ground_truth", ""),
                    "task_type": "vqa",
                    "metadata": {"source": "VRSBench", "type": entry.get("type"), "image_id": entry.get("image_id")},
                })
                
        if max_samples:
            self.items = self.items[:max_samples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> SatQuerySample:
        item = self.items[idx]
        
        # Load image
        img_path = item["image_path"]
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
        else:
            # Fallback synthetic image if needed
            img = Image.new("RGB", (512, 512), color=(100, 150, 100))
            
        if self.transform:
            img = self.transform(img)
            
        return {
            "images": [img], # Single image for Specialist A
            "query": item["query"],
            "answer": item["answer"],
            "task_type": item["task_type"],
            "metadata": item["metadata"],
        }
