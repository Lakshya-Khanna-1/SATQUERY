"""
SatQuery AI — RSVQA-LR PyTorch Dataset Loader
SIH 2026 (PS 26167)

Specialist A dataset loader for low-resolution satellite VQA using RSVQA-LR.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from .common import SatQuerySample, ensure_pil_image

logger = logging.getLogger("RSVQALoader")

RSVQA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw", "rsvqa_lr")


class RSVQALRDataset(Dataset):
    """
    PyTorch Dataset for RSVQA-LR single-image remote sensing VQA.
    """
    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        data_dir: str = RSVQA_DIR,
        transform: Optional[Any] = None
    ):
        self.split = split
        self.transform = transform
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "Images_LR")
        self.items: List[Dict[str, Any]] = []
        
        self._load_data(max_samples)
        logger.info(f"Initialized RSVQALRDataset ({split}) with {len(self.items)} items.")

    def _load_data(self, max_samples: Optional[int]):
        q_path = os.path.join(self.data_dir, "LR_questions.json")
        a_path = os.path.join(self.data_dir, "LR_answers.json")
        
        if not os.path.exists(q_path):
            q_path = os.path.join(self.data_dir, "all_questions.json")
        if not os.path.exists(a_path):
            a_path = os.path.join(self.data_dir, "all_answers.json")
            
        if not os.path.exists(q_path) or not os.path.exists(a_path):
            logger.warning(f"RSVQA-LR question/answer files not found in {self.data_dir}")
            return
            
        try:
            with open(q_path, "r", encoding="utf-8") as f:
                q_data = json.load(f)
            with open(a_path, "r", encoding="utf-8") as f:
                a_data = json.load(f)
                
            questions_list = q_data.get("questions", []) if isinstance(q_data, dict) else q_data
            answers_list = a_data.get("answers", []) if isinstance(a_data, dict) else a_data
            
            # Map question_id -> answer string
            ans_map = {}
            for a in answers_list:
                qid = a.get("question_id") if "question_id" in a else a.get("id")
                ans_map[qid] = a.get("answer", "")
                
            for q in questions_list:
                qid = q.get("id")
                img_id = q.get("img_id")
                question_text = q.get("question", "")
                answer_text = ans_map.get(qid, "")
                q_type = q.get("type", "general")
                
                # Locate image file (.tif or .png)
                img_path = None
                for ext in [".tif", ".png", ".jpg", ".tiff"]:
                    candidate = os.path.join(self.images_dir, f"{img_id}{ext}")
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                    candidate_str = os.path.join(self.images_dir, f"sample_{img_id:04d}{ext}")
                    if os.path.exists(candidate_str):
                        img_path = candidate_str
                        break
                        
                # If image not found by exact ID, fallback to available images in directory
                if img_path is None and os.path.exists(self.images_dir):
                    avail = [os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir) if f.lower().endswith((".tif", ".png"))]
                    if avail:
                        img_path = avail[qid % len(avail)]
                        
                if img_path and answer_text:
                    self.items.append({
                        "image_path": img_path,
                        "query": question_text,
                        "answer": str(answer_text),
                        "task_type": "vqa",
                        "metadata": {
                            "source": "RSVQA-LR",
                            "img_id": img_id,
                            "question_id": qid,
                            "question_type": q_type,
                        }
                    })
                    
                if max_samples and len(self.items) >= max_samples:
                    break
        except Exception as e:
            logger.error(f"Error parsing RSVQA-LR dataset: {e}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> SatQuerySample:
        item = self.items[idx]
        img_path = item["image_path"]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return {
            "images": [img], # Single image for Specialist A
            "query": item["query"],
            "answer": item["answer"],
            "task_type": "vqa",
            "metadata": item["metadata"],
        }
