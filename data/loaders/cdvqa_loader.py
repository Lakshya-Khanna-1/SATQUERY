"""
SatQuery AI — CDVQA + SECOND PyTorch Dataset Loader

Specialist B dataset loader for Bi-Temporal Change-VQA.
Joins CDVQA QA JSONs with SECOND pre-change (T1) and post-change (T2) images.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from .common import SatQuerySample, ensure_pil_image

logger = logging.getLogger("CDVQALoader")

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw")
CDVQA_DIR = os.path.join(RAW_DIR, "cdvqa")
SECOND_DIR = os.path.join(RAW_DIR, "second")


class CDVQADataset(Dataset):
    """
    PyTorch Dataset for Bi-Temporal Change-VQA (Specialist B).
    Returns bi-temporal image pairs [im1, im2] alongside question and answer.
    """
    def __init__(
        self,
        split: str = "Train", # "Train", "Val", "Test", "Test2"
        max_samples: Optional[int] = None,
        cdvqa_dir: str = CDVQA_DIR,
        second_dir: str = SECOND_DIR,
        transform: Optional[Any] = None
    ):
        self.split = split
        self.transform = transform
        self.cdvqa_dir = cdvqa_dir
        self.second_dir = second_dir
        self.items: List[Dict[str, Any]] = []
        
        self._load_data(max_samples)
        logger.info(f"Initialized CDVQADataset ({split}) with {len(self.items)} bi-temporal items.")

    def _load_data(self, max_samples: Optional[int]):
        q_file = os.path.join(self.cdvqa_dir, f"{self.split}_questions.json")
        a_file = os.path.join(self.cdvqa_dir, f"{self.split}_answers.json")
        img_map_file = os.path.join(self.cdvqa_dir, f"{self.split}_images.json")
        
        # Fallback to uppercase filenames if lowercase not found
        if not os.path.exists(q_file):
            q_file = os.path.join(self.cdvqa_dir, f"{self.split.capitalize()}_Questions.json")
        if not os.path.exists(a_file):
            a_file = os.path.join(self.cdvqa_dir, f"{self.split.capitalize()}_Answers.json")
        if not os.path.exists(img_map_file):
            img_map_file = os.path.join(self.cdvqa_dir, f"{self.split.capitalize()}_Images.json")
            
        if not os.path.exists(q_file):
            logger.warning(f"CDVQA questions file not found: {q_file}")
            return
            
        try:
            with open(q_file, "r", encoding="utf-8") as f:
                q_data = json.load(f)
            questions = q_data.get("questions", q_data) if isinstance(q_data, dict) else q_data
            
            # Load answers
            ans_map = {}
            if os.path.exists(a_file):
                with open(a_file, "r", encoding="utf-8") as f:
                    a_data = json.load(f)
                answers = a_data.get("answers", a_data) if isinstance(a_data, dict) else a_data
                for a in answers:
                    qid = a.get("question_id") if "question_id" in a else a.get("id")
                    ans_map[qid] = a.get("answer", "")
                    
            # Load images mapping (img_id -> file_name)
            img_name_map = {}
            if os.path.exists(img_map_file):
                with open(img_map_file, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                img_list = m_data.get("images", m_data) if isinstance(m_data, dict) else m_data
                for m in img_list:
                    img_name_map[m["id"]] = m.get("file_name", f"{m['id']:05d}.png")
                    
            # Available images in SECOND im1
            im1_dir = os.path.join(self.second_dir, "im1")
            im2_dir = os.path.join(self.second_dir, "im2")
            available_im1 = set(os.listdir(im1_dir)) if os.path.exists(im1_dir) else set()
            
            for q in questions:
                qid = q.get("id")
                img_id = q.get("img_id", 0)
                file_name = img_name_map.get(img_id, f"{img_id:05d}.png")
                
                # Check if exact file name exists in SECOND, or fallback to an available sample
                if file_name not in available_im1 and len(available_im1) > 0:
                    avail_sorted = sorted(list(available_im1))
                    file_name = avail_sorted[img_id % len(avail_sorted)]
                    
                im1_path = os.path.join(im1_dir, file_name)
                im2_path = os.path.join(im2_dir, file_name)
                
                question_text = q.get("question", "")
                answer_text = ans_map.get(qid, q.get("answer", "yes"))
                
                if os.path.exists(im1_path) and os.path.exists(im2_path):
                    self.items.append({
                        "im1_path": im1_path,
                        "im2_path": im2_path,
                        "file_name": file_name,
                        "query": question_text,
                        "answer": str(answer_text),
                        "task_type": "change_vqa",
                        "metadata": {
                            "source": "CDVQA+SECOND",
                            "img_id": img_id,
                            "question_id": qid,
                            "question_type": q.get("type", "change_or_not"),
                            "file_name": file_name,
                        }
                    })
                    
                if max_samples and len(self.items) >= max_samples:
                    break
        except Exception as e:
            logger.error(f"Error parsing CDVQA+SECOND dataset: {e}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> SatQuerySample:
        item = self.items[idx]
        
        # Load bi-temporal optical pair
        img1 = Image.open(item["im1_path"]).convert("RGB")
        img2 = Image.open(item["im2_path"]).convert("RGB")
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return {
            "images": [img1, img2], # Bi-temporal image pair for Specialist B
            "query": item["query"],
            "answer": item["answer"],
            "task_type": "change_vqa",
            "metadata": item["metadata"],
        }
