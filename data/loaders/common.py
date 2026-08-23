"""
SatQuery AI — Unified Dataset Common Schema & Utilities
SIH 2026 (PS 26167)

Defines the common schema and collation functions for all remote sensing datasets:
{
    "images": list[PIL.Image.Image | torch.Tensor | np.ndarray],
    "query": str,
    "answer": str,
    "task_type": Literal["vqa", "caption", "grounding", "change_vqa", "fusion"],
    "metadata": dict
}
"""

from typing import TypedDict, Literal, List, Dict, Any, Union, Optional
from PIL import Image
import torch
import numpy as np

TaskType = Literal["vqa", "caption", "grounding", "change_vqa", "fusion"]


class SatQuerySample(TypedDict):
    images: List[Any] # List of 1 or 2 images (PIL.Image or Tensor or ndarray)
    query: str
    answer: str
    task_type: TaskType
    metadata: Dict[str, Any]


def satquery_collate_fn(batch: List[SatQuerySample]) -> Dict[str, Any]:
    """
    Collate function to assemble heterogeneous remote-sensing multimodal samples
    into clean training/evaluation batches.
    """
    queries = [item["query"] for item in batch]
    answers = [item["answer"] for item in batch]
    task_types = [item["task_type"] for item in batch]
    metadata = [item["metadata"] for item in batch]
    images_list = [item["images"] for item in batch]
    
    return {
        "images": images_list, # list of lists of images
        "query": queries,
        "answer": answers,
        "task_type": task_types,
        "metadata": metadata,
        "batch_size": len(batch),
    }


def ensure_pil_image(img_input: Any) -> Image.Image:
    """
    Converts diverse image formats (path, ndarray, Tensor) to PIL Image.
    """
    if isinstance(img_input, Image.Image):
        return img_input
    elif isinstance(img_input, str):
        return Image.open(img_input).convert("RGB")
    elif isinstance(img_input, np.ndarray):
        # If shape is (C, H, W)
        if img_input.ndim == 3 and img_input.shape[0] in [1, 2, 3, 4, 12]:
            if img_input.shape[0] >= 3:
                # Use first 3 channels for RGB
                rgb = img_input[:3].transpose(1, 2, 0)
                # Scale if uint16 or float
                if rgb.dtype == np.uint16:
                    rgb = (rgb / 256).astype(np.uint8)
                elif np.issubdtype(rgb.dtype, np.floating):
                    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                return Image.fromarray(rgb)
            elif img_input.shape[0] == 1:
                return Image.fromarray(img_input[0])
        elif img_input.ndim == 2:
            return Image.fromarray(img_input)
        return Image.fromarray(img_input)
    elif isinstance(img_input, torch.Tensor):
        arr = img_input.detach().cpu().numpy()
        return ensure_pil_image(arr)
    else:
        raise ValueError(f"Unsupported image input type: {type(img_input)}")
