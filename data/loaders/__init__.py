"""
SatQuery AI — Unified Remote Sensing Dataset Loaders
SIH 2026 (PS 26167)

Exports standard PyTorch Dataset classes across all specialists:
- VRSBenchDataset (Specialist A: VQA, Captioning, Grounding)
- RSVQALRDataset (Specialist A: Low-res Satellite VQA)
- CDVQADataset (Specialist B: Bi-temporal Change-VQA)
- BigEarthNetDataset (Specialist C: Optical–SAR Multimodal Fusion)
"""

from .common import SatQuerySample, TaskType, satquery_collate_fn, ensure_pil_image
from .vrsbench_loader import VRSBenchDataset
from .rsvqa_loader import RSVQALRDataset
from .cdvqa_loader import CDVQADataset
from .bigearthnet_loader import BigEarthNetDataset

__all__ = [
    "SatQuerySample",
    "TaskType",
    "satquery_collate_fn",
    "ensure_pil_image",
    "VRSBenchDataset",
    "RSVQALRDataset",
    "CDVQADataset",
    "BigEarthNetDataset",
    "get_dataset",
]


def get_dataset(name: str, **kwargs):
    """
    Factory function to instantiate any of the 4 unified dataset loaders.
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")
    if "vrsbench" in name_lower or "vrs" in name_lower:
        return VRSBenchDataset(**kwargs)
    elif "rsvqa" in name_lower:
        return RSVQALRDataset(**kwargs)
    elif "cdvqa" in name_lower or "second" in name_lower:
        return CDVQADataset(**kwargs)
    elif "bigearthnet" in name_lower or "ben" in name_lower:
        return BigEarthNetDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset name: '{name}'. Choose from: vrsbench, rsvqa, cdvqa, bigearthnet")
