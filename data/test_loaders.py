"""
SatQuery AI — Dataset Loaders & Pairing Verification Test Suite
SIH 2026 (PS 26167)

Tests:
1. pairing.py: Checks CRS, Bounding Box, Resolution detection and rasterio reprojection.
2. PyTorch Dataset classes across all 4 datasets:
   - VRSBenchDataset (caption, vqa, grounding)
   - RSVQALRDataset (vqa)
   - CDVQADataset (change_vqa)
   - BigEarthNetDataset (fusion)
3. PyTorch DataLoader batch generation and common schema validation.
"""

import sys
import os

# Ensure data directory is in sys.path
sys.path.insert(0, os.path.dirname(__file__))

import logging
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestLoaders")

from pairing import run_misalignment_test
from loaders import (
    VRSBenchDataset,
    RSVQALRDataset,
    CDVQADataset,
    BigEarthNetDataset,
    satquery_collate_fn,
    SatQuerySample,
)

VALID_TASK_TYPES = {"vqa", "caption", "grounding", "change_vqa", "fusion"}


def validate_sample_schema(sample: dict, dataset_name: str) -> None:
    required_keys = ["images", "query", "answer", "task_type", "metadata"]
    for k in required_keys:
        assert k in sample, f"[{dataset_name}] Missing key '{k}' in sample dict! Found: {list(sample.keys())}"
        
    assert isinstance(sample["images"], list), f"[{dataset_name}] 'images' must be a list! Found: {type(sample['images'])}"
    assert len(sample["images"]) in [1, 2], f"[{dataset_name}] 'images' must contain 1 or 2 images! Found: {len(sample['images'])}"
    assert isinstance(sample["query"], str) and len(sample["query"]) > 0, f"[{dataset_name}] 'query' must be a non-empty string!"
    assert isinstance(sample["answer"], str) and len(sample["answer"]) > 0, f"[{dataset_name}] 'answer' must be a non-empty string!"
    assert sample["task_type"] in VALID_TASK_TYPES, f"[{dataset_name}] 'task_type' '{sample['task_type']}' not in {VALID_TASK_TYPES}!"
    assert isinstance(sample["metadata"], dict), f"[{dataset_name}] 'metadata' must be a dict!"


def test_vrsbench_loader():
    logger.info("\n>>> Testing VRSBenchDataset...")
    ds = VRSBenchDataset(max_samples=10)
    assert len(ds) > 0, "VRSBenchDataset is empty!"
    sample = ds[0]
    validate_sample_schema(sample, "VRSBenchDataset")
    logger.info(f"  [PASS] Single sample: task={sample['task_type']}, query='{sample['query'][:40]}...', images={len(sample['images'])}")
    
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=satquery_collate_fn)
    batch = next(iter(loader))
    assert batch["batch_size"] == 2
    logger.info(f"  [PASS] DataLoader batch: batch_size={batch['batch_size']}, tasks={batch['task_type']}")


def test_rsvqa_loader():
    logger.info("\n>>> Testing RSVQALRDataset...")
    ds = RSVQALRDataset(max_samples=10)
    assert len(ds) > 0, "RSVQALRDataset is empty!"
    sample = ds[0]
    validate_sample_schema(sample, "RSVQALRDataset")
    logger.info(f"  [PASS] Single sample: task={sample['task_type']}, query='{sample['query'][:40]}...', answer='{sample['answer']}'")
    
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=satquery_collate_fn)
    batch = next(iter(loader))
    assert batch["batch_size"] == 2
    logger.info(f"  [PASS] DataLoader batch: batch_size={batch['batch_size']}, queries={len(batch['query'])}")


def test_cdvqa_loader():
    logger.info("\n>>> Testing CDVQADataset...")
    ds = CDVQADataset(split="Train", max_samples=10)
    assert len(ds) > 0, "CDVQADataset is empty!"
    sample = ds[0]
    validate_sample_schema(sample, "CDVQADataset")
    assert len(sample["images"]) == 2, f"CDVQA must return 2 bi-temporal images, got {len(sample['images'])}"
    assert sample["task_type"] == "change_vqa"
    logger.info(f"  [PASS] Bi-temporal sample: task={sample['task_type']}, query='{sample['query'][:40]}...', images={len(sample['images'])}")
    
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=satquery_collate_fn)
    batch = next(iter(loader))
    assert batch["batch_size"] == 2
    logger.info(f"  [PASS] DataLoader batch: batch_size={batch['batch_size']}, tasks={batch['task_type']}")


def test_bigearthnet_loader():
    logger.info("\n>>> Testing BigEarthNetDataset...")
    ds = BigEarthNetDataset(max_samples=10)
    assert len(ds) > 0, "BigEarthNetDataset is empty!"
    sample = ds[0]
    validate_sample_schema(sample, "BigEarthNetDataset")
    assert len(sample["images"]) == 2, f"BigEarthNet must return [Optical, SAR] pair, got {len(sample['images'])}"
    assert sample["task_type"] == "fusion"
    logger.info(f"  [PASS] Optical-SAR sample: task={sample['task_type']}, S2 bands={sample['metadata']['s2_bands']}, S1 bands={sample['metadata']['s1_bands']}")
    
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=satquery_collate_fn)
    batch = next(iter(loader))
    assert batch["batch_size"] == 2
    logger.info(f"  [PASS] DataLoader batch: batch_size={batch['batch_size']}, tasks={batch['task_type']}")


def run_all_tests():
    logger.info("=" * 60)
    logger.info("STARTING TASK 2 PREPROCESSING & DATASET LOADERS TEST SUITE")
    logger.info("=" * 60)
    
    # 1. Pairing & Resampling Check
    logger.info("\n>>> [1/5] Testing pairing.py CRS/BBox/Resolution alignment...")
    pairing_ok = run_misalignment_test()
    assert pairing_ok is True
    
    # 2. VRSBench Loader
    test_vrsbench_loader()
    
    # 3. RSVQA-LR Loader
    test_rsvqa_loader()
    
    # 4. CDVQA Loader
    test_cdvqa_loader()
    
    # 5. BigEarthNet Loader
    test_bigearthnet_loader()
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL TASK 2 TESTS PASSED CLEANLY (100% SUCCESS)")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    try:
        run_all_tests()
        sys.exit(0)
    except Exception as exc:
        logger.exception(f"Test suite encountered an error: {exc}")
        sys.exit(1)
