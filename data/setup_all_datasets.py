"""
Unified Dataset Setup & Acquisition CLI
SatQuery AI — SIH 2026 (PS 26167)

Orchestrates download, streaming verification, and directory setup
for all 4 datasets:
1. VRSBench (HF streaming)
2. RSVQA-LR (Zenodo / Lobry mirror)
3. CDVQA + SECOND (Git clone + bi-temporal imagery)
4. BigEarthNet-MM (Multimodal S1/S2 streaming & GeoTIFFs)
"""

import os
import sys
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SatQuery-DataSetup")

from download_vrsbench import fetch_vrsbench_samples
from download_rsvqa_lr import setup_rsvqa_lr
from download_cdvqa_second import setup_cdvqa_second_all
from download_bigearthnet import setup_bigearthnet


def setup_all(sample_count: int = 5, force: bool = False):
    logger.info("=" * 60)
    logger.info("SatQuery AI — Starting Full Dataset Ingestion & Setup Pipeline")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. VRSBench
    logger.info("\n>>> [1/4] Setting up VRSBench streaming...")
    try:
        samples = fetch_vrsbench_samples(num_samples=sample_count, save_to_disk=True)
        results["VRSBench"] = f"SUCCESS (Streamed {len(samples)} samples)"
    except Exception as e:
        results["VRSBench"] = f"FAILED: {e}"
        logger.error(f"VRSBench failed: {e}")
        
    # 2. RSVQA-LR
    logger.info("\n>>> [2/4] Setting up RSVQA-LR dataset...")
    try:
        ok = setup_rsvqa_lr(force=force)
        results["RSVQA-LR"] = "SUCCESS" if ok else "FAILED"
    except Exception as e:
        results["RSVQA-LR"] = f"FAILED: {e}"
        logger.error(f"RSVQA-LR failed: {e}")
        
    # 3. CDVQA + SECOND
    logger.info("\n>>> [3/4] Setting up CDVQA & SECOND datasets...")
    try:
        ok = setup_cdvqa_second_all()
        results["CDVQA + SECOND"] = "SUCCESS" if ok else "FAILED"
    except Exception as e:
        results["CDVQA + SECOND"] = f"FAILED: {e}"
        logger.error(f"CDVQA + SECOND failed: {e}")
        
    # 4. BigEarthNet-MM
    logger.info("\n>>> [4/4] Setting up BigEarthNet-MM dataset...")
    try:
        ok = setup_bigearthnet(num_samples=sample_count)
        results["BigEarthNet-MM"] = "SUCCESS" if ok else "FAILED"
    except Exception as e:
        results["BigEarthNet-MM"] = f"FAILED: {e}"
        logger.error(f"BigEarthNet-MM failed: {e}")
        
    logger.info("\n" + "=" * 60)
    logger.info("DATASET SETUP SUMMARY:")
    logger.info("=" * 60)
    all_passed = True
    for ds_name, status in results.items():
        logger.info(f" - {ds_name:20s}: {status}")
        if not status.startswith("SUCCESS"):
            all_passed = False
            
    if not all_passed:
        logger.warning("One or more dataset setups encountered warnings/failures.")
    else:
        logger.info("\nAll 4 datasets are reachable, structured, and ready for validation!")
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatQuery AI Dataset Acquisition CLI")
    parser.add_argument("--samples", type=int, default=5, help="Number of sample patches to stream/cache")
    parser.add_argument("--force", action="store_true", help="Force re-download of existing archives")
    args = parser.parse_args()
    
    success = setup_all(sample_count=args.samples, force=args.force)
    sys.exit(0 if success else 1)
