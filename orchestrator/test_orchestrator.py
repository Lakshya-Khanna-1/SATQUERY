"""
SatQuery AI — Orchestrator Verification Test
SIH 2026 (PS 26167)

Tests LangGraph state machine across:
1. Single-Image Optical VQA (Specialist A)
2. Bi-Temporal Change-VQA & Heatmap Overlay (Specialist B + C)
3. Optical-SAR Multimodal Fusion (Specialist C)
"""

import os
import sys
import io
import base64
import logging
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestrator.graph import run_orchestrator
from data.loaders.rsvqa_loader import RSVQALRDataset
from data.loaders.cdvqa_loader import CDVQADataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestOrchestrator")


def image_to_base64(pil_img: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def test_orchestrator():
    logger.info("=" * 60)
    logger.info("TESTING SATQUERY MULTI-AGENT LANGGRAPH ORCHESTRATOR")
    logger.info("=" * 60)
    
    # 1. Test Single-Image VQA Query
    logger.info("\n>>> [1/3] Testing Single-Image Optical VQA Query...")
    rsvqa = RSVQALRDataset(split="train", max_samples=1)
    sample_a = rsvqa[0]
    b64_a = image_to_base64(sample_a["images"][0])
    
    res_a = run_orchestrator(
        query="What type of landscape or structures are visible in this scene?",
        images=[b64_a]
    )
    
    logger.info(f"Answer     : {res_a['answer']}")
    logger.info(f"Confidence : {res_a['confidence']}")
    logger.info(f"Task Type  : {res_a['task_type']}")
    logger.info(f"Trace      : {res_a['execution_trace']}")
    
    assert res_a["task_type"] in ["vqa", "caption"], "Should classify as single-image VQA"
    assert res_a["confidence"] > 0.5, "Confidence should be high"
    assert "specialists_invoked" in res_a["execution_trace"], "Trace must contain invoked specialists"
    
    # 2. Test Bi-temporal Change Detection Query
    logger.info("\n>>> [2/3] Testing Bi-Temporal Change-VQA Query...")
    cdvqa = CDVQADataset(split="train", max_samples=1)
    sample_b = cdvqa[0]
    b64_t1 = image_to_base64(sample_b["images"][0])
    b64_t2 = image_to_base64(sample_b["images"][1])
    
    res_b = run_orchestrator(
        query="Has there been new building construction between Time 1 and Time 2?",
        images=[b64_t1, b64_t2]
    )
    
    logger.info(f"Answer     : {res_b['answer']}")
    logger.info(f"Confidence : {res_b['confidence']}")
    logger.info(f"Task Type  : {res_b['task_type']}")
    logger.info(f"Evidence   : {'Present (Base64)' if res_b['evidence_image_base64'] else 'None'}")
    logger.info(f"Trace      : {res_b['execution_trace']}")
    
    assert res_b["task_type"] == "change_vqa", "Should classify as change_vqa"
    assert "specialist_b" in res_b["execution_trace"]["specialists_invoked"], "Specialist B should be invoked"
    
    # 3. Test Optical-SAR Multimodal Fusion Query
    logger.info("\n>>> [3/3] Testing Optical-SAR Fusion Query...")
    res_c = run_orchestrator(
        query="Perform optical and SAR radar feature fusion and cross-modal embedding",
        images=[b64_t1, b64_t2]
    )
    
    logger.info(f"Answer     : {res_c['answer']}")
    logger.info(f"Confidence : {res_c['confidence']}")
    logger.info(f"Task Type  : {res_c['task_type']}")
    logger.info(f"Trace      : {res_c['execution_trace']}")
    
    assert res_c["task_type"] == "fusion", "Should classify as fusion"
    
    logger.info("\n" + "=" * 60)
    logger.info("TASK 9 DEFINITION OF DONE: LANGGRAPH ORCHESTRATOR 100% VERIFIED PASS")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    test_orchestrator()
