"""
SatQuery AI — LangGraph Multi-Agent Orchestrator
SIH 2026 (PS 26167)

State Graph Workflow:
1. classify_task (LLM Task Intent Classification)
2. validate_input (Geospatial & Format Verification via pairing.py)
3. select_and_call_specialist (Dynamic Multi-Service Routing to A/B/C)
4. combine_outputs (Evidence & Confidence Synthesis)
5. log_execution (Auditable Execution Trace Generation)
"""

import os
import sys
import time
import json
import logging
from typing import List, Dict, Any, Optional, TypedDict
import requests

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.pairing import check_spatial_alignment, align_image_pair
from orchestrator.gemini_synthesizer import synthesize_with_gemini, classify_intent_with_gemini

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LangGraphOrchestrator")

SERVICE_URLS = {
    "specialist_a": os.environ.get("SERVICE_A_URL", "http://127.0.0.1:8001"),
    "specialist_b": os.environ.get("SERVICE_B_URL", "http://127.0.0.1:8002"),
    "specialist_c": os.environ.get("SERVICE_C_URL", "http://127.0.0.1:8003"),
}
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


class AgentState(TypedDict):
    query: str
    images: List[str]
    metadata: Dict[str, Any]
    analysis_mode: str
    task_type: str
    input_modality: str
    validation_passed: bool
    was_resampled: bool
    validation_details: Dict[str, Any]
    specialists_to_call: List[str]
    specialist_responses: List[Dict[str, Any]]
    final_answer: str
    final_confidence: float
    evidence_image_base64: Optional[str]
    execution_trace: Dict[str, Any]
    start_time: float


# Node 1: Task Classification & Dynamic Telemetry Expansion
def classify_task(state: AgentState) -> AgentState:
    query = state["query"]
    images = state.get("images", [])
    img_count = len(images)
    q_lower = query.lower()

    logger.info(f"Classifying task for query: '{query}' ({img_count} input images)...")
    
    # Use Gemini Flash AI to parse user intent intelligently
    intent = classify_intent_with_gemini(query)
    logger.info(f"Gemini intent classification result: '{intent}'")

    # 1. Prioritize Optical-SAR Fusion if intent or keywords indicate fusion
    if intent == "SAR_FUSION" or any(k in q_lower for k in ["fusion", "sar", "radar", "cross-modal", "microwave"]):
        task_type = "fusion"
        input_modality = "cross_modal_pair"
    # 2. Check for Bi-Temporal Change Detection
    elif intent == "BITEMPORAL_CHANGE" or any(k in q_lower for k in ["what has changed", "changed", "change", "past", "history", "historical", "years ago", "before and after", "temporal", "evolution", "expansion"]):
        task_type = "change_vqa"
        input_modality = "bitemporal_pair"
        
        if img_count == 1:
            # If user provided 1 image and requested historical change, auto-fetch historical baseline
            import re
            from data.satellite_fetcher import fetch_satellite_pair_by_coords

            meta = state.get("metadata", {}) or {}
            lat = meta.get("lat")
            lon = meta.get("lon")
            radius_km = meta.get("radius_km", 5.0)
            years_ago = meta.get("years_ago", 5)

            if lat is None or lon is None:
                coord_match = re.search(r"(\d+\.\d+)\s*°?\s*N?.*,?\s*(\d+\.\d+)\s*°?\s*E?", query)
                if coord_match:
                    try:
                        lat = float(coord_match.group(1))
                        lon = float(coord_match.group(2))
                    except Exception:
                        lat, lon = None, None
                else:
                    lat, lon = None, None

            if lat is not None and lon is not None:
                years_match = re.search(r"(\d+)\s*years?", q_lower)
                if years_match:
                    years_ago = int(years_match.group(1))

                logger.info(f"Auto-fetching historical satellite baseline for coordinates ({lat}, {lon}), {years_ago} years ago...")
                satellite_data = fetch_satellite_pair_by_coords(lat=float(lat), lon=float(lon), radius_km=float(radius_km), years_ago=int(years_ago))

                # Historical baseline is Time 1, present observation is Time 2
                present_img = images[0] if img_count > 0 else satellite_data["time2_base64"]
                images = [satellite_data["time1_base64"], present_img]
                state["images"] = images
    elif img_count >= 2:
        # Default for 2 images without specific fusion keywords
        task_type = "change_vqa"
        input_modality = "bitemporal_pair"
    else:
        # Single Image VQA / Caption / Grounding
        if any(k in q_lower for k in ["caption", "describe", "overview", "summary"]):
            task_type = "caption"
        elif any(k in q_lower for k in ["where", "locate", "bounding", "box", "find"]):
            task_type = "grounding"
        else:
            task_type = "vqa"
        input_modality = "single_optical"

    state["task_type"] = task_type
    state["input_modality"] = input_modality
    logger.info(f"Classified: task_type='{task_type}', input_modality='{input_modality}'")
    return state


# Node 2: Input Validation & Spatial Co-Registration
def validate_input(state: AgentState) -> AgentState:
    images = state.get("images", [])
    task_type = state["task_type"]
    
    validation_passed = True
    was_resampled = False
    details = {}

    if task_type in ["change_vqa", "fusion"] and len(images) < 2:
        logger.warning(f"Task '{task_type}' requires 2 images, but received {len(images)}.")
        validation_passed = False
        details["error"] = f"Insufficient images for {task_type}. Exactly 2 images required."
    elif len(images) == 0:
        validation_passed = False
        details["error"] = "No image data provided in query request."
    else:
        # Strictly validate that each provided base64 string decodes into a valid image
        import io, base64
        from PIL import Image as PILImage
        from data.geotiff_parser import extract_geotiff_metadata
        geotiff_info = None

        for idx, img_b64 in enumerate(images):
            try:
                if not isinstance(img_b64, str) or not img_b64.strip():
                    raise ValueError(f"Image #{idx+1} payload is empty or not a string.")
                clean_b64 = img_b64.split(",")[1] if "," in img_b64 else img_b64
                clean_b64 = "".join(clean_b64.split())
                rem = len(clean_b64) % 4
                if rem == 1:
                    clean_b64 = clean_b64[:-1]
                elif rem > 1:
                    clean_b64 += '=' * (4 - rem)
                raw_bytes = base64.b64decode(clean_b64)
                with PILImage.open(io.BytesIO(raw_bytes)) as pil_img:
                    pil_img.verify()

                # Check if it contains genuine GeoTIFF metadata
                gt_meta = extract_geotiff_metadata(raw_bytes)
                if gt_meta:
                    geotiff_info = gt_meta
            except Exception as e:
                logger.error(f"Image validation failed for image #{idx+1}: {e}")
                validation_passed = False
                details["error"] = f"Corrupted or unreadable base64 satellite image data (Image #{idx+1}: {str(e)})."
                break
        
        if validation_passed:
            details["image_count"] = len(images)
            details["status"] = "Valid image schema and verified decodable satellite pixels"
            if geotiff_info:
                state["geotiff_metadata"] = geotiff_info
                details["geotiff"] = geotiff_info
        
    state["validation_passed"] = validation_passed
    state["was_resampled"] = was_resampled
    state["validation_details"] = details
    return state


# Node 3: Specialist Routing & Invocation
def select_and_call_specialist(state: AgentState) -> AgentState:
    if not state.get("validation_passed", True):
        state["specialist_responses"] = [{
            "specialist": "system_guard",
            "answer": f"Validation Error: {state['validation_details'].get('error', 'Invalid input payload')}",
            "confidence": 0.0,
            "evidence_image_base64": None,
            "error": True
        }]
        return state

    task_type = state["task_type"]
    query = state["query"]
    images = state["images"]
    
    specialists = []
    if task_type in ["vqa", "caption", "grounding"]:
        specialists = ["specialist_a"]
    elif task_type == "change_vqa":
        # Dedicated Bi-Temporal Change-VQA Specialist
        specialists = ["specialist_b"]
    elif task_type == "fusion":
        # Dedicated Optical-SAR Cross-Modal Fusion Specialist
        specialists = ["specialist_c"]
    else:
        specialists = ["specialist_a"]

    state["specialists_to_call"] = specialists
    responses = []

    for spec in specialists:
        service_url = SERVICE_URLS.get(spec)
        logger.info(f"Invoking {spec} at {service_url}/infer...")
        
        payload = {
            "images": images,
            "query": query,
            "task_type": task_type
        }
        
        try:
            if os.environ.get("SATQUERY_WORKER_TOKEN"):
                from orchestrator.gpu_relay import infer
                responses.append(infer(spec, payload))
                continue
            res = requests.post(f"{service_url}/infer", json=payload, timeout=60)
            if res.status_code == 200:
                data = res.json()
                responses.append(data)
            else:
                logger.warning(f"{spec} returned status {res.status_code}: {res.text}")
                responses.append({
                    "specialist": spec,
                    "answer": f"Specialist service error: {spec} returned HTTP {res.status_code} ({res.text[:120]}).",
                    "confidence": 0.0,
                    "evidence_image_base64": None,
                    "error": True
                })
        except Exception as e:
            logger.error(f"Failed to connect to {spec} ({service_url}): {e}")
            responses.append({
                "specialist": spec,
                "answer": f"The {spec} service could not complete the request ({e}).",
                "confidence": 0.0,
                "evidence_image_base64": None,
                "error": True
            })

    state["specialist_responses"] = responses
    return state


# Node 4: Combine Outputs & Evidence via Gemini Synthesis Layer
def combine_outputs(state: AgentState) -> AgentState:
    # If input validation failed, short-circuit directly with error
    if not state.get("validation_passed", True):
        err_msg = state.get("validation_details", {}).get("error", "Invalid input payload")
        state["final_answer"] = f"Error: Input validation failed. {err_msg}"
        state["final_confidence"] = 0.0
        state["evidence_image_base64"] = None
        return state

    responses = state.get("specialist_responses", [])
    query = state.get("query", "")
    task_type = state.get("task_type", "vqa")
    
    if not responses:
        state["final_answer"] = "Error: No response generated by specialist models."
        state["final_confidence"] = 0.0
        state["evidence_image_base64"] = None
        return state

    # If all specialists returned errors, do not call Gemini synthesis
    if all(resp.get("error") for resp in responses):
        err_details = responses[0].get("answer", "Backend specialist service unreachable.")
        state["final_answer"] = f"Error: {err_details}"
        state["final_confidence"] = 0.0
        state["evidence_image_base64"] = None
        return state

    answers = []
    confs = []
    evidence_img = None

    for resp in responses:
        if not resp.get("error"):
            ans = resp.get("answer", "").strip()
            if ans:
                answers.append(ans)
            confs.append(resp.get("confidence", 0.9))
            if resp.get("evidence_image_base64"):
                evidence_img = resp.get("evidence_image_base64")

    # Combine text answers
    if len(answers) == 1:
        raw_combined = answers[0]
    else:
        # Merged bi-temporal / fusion response
        raw_combined = "\n\n".join(answers)

    avg_conf = sum(confs) / max(1, len(confs)) if confs else 0.0

    # Pass raw output through Gemini Natural Language Synthesis Layer
    logger.info(f"Synthesizing raw specialist output into natural language via Gemini API (Mode: {state.get('analysis_mode', 'quick')})...")
    natural_answer = synthesize_with_gemini(
        query=query, 
        raw_answer=raw_combined, 
        task_type=task_type,
        images_b64=state.get("images", []),
        analysis_mode=state.get("analysis_mode", "quick")
    )

    state["final_answer"] = natural_answer
    state["final_confidence"] = round(avg_conf, 3)
    state["evidence_image_base64"] = evidence_img
    return state


# Node 5: Auditable Execution Trace Logging
def log_execution(state: AgentState) -> AgentState:
    elapsed_ms = round((time.time() - state.get("start_time", time.time())) * 1000, 2)
    
    from data.geotiff_parser import calculate_utm_crs
    meta = state.get("metadata", {}) or {}
    lat = meta.get("lat")
    lon = meta.get("lon")
    dynamic_crs = calculate_utm_crs(lat, lon)
    
    trace = {
        "task_type": state["task_type"],
        "analysis_mode": state.get("analysis_mode", "quick"),
        "input_modality": state["input_modality"],
        "images_received": len(state.get("images", [])),
        "validation_passed": state.get("validation_passed", True),
        "was_resampled": state.get("was_resampled", False),
        "crs": dynamic_crs,
        "specialists_invoked": state.get("specialists_to_call", []),
        "latency_ms": elapsed_ms,
        "confidence_score": state.get("final_confidence", 0.0),
        "models_used": [
            resp.get("execution_summary", {}).get("model_used", resp.get("specialist", "specialist"))
            for resp in state.get("specialist_responses", [])
        ],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    if state.get("geotiff_metadata"):
        trace["geotiff_metadata"] = state["geotiff_metadata"]
        if state["geotiff_metadata"].get("crs"):
            trace["crs"] = state["geotiff_metadata"]["crs"]
    
    state["execution_trace"] = trace
    logger.info(f"Execution complete in {elapsed_ms}ms with confidence {state['final_confidence']}")
    return state


# Build LangGraph State Machine
def create_satquery_orchestrator():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("classify_task", classify_task)
    workflow.add_node("validate_input", validate_input)
    workflow.add_node("select_and_call_specialist", select_and_call_specialist)
    workflow.add_node("combine_outputs", combine_outputs)
    workflow.add_node("log_execution", log_execution)

    # Add Edges
    workflow.set_entry_point("classify_task")
    workflow.add_edge("classify_task", "validate_input")
    workflow.add_edge("validate_input", "select_and_call_specialist")
    workflow.add_edge("select_and_call_specialist", "combine_outputs")
    workflow.add_edge("combine_outputs", "log_execution")
    workflow.add_edge("log_execution", END)

    app = workflow.compile()
    return app


# Singleton orchestrator app
orchestrator_graph = create_satquery_orchestrator()


def run_orchestrator(
    query: str, 
    images: List[str], 
    analysis_mode: str = "quick",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    initial_state: AgentState = {
        "query": query,
        "images": images,
        "metadata": metadata or {},
        "analysis_mode": analysis_mode,
        "task_type": "",
        "input_modality": "",
        "validation_passed": True,
        "was_resampled": False,
        "validation_details": {},
        "specialists_to_call": [],
        "specialist_responses": [],
        "final_answer": "",
        "final_confidence": 0.0,
        "evidence_image_base64": None,
        "execution_trace": {},
        "start_time": time.time()
    }
    
    final_state = orchestrator_graph.invoke(initial_state)
    
    return {
        "query": final_state["query"],
        "answer": final_state["final_answer"],
        "confidence": final_state["final_confidence"],
        "task_type": final_state["task_type"],
        "analysis_mode": final_state.get("analysis_mode", "quick"),
        "images": final_state.get("images", []),
        "evidence_image_base64": final_state["evidence_image_base64"],
        "execution_trace": final_state["execution_trace"]
    }
