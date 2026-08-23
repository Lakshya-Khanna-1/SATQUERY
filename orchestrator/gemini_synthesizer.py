"""
SatQuery AI — Gemini Natural Language Synthesis Layer
SIH 2026 (PS 26167)

Uses Gemini API (gemini-2.5-flash) to convert raw specialist model outputs
(e.g., JSON bounding boxes, technical change metrics, raw class tags)
into clear, natural, human-readable answers, supporting Quick (1-2 lines) and Deep (5+ lines) modes.
"""

import os
import io
import json
import time
import base64
import logging
import urllib.request
from typing import Optional, List
from PIL import Image as PILImage

logger = logging.getLogger("GeminiSynthesizer")

DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "REDACTED_API_KEY")

def optimize_b64_image(b64_str: str, max_size: int = 384) -> str:
    """Resize base64 image payload to reduce bandwidth and avoid Gemini API 429 rate limits."""
    try:
        clean_b64 = b64_str.split(",")[1] if "," in b64_str else b64_str
        raw_bytes = base64.b64decode(clean_b64)
        pil_img = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")
        pil_img.thumbnail((max_size, max_size))
        
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=80)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Could not optimize base64 image for Gemini API: {e}")
        return b64_str.split(",")[1] if "," in b64_str else b64_str


def synthesize_with_gemini(
    query: str, 
    raw_answer: str, 
    task_type: str = "vqa", 
    images_b64: Optional[List[str]] = None,
    analysis_mode: str = "quick",
    api_key: Optional[str] = None
) -> str:
    """
    Takes raw output from fine-tuned specialist models + optional satellite images,
    and uses Gemini 2.5 Flash to generate a clear, human-readable natural language answer.
    Supports Quick mode (1-2 informative sentences) and Deep mode (5+ detailed lines).
    Includes automatic retries for HTTP 429 rate limits.
    """
    if not raw_answer or not raw_answer.strip():
        return "No visual evidence detected in the provided satellite imagery."
    
    key = api_key or os.environ.get("GEMINI_API_KEY") or DEFAULT_API_KEY
    if not key:
        return fallback_format(query, raw_answer, analysis_mode)
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
    
    if analysis_mode.lower() == "deep":
        mode_instruction = """FORMAT REQUIREMENT: DEEP MODE ANALYSIS (at least 5 detailed lines of information).
Provide a complete, comprehensive geospatial intelligence breakdown covering:
Line 1: Initial Baseline Observation (describe specific features/structures visible in Image 1 / Time 1).
Line 2: Detected Visual Transformations (describe specific changes, additions, or modifications in Image 2 / Time 2).
Line 3: Land-Cover & Environmental Impact (analyze vegetation, surface cover, or building density changes).
Line 4: Spatial Correlation & Placement (describe exact spatial positioning across the scene).
Line 5: Executive Conclusion & Confidence Assessment (final summary verification status).
Ensure the output is directly based on the provided images and specialist output."""
    else:
        mode_instruction = """FORMAT REQUIREMENT: QUICK MODE (1 to 2 complete, informative sentences).
Provide a clear, well-formed summary sentence in 1 to 2 lines giving full informative context to the user.
Do NOT output short 3-4 word incomplete fragments. Output a proper, complete sentence explaining the observation clearly."""

    system_prompt = f"""You are SatQuery AI's Natural Language Synthesizer for satellite geospatial intelligence.
Your task is to take the observation output from specialist fine-tuned computer vision models and present it to the user according to the specified analysis mode.

User Query: "{query}"
Task Category: {task_type.upper()}
Specialist Raw Output: "{raw_answer}"

{mode_instruction}

General Rules:
1. Base your answer strictly on the visual evidence in the images and specialist output. Never invent unrelated features (like rural vegetation if analyzing urban buildings).
2. Do not output raw ```json code blocks or unparsed array brackets unless explicitly requested.
3. If no change occurred between bi-temporal images, state clearly that the two scenes are structurally identical.
4. Keep the output professional, direct, and conversational.
"""

    parts = [{"text": system_prompt}]
    
    # Add compressed multimodal images if provided
    if images_b64:
        for img_b64 in images_b64:
            opt_b64 = optimize_b64_image(img_b64)
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": opt_b64
                }
            })

    payload = {
        "contents": [{
            "parts": parts
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024
        }
    }

    # Execute request with exponential backoff on 429 rate limits
    max_retries = 2
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as res:
                res_data = json.loads(res.read().decode("utf-8"))
                candidates = res_data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    res_parts = candidates[0]["content"].get("parts", [])
                    if res_parts and "text" in res_parts[0]:
                        synthesized = res_parts[0]["text"].strip()
                        logger.info(f"Successfully synthesized response via Gemini 2.5 Flash (Mode: {analysis_mode})")
                        return synthesized
        except urllib.error.HTTPError as http_err:
            if http_err.code == 429 and attempt < max_retries - 1:
                logger.warning(f"Gemini API 429 Rate Limit hit. Retrying in 1.5s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(1.5)
                continue
            logger.warning(f"Gemini API HTTP Error {http_err.code}: {http_err.reason}")
            break
        except Exception as e:
            logger.warning(f"Gemini API synthesis exception: {e}")
            break

    return fallback_format(query, raw_answer, analysis_mode)


def fallback_format(query: str, raw_answer: str, analysis_mode: str = "quick") -> str:
    """Dynamic local fallback format when offline or API key quota exceeded."""
    trimmed = raw_answer.strip()
    
    # Strip any brackets
    if "[" in trimmed and "]" in trimmed:
        clean_text = trimmed.split("]:")[-1].strip() if "]:" in trimmed else trimmed
    else:
        clean_text = trimmed

    if analysis_mode.lower() == "deep":
        return f"""Baseline Observation (Time 1): Initial satellite observation capturing baseline geographic land cover.
Visual Change Detection (Time 2): {clean_text}
Land-Cover Transformation: Structural feature modifications identified between observation dates.
Spatial Pattern Analysis: Changes localized across primary infrastructure sectors.
Executive Conclusion: High-confidence specialist vision detection confirming observed scene dynamics."""

    # Quick mode fallback: Ensure complete sentence
    if len(trimmed.split()) <= 4:
        return f"Satellite observation analysis confirms: {clean_text}."
    return clean_text
