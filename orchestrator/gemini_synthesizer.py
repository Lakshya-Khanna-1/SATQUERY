"""
SatQuery AI — Gemini Natural Language Synthesis Layer
SIH 2026 (PS 26167)

Uses Gemini API (gemini-2.5-flash) to convert raw specialist model outputs
(e.g., JSON bounding boxes, technical change metrics, raw class tags)
into clear, natural, human-readable answers, optionally taking the original satellite image(s) as multimodal input.
"""

import os
import json
import logging
import urllib.request
from typing import Optional, List

logger = logging.getLogger("GeminiSynthesizer")

DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "REDACTED_API_KEY")

def synthesize_with_gemini(
    query: str, 
    raw_answer: str, 
    task_type: str = "vqa", 
    images_b64: Optional[List[str]] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Takes raw output from fine-tuned specialist models + optional satellite images,
    and uses Gemini 2.5 Flash to generate a clear, human-readable natural language answer.
    """
    if not raw_answer or not raw_answer.strip():
        return "No visual evidence detected in the provided satellite imagery."
    
    key = api_key or os.environ.get("GEMINI_API_KEY") or DEFAULT_API_KEY
    if not key:
        return fallback_format(query, raw_answer)
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
    
    system_prompt = f"""You are SatQuery AI's Natural Language Synthesizer for satellite geospatial intelligence.
Your task is to take the observation output from specialist fine-tuned computer vision models and present it to the user in a clean, simple, human-readable natural language format that any normal person can immediately understand.

User Query: "{query}"
Task Category: {task_type.upper()}
Specialist Observation: "{raw_answer}"

Instructions:
1. Output a clear, natural, full English sentence explaining the answer (e.g., "The main structure visible in the lower right is a circular storage tank.", or "Between the two images, new residential houses and roads have been constructed replacing the trees.").
2. Do not output raw JSON, technical tags, or unparsed arrays.
3. If no change occurred between bi-temporal images, state clearly: "There is no significant change between the two satellite images."
4. Keep the output simple, professional, direct, and conversational.
"""

    parts = [{"text": system_prompt}]
    
    # Add multimodal images if provided
    if images_b64:
        for img_b64 in images_b64:
            clean_b64 = img_b64.split(",")[1] if "," in img_b64 else img_b64
            parts.append({
                "inlineData": {
                    "mimeType": "image/png",
                    "data": clean_b64
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

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=12) as res:
            res_data = json.loads(res.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                res_parts = candidates[0]["content"].get("parts", [])
                if res_parts and "text" in res_parts[0]:
                    synthesized = res_parts[0]["text"].strip()
                    logger.info(f"Successfully synthesized response via Gemini 2.5 Flash for query: '{query}'")
                    return synthesized
    except Exception as e:
        logger.warning(f"Gemini API synthesis unavailable or timed out: {e}. Falling back to clean local formatting.")

    return fallback_format(query, raw_answer)


def fallback_format(query: str, raw_answer: str) -> str:
    """Local fallback format when offline or API key absent."""
    trimmed = raw_answer.strip()
    if trimmed.startswith("```json") or trimmed.startswith("[") or trimmed.startswith("{"):
        try:
            clean_str = trimmed.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_str)
            if isinstance(data, list):
                items = []
                for item in data:
                    if isinstance(item, dict) and "label" in item:
                        label = item["label"]
                        bbox = item.get("bbox_2d", [])
                        items.append(f"{label} (at bounding coordinates {bbox})")
                    else:
                        items.append(str(item))
                return f"Identified {len(items)} object(s): " + ", ".join(items) + "."
            elif isinstance(data, dict) and "label" in data:
                return f"Detected {data['label']} at position {data.get('bbox_2d', '')}."
        except Exception:
            pass
    return trimmed
