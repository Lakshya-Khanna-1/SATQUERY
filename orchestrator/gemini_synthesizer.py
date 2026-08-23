"""
SatQuery AI — Gemini Natural Language Synthesis Layer
SIH 2026 (PS 26167)

Uses Gemini API (gemini-2.5-flash) to convert raw specialist model outputs
(e.g., JSON bounding boxes, technical change metrics, raw class tags)
into clear, natural, human-readable answers.
"""

import os
import json
import logging
import urllib.request
from typing import Optional

logger = logging.getLogger("GeminiSynthesizer")

DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "REDACTED_API_KEY")

def synthesize_with_gemini(query: str, raw_answer: str, task_type: str = "vqa", api_key: Optional[str] = None) -> str:
    """
    Takes the raw output from fine-tuned specialist models and uses Gemini 2.5 Flash
    to generate a clear, human-readable natural language answer.
    """
    if not raw_answer or not raw_answer.strip():
        return "No visual evidence detected in the provided satellite imagery."
    
    key = api_key or os.environ.get("GEMINI_API_KEY") or DEFAULT_API_KEY
    if not key:
        return fallback_format(query, raw_answer)
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
    
    system_prompt = f"""You are SatQuery AI's Natural Language Synthesizer for satellite geospatial intelligence.
Your task is to take the raw observation output from specialist fine-tuned computer vision models and present it to the user in a clean, natural, human-readable format.

User Query: "{query}"
Task Category: {task_type.upper()}
Raw Specialist Model Output:
{raw_answer}

Instructions:
1. Translate raw JSON data (such as bounding boxes, coordinates, class labels) or technical change metrics into a simple, natural sentence that a normal person can easily understand.
2. Never output raw ```json code blocks or raw unparsed arrays unless explicitly requested.
3. If bounding boxes are present (e.g. [ymin, xmin, ymax, xmax]), describe where the object is located in the image (e.g. upper-left, center, lower-right) along with the detected item name.
4. Keep the answer concise, accurate, direct, and conversational.
"""

    payload = {
        "contents": [{
            "parts": [{"text": system_prompt}]
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
        with urllib.request.urlopen(req, timeout=10) as res:
            res_data = json.loads(res.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts and "text" in parts[0]:
                    synthesized = parts[0]["text"].strip()
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
            # Strip markdown formatting
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
