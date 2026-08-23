# /services — Specialist Microservices

FastAPI microservice endpoints exposing standard inference APIs:

- **`vqa_service.py`**: Interacts with `satquery-vqa` via Ollama for single-image VQA/grounding.
- **`changevqa_service.py`**: Interacts with `satquery-changevqa` via Ollama for bi-temporal Change-VQA.
- **`fusion_service.py`**: Interacts directly with Specialist C PyTorch checkpoint for optical-SAR joint embeddings & change masks.

All services adhere to the unified POST `/infer` schema:
```json
{
  "images": ["<base64_img1>", "..."],
  "query": "string"
}
```
Returning:
```json
{
  "answer": "string",
  "confidence": 0.92,
  "evidence_image_base64": "<base64_overlay_or_null>"
}
```
