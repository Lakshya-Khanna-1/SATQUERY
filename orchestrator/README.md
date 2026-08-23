# /orchestrator — LangGraph Multi-Agent Controller

This directory contains the core orchestration state graph powered by `qwen2.5:7b-instruct` (via Ollama):

1. **`classify_task`**: Parses natural language query, detects modality requirements (single optical, SAR, cross-modal pair, bi-temporal pair).
2. **`validate_input`**: Performs format, band count, and spatial co-registration verification.
3. **`select_and_call_specialist`**: Dispatches requests to Specialist A, B, and/or C microservices.
4. **`combine_outputs`**: Synthesizes textual findings, overlays grounding/change masks, and computes unified confidence scores.
5. **`log_execution`**: Constructs auditable trace records (task type, models invoked, runtime parameters).

Exposes POST `/query`.
