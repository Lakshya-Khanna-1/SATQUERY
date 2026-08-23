# AGENTS.md — SatQuery AI

This file is persistent context. Read it before every task. Also read `specs/PRD.md` for
full requirements and `TASKLIST.md` for what to build next — always work through
TASKLIST.md in order, one task at a time, and don't jump ahead.

## What this project is

SatQuery AI — an agentic vision-language assistant for remote-sensing image analysis
(SIH 2026, PS 26167, ISRO/SAC). It answers natural-language queries about satellite
imagery: single-image VQA/captioning/grounding, bi-temporal change analysis, and
optical–SAR fusion — routed through fine-tuned specialist models by an agentic
controller, not a single generic VLM.

## Tech stack

- **Data/preprocessing:** Python, rasterio, GDAL, PyTorch Dataset classes
- **Fine-tuning:** HuggingFace `transformers`, `peft`, `bitsandbytes` (QLoRA, 4-bit)
- **Model serving:** Ollama (for GGUF-exported specialists + orchestrator LLM),
  FastAPI (for non-chat models like the fusion/segmentation specialist)
- **Orchestration:** LangGraph (the actual controller logic), n8n (thin webhook/retry
  front door in front of the orchestrator — not where logic lives)
- **Frontend:** Next.js + Tailwind
- **Reports:** reportlab or weasyprint (PDF generation)

## Critical correction the agent must respect

**Ollama does not fine-tune models.** It only serves GGUF weights for inference.
The real pipeline is:
```
HF base model → QLoRA fine-tune (peft + bitsandbytes) → merge adapter into base
→ convert to GGUF (llama.cpp) → ollama create (Modelfile) → serve
```
Never write code that assumes Ollama can train or fine-tune directly.

## Repo structure (create and respect this)

```
/data          → download + preprocessing scripts, unified dataset loaders
/finetune      → QLoRA training scripts, one subfolder per specialist
/models        → GGUF export scripts + Ollama Modelfiles
/services      → FastAPI wrappers, one per specialist model
/orchestrator  → LangGraph controller
/n8n           → exported n8n workflow JSON
/frontend      → Next.js minimalist GUI
/reports       → PDF report generator
```

## Non-negotiables

1. No generic LLM answers a remote-sensing query without going through a fine-tuned
   specialist first. Every response must state which specialist handled it.
2. Every response includes: task type, model(s) used, confidence score, and an
   auditable execution summary (this is a judged requirement in the PS).
3. Support GeoTIFF/TIFF as the primary format. PNG/JPEG only for public benchmark
   demo data (VRSBench/RSVQA/CDVQA images).
4. The system must be sensor-agnostic — validate images by band count/resolution/CRS,
   not hardcoded sensor names, since the hidden ISRO/SAC eval set uses Cartosat-2S +
   RISAT imagery we never see during development.
5. UI stays minimalist: one upload zone, one query box, one result panel. No visible
   sidebar or settings panel by default.
6. Hardware assumption: a single consumer GPU (RTX A2000-class). Use QLoRA 4-bit and
   small (2–3B) VLMs — do not attempt full fine-tunes or 7B+ vision models for the
   specialists.

## Model stack (fixed — don't substitute without asking)

| Role | Base model | Fine-tune data | Method |
|---|---|---|---|
| Specialist A — single-image VQA/caption/grounding | Qwen/Qwen2.5-VL-3B-Instruct | VRSBench + RSVQA-LR | QLoRA |
| Specialist B — change-VQA (bi-temporal) | Qwen/Qwen2.5-VL-3B-Instruct (separate copy) | CDVQA (+ SECOND images) | QLoRA, multi-image prompt |
| Specialist C — optical–SAR fusion + change map | Dual-branch ResNet-50/ViT-S encoder + U-Net head | BigEarthNet-MM (+ SECOND masks) | Contrastive fine-tune + segmentation head |
| Orchestrator | Qwen2.5:7b-instruct via Ollama | — (prompted only, not fine-tuned) | — |

## Dataset sources (confirmed — use exactly these)

- **VRSBench** — `load_dataset("xiang709/VRSBench", name="VRSBench", split="train", streaming=True)` — stream, don't bulk-download.
- **RSVQA-LR** — download the "Low resolution dataset" from rsvqa.sylvainlobry.com (or Zenodo record 6344334). Skip RSVQA-HR and RSVQAxBEN for v1.
- **CDVQA** — `github.com/YZHJessica/CDVQA` gives only the QA JSON files (Train/Val/Test/Test2 questions, answers, image-filename lists). The actual bi-temporal images come from **SECOND**, hosted at `captain-whu.com/PROJECT/SCD/` — match images to the JSON by filename.
- **BigEarthNet-MM** — stream a subset via `torchgeo`'s `BigEarthNet` dataset class (auto-downloads/caches only what's requested) or the Hugging Face mirror `lc-col/bigearthnet` with `streaming=True`. Do not pull the full ~130GB archive.

## Working style

- Complete one task from TASKLIST.md fully (code + a quick smoke test) before
  reporting done and moving to the next.
- If a task is ambiguous, make the smallest reasonable assumption, state it in your
  output, and proceed — don't stall waiting for clarification.
- Prefer small, runnable increments over large speculative builds.
