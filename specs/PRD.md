# PRD — SatQuery AI (SIH 2026, PS 26167)

## 1. Background

Remote-sensing imagery supports agriculture, disaster management, urban planning,
forestry, water-resource assessment, and infrastructure mapping. Most existing
remote-sensing AI tools are single-task and require GIS/domain expertise to operate.
Many real questions can't be answered from one optical image alone — SAR adds
structural/all-weather information optical lacks, and bi-temporal pairs are needed to
detect and describe change. A general-purpose LLM/VLM can't do this reliably without
domain adaptation. SatQuery AI is an agentic system that adapts specialist models to
remote-sensing data and routes queries to the right specialist(s) automatically.

## 2. Input scope

- **Single image** — one optical/multispectral or SAR image (captioning, VQA, grounding)
- **Cross-modal pair** — co-registered optical + SAR of the same area (joint analysis)
- **Bi-temporal pair** — two images of the same area at different times (change analysis)
- **Formats** — GeoTIFF/TIFF primary; PNG/JPEG accepted only for public benchmark data

## 3. Mandatory functional requirements

1. **Remote-sensing adaptation** — at least one vision/VLM component fine-tuned on
   BigEarthNet or another open dataset (not a stock generic VLM).
2. **Single-image baseline** — VQA is mandatory; plus at least one of
   captioning/scene-description or text-guided region grounding.
3. **Multi-image change analysis** — change description or change-VQA from a
   bi-temporal pair is mandatory. A spatial change map is optional (needs reference masks).
4. **Cross-modal pair analysis** — extract complementary info from co-registered
   optical + SAR.
5. **Agentic orchestration** — automatically classify the query, validate/check input
   compatibility, select and sequence the right specialist(s), execute, combine
   outputs, estimate confidence, and return an auditable execution trace (task type,
   model/tool names, key parameters). Internal chain-of-thought is not evaluated —
   only the observable execution trace is.

## 4. Representative queries the system must handle

- "Describe the land-cover and major objects visible in this image."
- "Highlight the water body referred to in the query."
- "What changed between these two dates, and where did the change occur?"
- "Use the optical and SAR images together to identify built-up and water-covered regions."
- "Has the built-up area increased, decreased, or remained unchanged?"

## 5. Architecture (4 parts)

1. **Data & Preprocessing Pipeline** — GeoTIFF/TIFF ingestion, band/CRS validation,
   optical–SAR co-registration checks, dataset loaders for VRSBench/RSVQA/CDVQA/BigEarthNet.
2. **Model Fine-Tuning Layer** — QLoRA fine-tunes for Specialists A and B, a
   contrastive + segmentation fine-tune for Specialist C, GGUF export + Ollama
   registration, benchmark eval scripts.
3. **Agentic Orchestration Layer** — LangGraph controller (classify → validate →
   route → combine → log) wrapped by a thin n8n webhook/retry layer.
4. **Frontend + Reporting** — minimalist Next.js GUI, evidence images, confidence,
   collapsible execution trace, downloadable PDF report.

## 6. Model stack

| Role | Base model | Fine-tune data | Method | Output |
|---|---|---|---|---|
| A — VQA/caption/grounding | Qwen2.5-VL-3B-Instruct | VRSBench + RSVQA-LR | QLoRA | Text + boxes |
| B — Change-VQA | Qwen2.5-VL-3B-Instruct (2nd copy) | CDVQA + SECOND images | QLoRA, multi-image | Text |
| C — Optical–SAR fusion + change map | Dual-branch ResNet/ViT + U-Net head | BigEarthNet-MM + SECOND masks | Contrastive + segmentation | Embedding + overlay image |
| Orchestrator | Qwen2.5:7b-instruct (Ollama) | — | Prompted only | Routing + report text |

Image-as-output is satisfied deterministically (grounding boxes, change-map overlay,
fused false-color composite drawn with PIL/matplotlib) — no generative diffusion needed.

## 7. Datasets

| Dataset | Role | Source |
|---|---|---|
| VRSBench | Captioning/grounding/VQA fine-tune + eval | `xiang709/VRSBench` on HF, stream |
| RSVQA-LR | Single-image VQA fine-tune + eval | rsvqa.sylvainlobry.com → "Low resolution dataset" |
| CDVQA | Change-VQA fine-tune + eval | `github.com/YZHJessica/CDVQA` (JSON) + SECOND images from `captain-whu.com/PROJECT/SCD/` |
| BigEarthNet-MM | Optical–SAR domain adaptation | `bigearth.net`/Zenodo, streamed via torchgeo or HF `lc-col/bigearthnet` |

The ISRO/SAC hidden eval set (Cartosat-2S optical + RISAT SAR) is never available to
the team — design for sensor-agnostic generalization instead of hardcoding to it.

## 8. Non-functional requirements

- **Auditability** — every answer ships task type, model(s) used, key parameters,
  and a normalized confidence score.
- **Minimalist UI** — upload zone + query box + result panel only, visible by default.
- **Runs on one consumer GPU** — QLoRA 4-bit, 2–3B vision models.
- **Judged on public benchmark test splits + the ISRO/SAC set**, scores normalized
  before combining across metrics.

## 9. Deliverables

An interactive GUI/web app with the agentic remote-sensing backend, all code and
trained model weights/adapters, plus a working demonstration covering: single-image
VQA, one additional single-image task, change understanding, optical–SAR paired
analysis, and agentic orchestration end-to-end.

## 10. Out of scope for v1

- RSVQA-HR and RSVQAxBEN (add later if time allows)
- Full 130GB BigEarthNet archive (train on a streamed subset)
- Any hardware/edge-device component (software-only, per team preference)
