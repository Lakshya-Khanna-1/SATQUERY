# SatQuery AI — Vision-Language Assistant for Remote Sensing

[![SIH 2026](https://img.shields.io/badge/SIH%202026-PS%2026167-blue.svg)](https://www.sih.gov.in/)
[![ISRO/SAC](https://img.shields.io/badge/Organization-ISRO%2FSAC-orange.svg)](https://www.isro.gov.in/)
[![Architecture](https://img.shields.io/badge/Architecture-Agentic%20Specialists-green.svg)](#architecture-overview)
[![Verified Tasks](https://img.shields.io/badge/All%20Tasks-100%25%20Verified-brightgreen.svg)](#verification--test-suite)

> **SatQuery AI** is an agentic vision-language assistant built for remote-sensing image intelligence (SIH 2026, PS 26167, ISRO/SAC). It handles natural-language queries across single-image VQA, captioning, region grounding, bi-temporal change detection & Change-VQA, and optical–SAR cross-modal fusion.

---

## 🏛️ Architecture Overview

SatQuery AI abandons monolithic, generic VLMs in favor of an **Agentic Specialist Architecture**:

```
                                  +-------------------+
                                  |   User Interface  | (Next.js + Tailwind)
                                  +---------+---------+
                                            | (POST /satquery)
                                  +---------v---------+
                                  |    n8n Webhook    | (Front door, retry & health check)
                                  +---------+---------+
                                            | (POST /query)
                                  +---------v---------+
                                  | LangGraph Control | (Query classification, input validation,
                                  |   (Qwen2.5:7b)    |  specialist routing, confidence fusion)
                                  +---------+---------+
                                            |
         +----------------------------------+----------------------------------+
         |                                  |                                  |
+--------v--------+                +--------v--------+                +--------v--------+
|  Specialist A   |                |  Specialist B   |                |  Specialist C   |
| Single-Image    |                | Change-VQA      |                | Optical-SAR     |
| VQA / Grounding |                | (Bi-Temporal)   |                | Fusion & Mask   |
| (Qwen2.5-VL-3B) |                | (Qwen2.5-VL-3B) |                | (Dual ResNet/   |
| (Ollama GGUF)   |                | (Ollama GGUF)   |                |  ViT + U-Net)   |
+-----------------+                +-----------------+                +-----------------+
         |                                  |                                  |
         +----------------------------------+----------------------------------+
                                            |
                                  +---------v---------+
                                  | Combined Response | (Answer text, evidence overlay,
                                  | & Execution Trace |  normalized confidence, audit log)
                                  +---------+---------+
                                            |
                                  +---------v---------+
                                  | PDF Report Engine | (ReportLab / WeasyPrint)
                                  +-------------------+
```

---

## 🧠 Model Stack & Specialists

| Role | Base Model | Training Dataset | Fine-Tuning Method | Serving Backend | Output |
|---|---|---|---|---|---|
| **Specialist A** (Single-Image VQA/Grounding) | `Qwen/Qwen2.5-VL-3B-Instruct` | VRSBench + RSVQA-LR | QLoRA (4-bit, rank 16, alpha 32) | Ollama (`satquery-vqa`) | Text answer + bounding boxes |
| **Specialist B** (Bi-Temporal Change-VQA) | `Qwen/Qwen2.5-VL-3B-Instruct` (2nd copy) | CDVQA + SECOND | QLoRA (multi-image prompt) | Ollama (`satquery-changevqa`) | Natural language change explanation |
| **Specialist C** (Optical–SAR Fusion) | Dual-branch ResNet-18 + U-Net | BigEarthNet-MM + SECOND masks | InfoNCE cross-modal alignment + 6-class segmentation | FastAPI (`services/fusion_service` :8003) | 256-d Joint embedding + RGBA Change overlay |
| **Orchestrator** | `qwen2.5:7b-instruct` | — | LangGraph State Machine | FastAPI (`orchestrator` :8000) | Auditable execution trace + Synthesis |

---

## 🛰️ Dataset Sources & Ingestion

| Dataset | Modality / Task | Source | Acquisition Strategy |
|---|---|---|---|
| **VRSBench** | Single-image captioning, grounding, VQA | `xiang709/VRSBench` (Hugging Face) | Streamed on-demand (`streaming=True`) |
| **RSVQA-LR** | Low-resolution single-image VQA | Sylvain Lobry / Zenodo (6344334) | Downloaded & unzipped to `/data/raw/rsvqa_lr` |
| **CDVQA** | Change-VQA question/answer JSONs | `github.com/YZHJessica/CDVQA` | Git cloned to `/data/raw/cdvqa` |
| **SECOND** | Bi-temporal optical pairs + semantic change masks | Wuhan University Captain Project | Fetched & extracted to `/data/raw/second` |
| **BigEarthNet-MM** | Sentinel-1 (SAR) & Sentinel-2 (Multispectral) pairs | `lc-col/bigearthnet` / TorchGeo | Streamed subset / patch cache |

---

## 📁 Repository Structure

```
satquery-ai/
├── README.md                      # Architecture & system documentation (this file)
├── AGENTS.md                      # Persistent agent instructions & rules
├── TASKLIST.md                    # Step-by-step roadmap & DoD tracker
├── pyproject.toml                 # Python environment & dependencies
│
├── data/                          # Data acquisition, streaming, pairing & validation
│   ├── loaders/                   # Unified PyTorch loaders (VRSBench, RSVQA, CDVQA, BigEarthNet)
│   ├── pairing.py                 # Spatial co-registration & reprojection engine
│   ├── validator.py               # Sensor-agnostic band count & CRS validation
│   └── test_loaders.py            # Unified loaders verification suite
│
├── finetune/                      # Fine-tuning code & adapter weights
│   ├── specialist_a/              # Specialist A QLoRA training & eval scripts
│   ├── specialist_b/              # Specialist B Bi-temporal QLoRA training & eval
│   ├── specialist_c/              # Specialist C DualBranchEncoder + BiTemporalUNet
│   └── download_weights_multithread.py # High-speed multi-threaded base model puller
│
├── models/                        # Checkpoints, GGUF conversion & Ollama Modelfiles
│   ├── Modelfile.vqa              # Ollama Modelfile for satquery-vqa
│   ├── Modelfile.changevqa        # Ollama Modelfile for satquery-changevqa
│   ├── export.py                  # LoRA merger & GGUF export utility
│   ├── serve_c.py                 # Specialist C standalone FastAPI server
│   └── OLLAMA_REGISTRATION.md     # Ollama deployment commands
│
├── services/                      # Microservices for each specialist
│   ├── vqa_service/               # Specialist A microservice (Port 8001)
│   ├── changevqa_service/         # Specialist B microservice (Port 8002)
│   └── fusion_service/           # Specialist C microservice (Port 8003)
│
├── orchestrator/                  # LangGraph multi-agent controller
│   ├── graph.py                   # State machine (classify, validate, route, merge, trace)
│   ├── main.py                    # Orchestrator FastAPI server (Port 8000)
│   └── test_orchestrator.py       # Orchestrator test suite
│
├── n8n/                           # n8n front door webhook workflow
│   ├── satquery_workflow.json     # Official n8n workflow definition
│   ├── mock_n8n_server.py         # n8n webhook proxy server (Port 5678)
│   └── test_n8n.py                # Webhook test suite
│
├── frontend/                      # Web application UI
│   ├── src/App.jsx                # Interactive multi-image upload, query & trace panel
│   ├── dist/                      # Production SPA build
│   └── vite.config.js             # Vite + React + Tailwind configuration
│
├── reports/                       # Automated PDF Report Engine
│   ├── generate.py                # ReportLab PDF Intelligence Report generator
│   └── SatQuery_E2E_Intelligence_Report.pdf # Sample generated intelligence report
│
└── scripts/                       # System-wide integration & demonstration scripts
    └── e2e_demo.py                # End-to-end full system validation script
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
```powershell
# Create & activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r finetune/requirements.txt
pip install langgraph langchain-core httpx reportlab fastapi uvicorn
```

### 2. Verify Unified Loaders & Spatial Pairing
```powershell
python data/test_loaders.py
```

### 3. Verify Specialist C (Multimodal Fusion & Segmentation)
```powershell
python finetune/specialist_c/eval_specialist_c.py
```

### 4. Test LangGraph Multi-Agent Orchestrator
```powershell
python orchestrator/test_orchestrator.py
```

### 5. Launch Full End-to-End Pipeline Demo
```powershell
python scripts/e2e_demo.py
```

### 6. Run the Web Application
```powershell
cd frontend
npm install
npm run dev
```

---

## 📊 Verification & Test Suite

| Component | Test Script | Status | Metrics / Results |
|---|---|---|---|
| **Data Loaders** | `data/test_loaders.py` | ✅ **PASS** | 100% compliant unified schema across 4 datasets |
| **Spatial Alignment** | `data/pairing.py` | ✅ **PASS** | Successfully detects CRS & resolution mismatch |
| **Specialist C** | `finetune/specialist_c/eval_specialist_c.py` | ✅ **PASS** | 256-d embedding, 9.38% change detected, RGBA overlay |
| **LangGraph Orchestrator** | `orchestrator/test_orchestrator.py` | ✅ **PASS** | Routes VQA, Change-VQA, and Fusion with trace |
| **n8n Webhook** | `n8n/test_n8n.py` | ✅ **PASS** | Validated 3-node workflow and retry logic |
| **PDF Report Generator** | `reports/generate.py` | ✅ **PASS** | Generates auditable executive PDF report |
| **Frontend UI** | `npm run build` | ✅ **PASS** | Production SPA compiled in 3.16s |
| **Full Integration Demo** | `scripts/e2e_demo.py` | ✅ **PASS** | All 6 pipeline stages executed in 6.5s |

---

## 🏆 SIH 2026 Alignment (Problem Statement 26167)
- **Sensor Agnostic**: Validates input by resolution, band count, and EPSG coordinate reference systems rather than hardcoded sensor names.
- **Auditable AI**: Every response produces an auditable execution trace with model names, timestamps, latency, and confidence scoring.
- **Hardware Optimized**: Designed for single consumer/workstation GPUs (RTX A2000 12GB) utilizing 4-bit QLoRA and modular dual-branch architectures.
  hi lol 
