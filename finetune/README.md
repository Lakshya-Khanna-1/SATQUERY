# /finetune — Specialist Model Fine-Tuning Layer

This directory contains QLoRA (4-bit peft + bitsandbytes) fine-tuning scripts and pipelines for SatQuery AI's specialist models:

- **`specialist_a/`**: Single-image specialist for VQA, captioning, and text-guided region grounding (Base: `Qwen/Qwen2.5-VL-3B-Instruct`, Data: `VRSBench` + `RSVQA-LR`).
- **`specialist_b/`**: Bi-temporal specialist for Change-VQA across pre- and post-change satellite pairs (Base: `Qwen/Qwen2.5-VL-3B-Instruct`, Data: `CDVQA` + `SECOND`).
- **`specialist_c/`**: Optical–SAR multimodal fusion and change-mask specialist (Dual-branch ResNet-50 / ViT-S encoder + lightweight U-Net segmentation head, Data: `BigEarthNet-MM` + `SECOND` masks).

> **Note**: Ollama does not train models. Models are trained via HuggingFace PEFT/bitsandbytes, merged into base weights, converted to GGUF, and served via Ollama.
