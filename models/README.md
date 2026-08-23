# /models — GGUF Export & Ollama Modelfiles

This directory manages GGUF export conversion via llama.cpp and Ollama serving definitions:

- `export.py`: Merges trained QLoRA adapters with base weights and converts to GGUF format.
- `Modelfile.vqa`: Ollama manifest for `satquery-vqa` (Specialist A).
- `Modelfile.changevqa`: Ollama manifest for `satquery-changevqa` (Specialist B).
- `serve_c.py`: Direct PyTorch FastAPI inference server for Specialist C (Optical-SAR Fusion & Segmentation).
