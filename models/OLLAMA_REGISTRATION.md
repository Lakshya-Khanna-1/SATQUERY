# SatQuery AI — Ollama Model Registration Guide

### 1. Build Merged Models
Run the export script to fuse LoRA adapters:
```bash
python models/export.py --all
```

### 2. Register with Ollama
Execute the following commands to create the local Ollama models:
```bash
# Register Specialist A (VQA/Captioning)
ollama create satquery-vqa -f models/Modelfile.vqa

# Register Specialist B (Change-VQA)
ollama create satquery-changevqa -f models/Modelfile.changevqa
```

### 3. Verify Local Execution
```bash
# Test Specialist A
ollama run satquery-vqa "Describe the landscape features visible in this satellite scene."

# Test Specialist B
ollama run satquery-changevqa "What is the primary land-cover change between these two observation dates?"
```
