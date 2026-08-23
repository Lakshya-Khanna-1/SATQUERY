import io
import base64
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from pathlib import Path

base_path = "models/base/specialist_a_base"
adapter_path = "models/adapters/specialist_b"

processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
from peft import PeftModel
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

img_a_path = Path("specialist_b_real_testset/images/pair_00_before.png")
img_b_path = Path("specialist_b_real_testset/images/pair_00_after.png")
img_t1 = Image.open(img_a_path).convert("RGB")
img_t2 = Image.open(img_b_path).convert("RGB")

prompts = [
    "What changed between the two images?",
    "Answer concisely in 1-5 words: What changed between the two images?",
    "Identify the change from Image 1 to Image 2:",
    "Describe the difference between the before image and after image:",
    "Has any new road or building been constructed in Image 2?"
]

for p in prompts:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_t1},
                {"type": "image", "image": img_t2},
                {"type": "text", "text": p}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=48)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
    print(f"Prompt: {p}\n -> Output: {out}\n")
