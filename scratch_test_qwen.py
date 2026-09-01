import os
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

base_path = "models/base/specialist_a_base"
adapter_path = "models/adapters/specialist_a"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(base_path)

print("Loading model on CUDA...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_path,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

if os.path.exists(adapter_path):
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

model.eval()
print("Specialist A model loaded successfully on GPU!")

# Test with Qwen VL utils
from qwen_vl_utils import process_vision_info

test_img = Image.new("RGB", (512, 512), (50, 50, 50))
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": test_img},
            {"type": "text", "text": "Describe this satellite image and any vehicles or aircraft visible."}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print("Model Output:\n", output_text)
