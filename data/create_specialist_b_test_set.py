"""
SatQuery AI — Specialist B Test Set Generator (Bi-Temporal Change Detection)
Creates 10 bi-temporal image pairs (Time 1 and Time 2) along with comprehensive questions & answers.
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT_DIR = r"C:\Users\Admitrator\Desktop\specialist_b_test_set"
IMG_DIR = os.path.join(OUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

def create_base_satellite_canvas(seed=42, width=512, height=512, bg_color=(80, 120, 60)):
    """Generate textured satellite image canvas."""
    np.random.seed(seed)
    # Texture noise
    arr = np.random.randint(-15, 15, (height, width, 3), dtype=np.int16)
    bg = np.full((height, width, 3), bg_color, dtype=np.int16)
    combined = np.clip(bg + arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(combined)
    return img

def generate_10_sample_pairs():
    samples_data = []

    # -------------------------------------------------------------
    # Sample 00: Urban Building Development
    # -------------------------------------------------------------
    img_t1 = create_base_satellite_canvas(seed=100, bg_color=(75, 130, 55)) # Green field
    draw1 = ImageDraw.Draw(img_t1)
    draw1.line([(0, 250), (512, 270)], fill=(120, 110, 90), width=12) # Dirt track
    img_t1.save(os.path.join(IMG_DIR, "sample_00_time1.png"))

    img_t2 = img_t1.copy()
    draw2 = ImageDraw.Draw(img_t2)
    draw2.line([(0, 250), (512, 270)], fill=(70, 70, 75), width=24) # Paved road
    # Draw new buildings
    draw2.rectangle([100, 100, 220, 200], fill=(210, 205, 195), outline=(50, 50, 50), width=3)
    draw2.rectangle([260, 120, 380, 210], fill=(200, 90, 80), outline=(50, 50, 50), width=3)
    draw2.rectangle([150, 320, 280, 420], fill=(220, 215, 200), outline=(50, 50, 50), width=3)
    img_t2.save(os.path.join(IMG_DIR, "sample_00_time2.png"))

    samples_data.append({
        "sample_index": 0,
        "time1_image": "sample_00_time1.png",
        "time2_image": "sample_00_time2.png",
        "task_type": "change_vqa",
        "conversations": [
            {"from": "human", "value": "<image>\n[change_vqa] Did new building construction occur between Time 1 and Time 2?"},
            {"from": "gpt", "value": "yes"},
            {"from": "human", "value": "<image>\n[change_vqa] What primary land-use change took place between Time 1 and Time 2?"},
            {"from": "gpt", "value": "building construction"},
            {"from": "human", "value": "<image>\n[change_vqa] Did the area of natural vegetation increase or decrease?"},
            {"from": "gpt", "value": "decrease"}
        ]
    })

    # -------------------------------------------------------------
    # Sample 01: Forest Deforestation
    # -------------------------------------------------------------
    img_t1 = create_base_satellite_canvas(seed=101, bg_color=(35, 95, 30)) # Dense forest
    img_t1.save(os.path.join(IMG_DIR, "sample_01_time1.png"))

    img_t2 = img_t1.copy()
    draw2 = ImageDraw.Draw(img_t2)
    # Cleared brown patches
    draw2.polygon([(100, 100), (350, 120), (380, 300), (80, 280)], fill=(160, 125, 80))
    draw2.line([(100, 200), (512, 350)], fill=(130, 100, 65), width=10) # Logging track
    img_t2.save(os.path.join(IMG_DIR, "sample_01_time2.png"))

    samples_data.append({
        "sample_index": 1,
        "time1_image": "sample_01_time1.png",
        "time2_image": "sample_01_time2.png",
        "task_type": "change_vqa",
        "conversations": [
            {"from": "human", "value": "<image>\n[change_vqa] Have the areas of trees changed between Time 1 and Time 2?"},
            {"from": "gpt", "value": "yes"},
            {"from": "human", "value": "<image>\n[change_vqa] What type of environmental change occurred between Time 1 and Time 2?"},
            {"from": "gpt", "value": "deforestation"},
            {"from": "human", "value": "<image>\n[change_vqa] Did tree canopy coverage increase or decrease?"},
            {"from": "gpt", "value": "decrease"}
        ]
    })

    # -------------------------------------------------------------
    # Sample 02: Highway Construction
    # -------------------------------------------------------------
    img_t1 = create_base_satellite_canvas(seed=102, bg_color=(120, 150, 70))
    draw1 = ImageDraw.Draw(img_t1)
    # Patchwork of fields
    draw1.rectangle([50, 50, 250, 250], fill=(130, 160, 65))
    draw1.rectangle([260, 50, 480, 250], fill=(110, 140, 80))
    img_t1.save(os.path.join(IMG_DIR, "sample_02_time1.png"))

    img_t2 = img_t1.copy()
    draw2 = ImageDraw.Draw(img_t2)
    # Dual carriageway highway
    draw2.line([(0, 250), (512, 250)], fill=(50, 50, 55), width=36)
    draw2.line([(0, 250), (512, 250)], fill=(240, 240, 240), width=2) # Median stripe
    img_t2.save(os.path.join(IMG_DIR, "sample_02_time2.png"))

    samples_data.append({
        "sample_index": 2,
        "time1_image": "sample_02_time1.png",
        "time2_image": "sample_02_time2.png",
        "task_type": "change_vqa",
        "conversations": [
            {"from": "human", "value": "<image>\n[change_vqa] Was a new road constructed between Time 1 and Time 2?"},
            {"from": "gpt", "value": "yes"},
            {"from": "human", "value": "<image>\n[change_vqa] What transportation infrastructure was added?"},
            {"from": "gpt", "value": "highway"},
            {"from": "human", "value": "<image>\n[change_vqa] Has agricultural land area changed between Time 1 and Time 2?"},
            {"from": "gpt", "value": "yes"}
        ]
    })

    # -------------------------------------------------------------
    # Sample 03: Reservoir Water Expansion
    # -------------------------------------------------------------
    img_t1 = create_base_satellite_canvas(seed=103, bg_color=(140, 130, 90))
    draw1 = ImageDraw.Draw(img_t1)
    # Narrow stream
    draw1.line([(250, 0), (270, 512)], fill=(40, 80, 130), width=15)
    img_t1.save(os.path.join(IMG_DIR, "sample_03_time1.png"))

    img_t2 = img_t1.copy()
    draw2 = ImageDraw.Draw(img_t2)
    # Large expanded lake/reservoir
    draw2.ellipse([100, 100, 412, 412], fill=(30, 70, 125))
    img_t2.save(os.path.join(IMG_DIR, "sample_03_time2.png"))

    samples_data.append({
        "sample_index": 3,
        "time1_image": "sample_03_time1.png",
        "time2_image": "sample_03_time2.png",
        "task_type": "change_vqa",
        "conversations": [
            {"from": "human", "value": "<image>\n[change_vqa] Did the water body area change between Time 1 and Time 2?"},
            {"from": "gpt", "value": "yes"},
            {"from": "human", "value": "<image>\n[change_vqa] Did the surface water coverage increase or decrease?"},
            {"from": "gpt", "value": "increase"},
            {"from": "human", "value": "<image>\n[change_vqa] Is there evidence of reservoir expansion?"},
            {"from": "gpt", "value": "yes"}
        ]
    })

    # -------------------------------------------------------------
    # Sample 04: Industrial Storage Tank Addition
    # -------------------------------------------------------------
    img_t1 = create_base_satellite_canvas(seed=104, bg_color=(150, 145, 135))
    img_t1.save(os.path.join(IMG_DIR, "sample_04_time1.png"))

    img_t2 = img_t1.copy()
    draw2 = ImageDraw.Draw(img_t2)
    # 3 White circular storage tanks
    draw2.ellipse([100, 150, 220, 270], fill=(245, 245, 250), outline=(80, 80, 80), width=4)
    draw2.ellipse([260, 150, 380, 270], fill=(245, 245, 250), outline=(80, 80, 80), width=4)
    draw2.ellipse([180, 300, 300, 420], fill=(245, 245, 250), outline=(80, 80, 80), width=4)
    img_t2.save(os.path.join(IMG_DIR, "sample_04_time2.png"))

    samples_data.append({
        "sample_index": 4,
        "time1_image": "sample_04_time1.png",
        "time2_image": "sample_04_time2.png",
        "task_type": "change_vqa",
        "conversations": [
            {"from": "human", "value": "<image>\n[change_vqa] Were new structures added to the site between Time 1 and Time 2?"},
            {"from": "gpt", "value": "yes"},
            {"from": "human", "value": "<image>\n[change_vqa] How many new storage tanks were constructed?"},
            {"from": "gpt", "value": "3"},
            {"from": "human", "value": "<image>\n[change_vqa] What color are the newly added storage tanks?"},
            {"from": "gpt", "value": "white"}
        ]
    })

    # -------------------------------------------------------------
    # Sample 05: Agricultural Crop Growth
    # -------------------------------------------------------------
    img_t1 = create_base_satellite_canvas(seed=105, bg_color=(160, 120, 80)) # Ploughed soil
    img_t1.save(os.path.join(IMG_DIR, "sample_05_time1.png"))

    img_t2 = create_base_satellite_canvas(seed=105, bg_color=(50, 160, 40)) # Lush green crops
    img_t2.save(os.path.join(IMG_DIR, "sample_05_time2.png"))

    samples_data.append({
        "sample_index": 5,
        "time1_image": "sample_05_time1.png",
        "time2_image": "sample_05_time2.png",
        "task_type": "change_vqa",
        "conversations": [
            {"from": "human", "value": "<image>\n[change_vqa] Has the vegetation status of the fields changed between Time 1 and Time 2?"},
            {"from": "gpt", "value": "yes"},
            {"from": "human", "value": "<image>\n[change_vqa] Did green crop vegetation coverage increase or decrease?"},
            {"from": "gpt", "value": "increase"},
            {"from": "human", "value": "<image>\n[change_vqa] What type of seasonal land change is depicted?"},
            {"from": "gpt", "value": "crop growth"}
        ]
    })

    # -------------------------------------------------------------
    # Sample 06: Solar Power Farm Construction
    # -------------------------------------------------------------
    img_t1 = create_base_satellite_canvas(seed=106, bg_color=(200, 170, 120)) # Desert terrain
    img_t1.save(os.path.join(IMG_DIR, "sample_06_time1.png"))

    img_t2 = img_t1.copy()
    draw2 = ImageDraw.Draw(img_t2)
    # Solar panel grid (dark blue panels)
    for r in range(4):
        for c in range(5):
            x1 = 60 + c * 80
            y1 = 60 + r * 90
            draw2.rectangle([x1, y1, x1 + 65, y1 + 70], fill=(25, 45, 90), outline=(200, 200, 220), width=2)
    img_t2.save(os.path.join(IMG_DIR, "sample_06_time2.png"))

    samples_data.append({
        "sample_index": 6,
        "time1_image": "sample_06_time1.png",
        "time2_image": "sample_06_time2.png",
        "task_type": "change_vqa",
        "conversations": [
            {"from": "human", "value": "<image>\n[change_vqa] Has a renewable energy facility been constructed between Time 1 and Time 2?"},
            {"from": "gpt", "value": "yes"},
            {"from": "human", "value": "<image>\n[change_vqa] What feature was added to the arid terrain?"},
            {"from": "gpt", "value": "solar panels"},
            {"from": "human", "value": "<image>\n[change_vqa] Are the newly installed panel arrays arranged in a grid pattern?"},
            {"from": "gpt", "value": "yes"}
        ]
    })

    # -------------------------------------------------------------
    # Sample 07: Coastal Port Extension
    # -------------------------------------------------------------
    img_t1 = create_base_satellite_canvas(seed=107, bg_color=(180, 160, 130)) # Coastline
    draw1 = ImageDraw.Draw(img_t1)
    draw1.rectangle([250, 0, 512, 512], fill=(35, 75, 120)) # Sea
    draw1.rectangle([200, 200, 260, 280], fill=(130, 130, 130)) # Small jetty
    img_t1.save(os.path.join(IMG_DIR, "sample_07_time1.png"))

    img_t2 = img_t1.copy()
    draw2 = ImageDraw.Draw(img_t2)
    # Extended concrete pier
    draw2.rectangle([200, 150, 420, 330], fill=(150, 150, 155), outline=(60, 60, 60), width=3)
    # Containers
    for i in range(6):
        draw2.rectangle([220 + i*30, 170, 240 + i*30, 230], fill=(190, 50, 40))
        draw2.rectangle([220 + i*30, 250, 240 + i*30, 310], fill=(40, 90, 180))
    img_t2.save(os.path.join(IMG_DIR, "sample_07_time2.png"))

    samples_data.append({
        "sample_index": 7,
        "time1_image": "sample_07_time1.png",
        "time2_image": "sample_07_time2.png",
        "task_type": "change_vqa",
        "conversations": [
            {"from": "human", "value": "<image>\n[change_vqa] Was port infrastructure expanded between Time 1 and Time 2?"},
            {"from": "gpt", "value": "yes"},
            {"from": "human", "value": "<image>\n[change_vqa] Did artificial shoreline structure area increase or decrease?"},
            {"from": "gpt", "value": "increase"},
            {"from": "human", "value": "<image>\n[change_vqa] Are shipping containers present in Time 2?"},
            {"from": "gpt", "value": "yes"}
        ]
    })

    # -------------------------------------------------------------
    # Sample 08: Flood Inundation
    # -------------------------------------------------------------
    img_t1 = create_base_satellite_canvas(seed=108, bg_color=(90, 140, 70))
    draw1 = ImageDraw.Draw(img_t1)
    draw1.line([(240, 0), (260, 512)], fill=(45, 85, 135), width=18) # River
    img_t1.save(os.path.join(IMG_DIR, "sample_08_time1.png"))

    img_t2 = img_t1.copy()
    draw2 = ImageDraw.Draw(img_t2)
    # Submerged floodwaters
    draw2.ellipse([50, 80, 460, 440], fill=(40, 75, 115))
    img_t2.save(os.path.join(IMG_DIR, "sample_08_time2.png"))

    samples_data.append({
        "sample_index": 8,
        "time1_image": "sample_08_time1.png",
        "time2_image": "sample_08_time2.png",
        "task_type": "change_vqa",
        "conversations": [
            {"from": "human", "value": "<image>\n[change_vqa] Did flooding occur between Time 1 and Time 2?"},
            {"from": "gpt", "value": "yes"},
            {"from": "human", "value": "<image>\n[change_vqa] Were surrounding agricultural fields submerged?"},
            {"from": "gpt", "value": "yes"},
            {"from": "human", "value": "<image>\n[change_vqa] Did total water area increase or decrease?"},
            {"from": "gpt", "value": "increase"}
        ]
    })

    # -------------------------------------------------------------
    # Sample 09: Airport Runway Extension
    # -------------------------------------------------------------
    img_t1 = create_base_satellite_canvas(seed=109, bg_color=(110, 135, 80))
    draw1 = ImageDraw.Draw(img_t1)
    draw1.line([(100, 250), (300, 250)], fill=(70, 70, 75), width=28) # Short runway
    img_t1.save(os.path.join(IMG_DIR, "sample_09_time1.png"))

    img_t2 = img_t1.copy()
    draw2 = ImageDraw.Draw(img_t2)
    # Extended runway + taxiway
    draw2.line([(100, 250), (480, 250)], fill=(70, 70, 75), width=28)
    draw2.line([(150, 180), (450, 180)], fill=(80, 80, 85), width=16) # Parallel taxiway
    img_t2.save(os.path.join(IMG_DIR, "sample_09_time2.png"))

    samples_data.append({
        "sample_index": 9,
        "time1_image": "sample_09_time1.png",
        "time2_image": "sample_09_time2.png",
        "task_type": "change_vqa",
        "conversations": [
            {"from": "human", "value": "<image>\n[change_vqa] Was the airport runway lengthened between Time 1 and Time 2?"},
            {"from": "gpt", "value": "yes"},
            {"from": "human", "value": "<image>\n[change_vqa] What aviation infrastructure was extended?"},
            {"from": "gpt", "value": "runway"},
            {"from": "human", "value": "<image>\n[change_vqa] Were new parallel taxiways constructed?"},
            {"from": "gpt", "value": "yes"}
        ]
    })

    # Save JSON dataset metadata
    json_path = os.path.join(OUT_DIR, "questions_answers.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples_data, f, indent=2)

    print(f"[SUCCESS] Created Specialist B Test Set at: {OUT_DIR}")
    print(f" - Images Folder: {IMG_DIR} (20 PNG images: 10 Time-1 and 10 Time-2 pairs)")
    print(f" - JSON File    : {json_path} (10 samples, 30 Change VQA questions & answers)")

if __name__ == "__main__":
    generate_10_sample_pairs()
