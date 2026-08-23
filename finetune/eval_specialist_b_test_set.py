"""
SatQuery AI — Specialist B Evaluation & Benchmarking Script
SIH 2026 (PS 26167)

Evaluates Specialist B (Change-VQA) on C:\\Users\\Admitrator\\Desktop\\specialist_b_test_set
"""

import os
import json
import logging
import urllib.request
import base64

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvalSpecialistB")

TEST_SET_DIR = r"C:\Users\Admitrator\Desktop\specialist_b_test_set"
JSON_PATH = os.path.join(TEST_SET_DIR, "questions_answers.json")
IMG_DIR = os.path.join(TEST_SET_DIR, "images")

ORCHESTRATOR_URL = "http://127.0.0.1:8000/query"

def encode_img_to_b64(path):
    with open(path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"

def clean_question(text):
    text = text.replace("<image>", "").replace("[change_vqa]", "").strip()
    return text

def evaluate_specialist_b():
    if not os.path.exists(JSON_PATH):
        logger.error(f"Questions/Answers file missing at {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)

    logger.info(f"Loaded {len(samples)} bi-temporal test samples from {TEST_SET_DIR}")
    
    total_questions = 0
    correct_questions = 0
    sample_reports = []

    for i, sample in enumerate(samples):
        t1_filename = sample["time1_image"]
        t2_filename = sample["time2_image"]
        
        path_t1 = os.path.join(IMG_DIR, t1_filename)
        path_t2 = os.path.join(IMG_DIR, t2_filename)

        if not os.path.exists(path_t1) or not os.path.exists(path_t2):
            logger.warning(f"Missing image pair: {t1_filename}, {t2_filename}")
            continue

        b64_t1 = encode_img_to_b64(path_t1)
        b64_t2 = encode_img_to_b64(path_t2)

        convs = sample.get("conversations", [])
        logger.info(f"\n--- Sample {i:02d} ({t1_filename} vs {t2_filename}) ---")
        
        q_counter = 0
        for j in range(len(convs) - 1):
            if convs[j]["from"] == "human":
                q_counter += 1
                question = clean_question(convs[j]["value"])
                ground_truth = convs[j+1]["value"].strip().lower()

                # Send query to Orchestrator API with both images
                payload = {
                    "images": [b64_t1, b64_t2],
                    "query": question
                }

                try:
                    req = urllib.request.Request(
                        ORCHESTRATOR_URL,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=15) as res:
                        resp_data = json.loads(res.read().decode("utf-8"))
                        pred_raw = resp_data.get("answer", "").strip()
                except Exception as e:
                    pred_raw = f"Error: {e}"

                pred_clean = pred_raw.lower()
                gt_words = [w for w in ground_truth.split() if len(w) > 2]
                is_match = (pred_clean == ground_truth) or (ground_truth in pred_clean) or (pred_clean in ground_truth) or any(w in pred_clean for w in gt_words)
                
                total_questions += 1
                if is_match:
                    correct_questions += 1

                status_mark = "✓ MATCH" if is_match else "✗ MISMATCH"
                logger.info(f" Q{q_counter}: '{question}'")
                logger.info(f"   Ground Truth : {ground_truth}")
                logger.info(f"   Prediction   : {pred_raw} [{status_mark}]")

    accuracy = (correct_questions / max(1, total_questions)) * 100
    logger.info("\n" + "="*70)
    logger.info(f"SPECIALIST B EVALUATION SUMMARY:")
    logger.info(f" Accuracy: {accuracy:.2f}% ({correct_questions}/{total_questions})")
    logger.info("="*70)

if __name__ == "__main__":
    evaluate_specialist_b()
