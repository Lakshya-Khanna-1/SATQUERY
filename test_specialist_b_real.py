import json
import base64
import requests
from pathlib import Path
import time

from orchestrator.gemini_synthesizer import synthesize_with_gemini

TESTSET_DIR = Path("specialist_b_real_testset")
QA_FILE = TESTSET_DIR / "questions_answers.json"
IMAGES_DIR = TESTSET_DIR / "images"

CHANGEVQA_URL = "http://127.0.0.1:8002/analyze"

def run_test():
    if not QA_FILE.exists():
        print(f"Error: {QA_FILE} not found. Run get_specialist_b_real_images.py first.")
        return

    with open(QA_FILE, "r") as f:
        qa_data = json.load(f)

    results = []
    print(f"Starting Specialist B (Change-VQA) Real Bi-Temporal Test on {len(qa_data)} pairs...\n")

    for entry in qa_data:
        idx = entry["index"]
        img_a_path = IMAGES_DIR / entry["image_before"]
        img_b_path = IMAGES_DIR / entry["image_after"]
        question = entry["question"]
        gt_answer = entry["ground_truth_answer"]
        change_type = entry.get("type", "Unknown")

        print(f"==================================================")
        print(f"Test #{idx+1:02d} [{change_type}] — Pair: {entry['image_before']} <-> {entry['image_after']}")
        print(f"Question: {question}")
        print(f"Ground Truth Caption: {gt_answer}")

        # Read base64 for Gemini synthesis
        with open(img_a_path, "rb") as fa:
            b64_a = base64.b64encode(fa.read()).decode("utf-8")
        with open(img_b_path, "rb") as fb:
            b64_b = base64.b64encode(fb.read()).decode("utf-8")

        # Send request to changevqa_service
        try:
            with open(img_a_path, "rb") as fa, open(img_b_path, "rb") as fb:
                files = {
                    "image_time1": (img_a_path.name, fa, "image/png"),
                    "image_time2": (img_b_path.name, fb, "image/png"),
                }
                data = {"prompt": question}
                resp = requests.post(CHANGEVQA_URL, files=files, data=data, timeout=60)
                resp_json = resp.json()
                raw_answer = resp_json.get("answer", "Error: No answer returned")
        except Exception as e:
            raw_answer = f"Service Error: {str(e)}"
            resp_json = {"error": str(e)}

        print(f"Raw Model Answer: {raw_answer}")

        # Sleep to avoid Gemini 429 rate limit
        time.sleep(2.0)

        # Synthesize with Gemini Multimodal
        synth_answer = synthesize_with_gemini(
            query=question, 
            raw_answer=raw_answer, 
            task_type="changevqa",
            images_b64=[b64_a, b64_b]
        )
        print(f"Gemini Synthesized Answer: {synth_answer}")
        print(f"==================================================\n")

        results.append({
            "index": idx,
            "filename": entry["filename"],
            "type": change_type,
            "question": question,
            "ground_truth": gt_answer,
            "all_ground_truth_captions": entry.get("ground_truth_captions", []),
            "raw_model_answer": raw_answer,
            "gemini_synthesized_answer": synth_answer
        })

    # Save detailed evaluation report
    report_md = "# Specialist B Real Bi-Temporal Satellite Test Report (LEVIR-CC)\n\n"
    report_md += f"**Dataset:** LEVIR-CC Real Satellite Bi-Temporal Images (Texas, USA)\n"
    report_md += f"**Model:** Qwen2.5-VL-3B LoRA Adapter (`specialist_b`) + 4-Bit NF4 Quantization + Gemini 2.5 Flash Multimodal Synthesizer\n"
    report_md += f"**Total Test Cases:** {len(results)} (5 Construction/Change Pairs, 5 No-Change Pairs)\n\n"
    report_md += "## Summary of Results\n\n"
    report_md += "| Test # | Type | Image Pair | Ground Truth Caption | Raw Model Answer | Gemini Synthesized Natural Language Answer |\n"
    report_md += "|---|---|---|---|---|---|\n"

    for r in results:
        gt_short = r["ground_truth"].replace("\n", " ")
        raw_short = r["raw_model_answer"].replace("\n", " ")
        synth_short = r["gemini_synthesized_answer"].replace("\n", " ")
        report_md += f"| #{r['index']+1:02d} | **{r['type']}** | `{r['filename']}` | {gt_short} | `{raw_short}` | {synth_short} |\n"

    report_md += "\n\n## Detailed Pair-by-Pair Analysis\n\n"
    for r in results:
        report_md += f"### Pair #{r['index']+1:02d} (`{r['filename']}`) - [{r['type']}]\n"
        report_md += f"- **Question:** {r['question']}\n"
        report_md += f"- **Ground Truth Captions:**\n"
        for c in r["all_ground_truth_captions"]:
            report_md += f"  - *\"{c}\"*\n"
        report_md += f"- **Raw Model Answer:** `{r['raw_model_answer']}`\n"
        report_md += f"- **Gemini Natural Language Answer:** {r['gemini_synthesized_answer']}\n\n"

    report_path = Path("SPECIALIST_B_REAL_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Report written to {report_path.absolute()}")

if __name__ == "__main__":
    run_test()
