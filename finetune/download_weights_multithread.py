"""
SatQuery AI — Resilient Multi-Threaded Chunk Downloader
SIH 2026 (PS 26167)
Downloads Hugging Face safetensors shards using parallel part files with auto-resume.
"""

import os
import sys
import time
import requests
import concurrent.futures
from typing import List, Tuple

BASE_URL = "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/resolve/main"
FILES = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors"
]
CHUNK_SIZE = 64 * 1024 * 1024  # 64MB chunks
MAX_WORKERS = 8


def get_file_size_and_final_url(filename: str) -> Tuple[int, str]:
    url = f"{BASE_URL}/{filename}"
    headers = {"User-Agent": "SatQuery-Downloader/1.0"}
    resp = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
    resp.raise_for_status()
    total_size = int(resp.headers.get("content-length", 0))
    final_url = resp.url
    return total_size, final_url


def download_chunk(url: str, part_path: str, start_byte: int, end_byte: int, chunk_idx: int, expected_size: int, retries: int = 5):
    if os.path.exists(part_path) and os.path.getsize(part_path) == expected_size:
        return chunk_idx, expected_size, True  # Already cached

    headers = {
        "Range": f"bytes={start_byte}-{end_byte}",
        "User-Agent": "SatQuery-Downloader/1.0"
    }
    for attempt in range(retries):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(part_path + ".tmp", "wb") as f:
                    for data in resp.iter_content(chunk_size=1024 * 1024):
                        if data:
                            f.write(data)
            os.replace(part_path + ".tmp", part_path)
            return chunk_idx, expected_size, False
        except Exception as e:
            if attempt == retries - 1:
                print(f"[ERROR] Chunk {chunk_idx} failed: {e}", flush=True)
                raise e
            time.sleep(1.5 ** attempt)


def download_file_multithreaded(filename: str, target_dir: str):
    target_path = os.path.join(target_dir, filename)
    parts_dir = os.path.join(target_dir, f".parts_{filename}")
    os.makedirs(parts_dir, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"[INFO] Resolving metadata for {filename}...", flush=True)
    total_size, final_url = get_file_size_and_final_url(filename)
    print(f"[INFO] File: {filename} | Size: {total_size / (1024*1024):.2f} MB", flush=True)

    if os.path.exists(target_path) and os.path.getsize(target_path) == total_size:
        print(f"[INFO] {filename} is already complete and verified. Skipping.", flush=True)
        return

    # Build chunk list
    chunks = []
    chunk_idx = 0
    for start in range(0, total_size, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE - 1, total_size - 1)
        expected = end - start + 1
        part_path = os.path.join(parts_dir, f"part_{chunk_idx:05d}.dat")
        chunks.append((final_url, part_path, start, end, chunk_idx, expected))
        chunk_idx += 1

    total_chunks = len(chunks)
    print(f"[INFO] Spawning {MAX_WORKERS} workers for {total_chunks} parts ({CHUNK_SIZE//(1024*1024)}MB each)...", flush=True)

    completed = 0
    downloaded_bytes = 0
    start_time = time.time()
    last_print = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_chunk, *c): c[4] for c in chunks
        }

        for future in concurrent.futures.as_completed(futures):
            idx, size, from_cache = future.result()
            completed += 1
            downloaded_bytes += size
            now = time.time()
            if now - last_print > 2.0 or completed == total_chunks:
                elapsed = max(0.1, now - start_time)
                speed_mb = (downloaded_bytes / (1024 * 1024)) / elapsed
                pct = (completed / total_chunks) * 100
                rem_mb = (total_size - (completed * CHUNK_SIZE)) / (1024 * 1024)
                eta = max(0, rem_mb / max(0.01, speed_mb))
                print(f"[PROGRESS] {completed}/{total_chunks} parts ({pct:.1f}%) | Speed: {speed_mb:.2f} MB/s | ETA: {eta:.0f}s", flush=True)
                last_print = now

    print(f"[INFO] All {total_chunks} parts downloaded! Stitching into {filename}...", flush=True)
    with open(target_path + ".tmp", "wb") as outfile:
        for c in chunks:
            part_path = c[1]
            with open(part_path, "rb") as infile:
                while True:
                    buf = infile.read(4 * 1024 * 1024)
                    if not buf:
                        break
                    outfile.write(buf)

    os.replace(target_path + ".tmp", target_path)
    
    # Cleanup parts
    import shutil
    shutil.rmtree(parts_dir, ignore_errors=True)

    total_elapsed = time.time() - start_time
    print(f"[SUCCESS] {filename} successfully assembled ({os.path.getsize(target_path)/(1024*1024):.2f} MB) in {total_elapsed:.1f}s!", flush=True)


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "models/base/specialist_a_base"
    os.makedirs(target_dir, exist_ok=True)
    for fn in FILES:
        download_file_multithreaded(fn, target_dir)
