"""
SatQuery AI — Unified Production System Launcher
SIH 2026 (PS 26167)

Launches all SatQuery multi-agent microservices in production mode:
1. Specialist A Microservice (VQA / Captioning) on Port 8001
2. Specialist B Microservice (Change-VQA) on Port 8002
3. Multi-Agent Orchestrator API (LangGraph) on Port 8000
4. n8n Webhook Proxy Server on Port 5678
5. SatQuery React Web App Frontend on Port 3000
"""

import os
import sys
import time
import subprocess
import webbrowser

def start_service(name, cmd, cwd):
    print(f"[*] Starting {name}...")
    proc = subprocess.Popen(cmd, cwd=cwd, shell=True)
    return proc

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    root_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(root_dir, ".venv", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    print("=" * 65)
    print("      SATQUERY AI -- AGENTIC EARTH OBSERVATION SYSTEM")
    print("              SIH 2026 -- Problem Statement ID 26167")
    print("=" * 65)

    processes = []

    try:
        # 1. Specialist A (Port 8001)
        proc_a = start_service(
            "Specialist A (VQA Microservice :8001)",
            f'"{venv_python}" services/vqa_service/main.py',
            root_dir
        )
        processes.append(proc_a)

        # 2. Specialist B (Port 8002)
        proc_b = start_service(
            "Specialist B (Change-VQA Microservice :8002)",
            f'"{venv_python}" services/changevqa_service/main.py',
            root_dir
        )
        processes.append(proc_b)

        # 3. Specialist C (Port 8003)
        proc_c = start_service(
            "Specialist C (Optical-SAR Fusion :8003)",
            f'"{venv_python}" services/fusion_service/main.py',
            root_dir
        )
        processes.append(proc_c)

        # 4. LangGraph Orchestrator (Port 8000)
        proc_orch = start_service(
            "Multi-Agent Orchestrator API (:8000)",
            f'"{venv_python}" orchestrator/main.py',
            root_dir
        )
        processes.append(proc_orch)

        # 4. n8n Webhook Proxy (Port 5678)
        proc_n8n = start_service(
            "n8n Webhook Proxy Server (:5678)",
            f'"{venv_python}" n8n/mock_n8n_server.py',
            root_dir
        )
        processes.append(proc_n8n)

        # 5. React Frontend (Port 3000)
        frontend_dir = os.path.join(root_dir, "frontend")
        if os.path.exists(frontend_dir):
            proc_ui = start_service(
                "SatQuery Web App UI (:3000)",
                "npm run dev",
                frontend_dir
            )
            processes.append(proc_ui)

        print("\n" + "=" * 65)
        print("[+] All SatQuery Microservices Successfully Activated!")
        print("   - Web Chat Interface: http://localhost:3000")
        print("   - n8n Webhook Proxy: http://127.0.0.1:5678/satquery")
        print("   - Orchestrator API: http://127.0.0.1:8000/query")
        print("=" * 65)
        print("Press Ctrl+C to stop all services.\n")

        time.sleep(3)
        webbrowser.open("http://localhost:3000")

        for proc in processes:
            proc.wait()

    except KeyboardInterrupt:
        print("\nShutting down SatQuery microservices...")
        for proc in processes:
            proc.terminate()

if __name__ == "__main__":
    main()
