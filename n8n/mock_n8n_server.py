"""
SatQuery AI — n8n Webhook Proxy & Execution Simulator
Port: 5678 (Default n8n Webhook Port)
SIH 2026 (PS 26167)

Simulates the exact runtime execution of the n8n workflow:
1. Receives POST /satquery from frontend
2. Dispatches HTTP POST to Orchestrator /query (http://127.0.0.1:8000/query)
3. Implements automatic single retry on 5xx error
4. Returns orchestrator response unchanged to client
"""

import os
import sys
import time
import logging
from typing import Dict, Any
import requests
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("n8nWebhookServer")

app = FastAPI(title="SatQuery AI n8n Webhook Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://127.0.0.1:8000/query")


@app.get("/health")
def health():
    return {"status": "healthy", "service": "n8n_webhook_proxy", "port": 5678}


@app.post("/satquery")
async def webhook_satquery(request: Request):
    payload = await request.json()
    logger.info("Received request on n8n webhook (POST /satquery). Forwarding to orchestrator...")

    max_retries = 2
    for attempt in range(max_retries):
        try:
            resp = requests.post(ORCHESTRATOR_URL, json=payload, timeout=35)
            if resp.status_code >= 500:
                logger.warning(f"Orchestrator returned {resp.status_code} (attempt {attempt+1}/{max_retries}). Retrying...")
                if attempt == max_retries - 1:
                    return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
                time.sleep(1.0)
                continue
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
        except Exception as e:
            logger.warning(f"Connection error to orchestrator (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=502, detail=f"Failed to reach orchestrator after retry: {str(e)}")
            time.sleep(1.0)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5678, reload=False)
