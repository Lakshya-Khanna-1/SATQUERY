"""
SatQuery AI — Orchestrator API (FastAPI)
Port: 8000
SIH 2026 (PS 26167)

Exposes POST /query endpoint wrapping the LangGraph Multi-Agent State Machine.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestrator.graph import run_orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OrchestratorAPI")

app = FastAPI(
    title="SatQuery AI — Multi-Agent Orchestrator API",
    description="LangGraph orchestrator coordinating Specialist A (VQA/Caption), Specialist B (Change-VQA), and Specialist C (Fusion & Segmentation).",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language user query")
    images: List[str] = Field(..., description="List of 1 or 2 base64 encoded satellite images")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata (CRS, bounds, sensor)")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "satquery-orchestrator",
        "framework": "LangGraph",
        "port": 8000
    }


@app.post("/query")
def process_query(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    if not req.images or len(req.images) == 0:
        raise HTTPException(status_code=400, detail="At least one satellite image is required.")

    try:
        result = run_orchestrator(
            query=req.query,
            images=req.images,
            metadata=req.metadata
        )
        return result
    except Exception as e:
        logger.error(f"Orchestration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Orchestration error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
