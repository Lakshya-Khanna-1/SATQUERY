"""
SatQuery AI — Multi-Agent Orchestrator API (FastAPI)
Port: 8000
SIH 2026 (PS 26167)

Exposes POST /query endpoint wrapping the LangGraph Multi-Agent State Machine
and POST /generate-pdf for automatic report generation.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orchestrator.graph import run_orchestrator
from orchestrator.pdf_generator import generate_satquery_pdf

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
    query: Optional[str] = Field("Detailed geospatial observation analysis", description="Natural language user query")
    images: List[str] = Field(..., description="List of 1 or 2 base64 encoded satellite images")
    analysis_mode: Optional[str] = Field("quick", description="Analysis mode: 'quick' (1-2 sentences) or 'deep' (5+ lines)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata (CRS, bounds, sensor)")


class PDFRequest(BaseModel):
    images: List[str] = Field(..., description="List of base64 images")
    query: Optional[str] = Field("Automatic Comprehensive Geospatial PDF Intelligence Report", description="Query or report title")
    analysis_mode: Optional[str] = Field("deep", description="Mode for PDF report (defaults to deep)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CoordinatesRequest(BaseModel):
    lat: float = Field(..., description="Target Latitude")
    lon: float = Field(..., description="Target Longitude")
    radius_km: Optional[float] = Field(5.0, description="Bounding radius in kilometers")
    years_ago: Optional[int] = Field(5, description="Years ago for historical satellite comparison")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "satquery-orchestrator",
        "framework": "LangGraph",
        "port": 8000
    }


@app.post("/fetch-coords-satellite")
def fetch_coords_satellite(req: CoordinatesRequest):
    try:
        from data.satellite_fetcher import fetch_satellite_pair_by_coords
        data = fetch_satellite_pair_by_coords(
            lat=req.lat,
            lon=req.lon,
            radius_km=req.radius_km or 5.0,
            years_ago=req.years_ago or 5
        )
        return data
    except Exception as e:
        logger.error(f"Satellite coordinate fetch failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Satellite fetch error: {str(e)}")


@app.post("/query")
def process_query(req: QueryRequest):
    prompt = req.query or "Detailed geospatial observation analysis"
    if not req.images or len(req.images) == 0:
        raise HTTPException(status_code=400, detail="At least one satellite image is required.")

    try:
        result = run_orchestrator(
            query=prompt,
            images=req.images,
            analysis_mode=req.analysis_mode or "quick",
            metadata=req.metadata
        )
        return result
    except Exception as e:
        logger.error(f"Orchestration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Orchestration error: {str(e)}")


@app.post("/generate-pdf")
def generate_pdf_report(req: PDFRequest):
    if not req.images or len(req.images) == 0:
        raise HTTPException(status_code=400, detail="At least one satellite image is required to generate a PDF report.")

    prompt = req.query or "Automatic Comprehensive Geospatial PDF Intelligence Report"
    try:
        # Run deep orchestrator analysis for the PDF report
        result = run_orchestrator(
            query=prompt,
            images=req.images,
            analysis_mode="deep",
            metadata=req.metadata
        )

        pdf_bytes = generate_satquery_pdf(
            query=prompt,
            analysis_text=result["answer"],
            images_b64=req.images,
            task_type=result.get("task_type", "change_vqa"),
            confidence=result.get("confidence", 0.95),
            execution_trace=result.get("execution_trace")
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=SatQuery_Geospatial_Intelligence_Report.pdf"
            }
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")


# -----------------------------------------------------------------------------
# PERSISTENT LOCAL MEMORY DATABASE ENDPOINTS (SQLite)
# -----------------------------------------------------------------------------

@app.get("/sessions")
def list_sessions():
    try:
        from data.database import get_all_sessions
        return get_all_sessions()
    except Exception as e:
        logger.error(f"Failed to fetch sessions: {e}")
        return []


@app.post("/sessions")
def new_session(payload: Dict[str, Any] = None):
    try:
        from data.database import create_session
        title = payload.get("title", "New Analysis Session") if payload else "New Analysis Session"
        return create_session(title=title)
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
def fetch_session(session_id: str):
    try:
        from data.database import get_session_detail
        s = get_session_detail(session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")
        return s
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch session detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/messages")
def post_session_message(session_id: str, msg: Dict[str, Any]):
    try:
        from data.database import save_message
        saved = save_message(session_id, msg)
        return saved
    except Exception as e:
        logger.error(f"Failed to save message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/sessions/{session_id}/pin")
def pin_session(session_id: str):
    try:
        from data.database import toggle_pin_session
        res = toggle_pin_session(session_id)
        if not res:
            raise HTTPException(status_code=404, detail="Session not found")
        return res
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle pin on session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    try:
        from data.database import delete_session
        delete_session(session_id)
        return {"status": "success", "id": session_id}
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
