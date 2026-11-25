"""
SatQuery AI — Specialist C Microservice (Fusion & Change-Map Segmentation)
Port: 8003
"""

import os
import sys
import io
import base64
import logging
from typing import List, Dict, Any, Optional
import requests
import numpy as np

from PIL import Image
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.serve_c import app as serve_c_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FusionService")

app = serve_c_app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003, reload=False)
