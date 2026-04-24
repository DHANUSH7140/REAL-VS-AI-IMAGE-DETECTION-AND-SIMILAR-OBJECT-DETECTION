"""
VisionProbe — main FastAPI application.
AI vs Real Image Detector.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure backend package is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_cache import ModelCache
from detector.router import router as detector_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("visionprobe")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models on startup, clean up on shutdown."""
    cache = ModelCache.get_instance()
    await cache.load_all_models()
    yield
    logger.info("Shutting down VisionProbe.")


app = FastAPI(
    title="VisionProbe API",
    version="2.0.0",
    description="AI Image Detection",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(detector_router, prefix="/api", tags=["detector"])


@app.get("/health")
async def health():
    """Health check endpoint — reports model status and device."""
    cache = ModelCache.get_instance()
    return {
        "status": "ok",
        "models_loaded": cache.all_loaded,
        "device": str(cache.device),
        "modules": ["detector"],
        "model_status": {
            "clip": getattr(cache, "clip_available", False),
            "efficientnet": getattr(cache, "effnet_available", False),
            "xgboost": getattr(cache, "xgb_available", False),
            "fft": True,
            "srm": True,
        },
    }


# Determine frontend path — mount AFTER all API routes
_frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="static")
else:
    logger.warning(f"Frontend directory not found at {_frontend_dir}")

