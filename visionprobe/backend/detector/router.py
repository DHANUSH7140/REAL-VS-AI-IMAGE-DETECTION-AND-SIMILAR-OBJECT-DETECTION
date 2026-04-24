"""
Detector API router — endpoints for AI vs Real image detection.
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from detector.inference import run_full_pipeline
from model_cache import ModelCache

logger = logging.getLogger("visionprobe.detector.router")
router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


@router.post("/analyze")
async def analyze_image(
    image: UploadFile = File(None),
    url: Optional[str] = Form(None),
    include_gradcam: Optional[bool] = Form(True),
    include_shap: Optional[bool] = Form(True),
):
    """
    Analyze a single image to determine if it is AI-generated or real.
    Accepts either a file upload or a URL.
    """
    image_bytes = None

    if image and image.filename:
        content = await image.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(400, "File too large. Max 20MB.")
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(400, "File must be an image (JPEG/PNG/WEBP).")
        image_bytes = content
    elif url:
        import requests as req
        try:
            resp = req.get(url, timeout=15, stream=True)
            resp.raise_for_status()
            image_bytes = resp.content
            if len(image_bytes) > MAX_FILE_SIZE:
                raise HTTPException(400, "Image from URL too large. Max 20MB.")
        except Exception as e:
            raise HTTPException(400, f"Could not fetch image from URL: {str(e)}")
    else:
        raise HTTPException(400, "Provide either an image file or a URL.")

    try:
        result = await run_full_pipeline(
            image_bytes,
            include_gradcam=include_gradcam,
            include_shap=include_shap,
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@router.post("/analyze/batch")
async def analyze_batch(
    images: list[UploadFile] = File(...),
    include_gradcam: Optional[bool] = Form(False),
    include_shap: Optional[bool] = Form(False),
):
    """Analyze up to 10 images in a batch."""
    if len(images) > 10:
        raise HTTPException(400, "Maximum 10 images per batch.")

    results = []
    for img in images:
        content = await img.read()
        if len(content) > MAX_FILE_SIZE:
            results.append({"error": f"{img.filename}: File too large."})
            continue
        try:
            result = await run_full_pipeline(content, include_gradcam, include_shap)
            results.append(result)
        except Exception as e:
            results.append({"error": f"{img.filename}: {str(e)}"})

    return JSONResponse(content={"results": results, "count": len(results)})


@router.get("/model-info")
async def model_info():
    """Returns information about each model branch and its capabilities."""
    cache = ModelCache.get_instance()
    models = [
        {
            "name": "OpenCLIP ViT-L/14",
            "architecture": "Vision Transformer (ViT-L/14) with contrastive language-image pretraining (OpenCLIP)",
            "what_it_detects": "Semantic inconsistencies — compares image against AI-generation vs real-photograph text descriptions using 768-dim embeddings",
            "available": cache.clip_available,
            "parameters": "428M",
            "branch": "Semantic Features",
        },
        {
            "name": "EfficientNetV2-Large",
            "architecture": "Compound-scaled CNN with fused MBConv blocks, intermediate features via global average pooling (1280-dim)",
            "what_it_detects": "Low-level pixel artifacts, compression patterns, and texture anomalies that differ between real and AI images",
            "available": cache.effnet_available,
            "parameters": "118M",
            "branch": "Artifact Detection",
        },
        {
            "name": "FFT Frequency Analysis",
            "architecture": "2D Fast Fourier Transform with azimuthal averaging",
            "what_it_detects": "Mid-frequency periodicity from diffusion model upsampling; spectral flatness anomalies from GANs; high-frequency energy patterns",
            "available": True,
            "branch": "Frequency Features",
        },
        {
            "name": "SRM Noise Analysis",
            "architecture": "Spatial Rich Model with 3 residual filters × 3 channels, statistical feature extraction (mean, std, kurtosis, skewness, entropy)",
            "what_it_detects": "Camera sensor noise signatures present in real photos but absent in AI-generated images",
            "available": True,
            "branch": "Noise Residuals",
        },
    ]
    return {
        "models": models,
        "classifier": "XGBoost with Isotonic/Platt calibration",
        "classifier_available": cache.xgb_available,
        "pipeline": "multi-branch-xgboost",
    }
