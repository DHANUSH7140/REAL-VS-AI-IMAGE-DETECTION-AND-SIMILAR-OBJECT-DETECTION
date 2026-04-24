"""
api/routes.py — Flask Blueprint with all application routes.

Routes:
    GET  /                   → Web UI
    POST /predict            → UI prediction (single model or ensemble)
    POST /predict/combined   → YOLO + classifier pipeline
    POST /predict/similar    → ROI similarity search
    POST /predict/patch      → Patch-based analysis
    POST /api/predict        → REST API endpoint
    POST /api/find_similar   → REST similarity API
    POST /api/patch_analysis → REST patch analysis API
    GET  /api/health         → System health check
    GET  /api/history        → Recent prediction history
"""

import os
import uuid
import time
import psutil
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify

from config import (UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MAX_HISTORY_ITEMS,
                    MODEL_CONFIGS, PATCH_SIZE, PATCH_STRIDE)
from models.loader import model_manager
from services.predictor import predict_single, predict_all
from services.ensemble import weighted_voting
from services.gradcam import generate_gradcam
from services.fft_features import extract_fft_features

from services.patch_analyzer import analyze_patches
from services.meta_ensemble import meta_predict, is_trained as meta_is_trained
from services.pipeline_executor import execute_parallel, time_execution
from services.explanation_engine import generate_explanation
from utils.logger import setup_logger

logger = setup_logger("api.routes")

_start_time = time.time()

main_bp = Blueprint("main", __name__)

# ──────────────────────────── HISTORY ───────────────────────────
_history = []


def _add_to_history(entry: dict):
    """Add a prediction result to the in-memory history."""
    entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _history.insert(0, entry)
    if len(_history) > MAX_HISTORY_ITEMS:
        _history.pop()


# ──────────────────────────── HELPERS ───────────────────────────

def _allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _save_upload(file) -> tuple:
    """
    Save an uploaded file to the uploads folder.

    Returns:
        Tuple of (save_path, image_url).
    """
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(save_path)
    image_url = f"/static/uploads/{unique_name}"
    return save_path, image_url


# ──────────────────────────── ROUTES ────────────────────────────

@main_bp.route("/")
def index():
    """Serve the main web page."""
    return render_template("index.html")


@main_bp.route("/predict", methods=["POST"])
def predict():
    """
    Handle image upload, run prediction (single or ensemble), return JSON.

    Form fields:
        file  (file)  — The image to analyze
        model (str)   — 'cnn', 'resnet', 'efficientnet', or 'ensemble'
    """
    # ── Validate file ───────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "error": f"Invalid file format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    # ── Save file ───────────────────────────────────────────────
    save_path, image_url = _save_upload(file)
    model_name = request.form.get("model", "efficientnet").lower()

    logger.info(f"Prediction request: model={model_name}, file={file.filename}")

    try:
        # ── VisionProbe XGBoost Prediction ───────────────────────────────
        result = predict_single(save_path, "xgboost")
        
        # ── FFT analysis (for UI compatibility) ─────────────────────────
        fft_result = extract_fft_features(save_path)

        # ── Generate Grad-CAM (fallback to EfficientNet) ────────────────
        try:
            best_model_name = "efficientnet"
            best_model = model_manager.get(best_model_name)
            gradcam_url = generate_gradcam(save_path, best_model, best_model_name)
        except Exception:
            gradcam_url = None

        response = {
            "label": result["label"],
            "confidence": result["confidence"],
            "raw_score": result.get("raw_score", 0),
            "model_used": "VisionProbe (XGBoost 4-Branch)",
            "method": "visionprobe_xgboost",
            "image_url": image_url,
            "gradcam_url": gradcam_url,
            "individual_results": {},
            "fft_analysis": fft_result,
        }

        # ── Explainability — generate reasoning ─────────────────
        try:
            explanation = generate_explanation(
                prediction={
                    "label": response["label"],
                    "confidence": response["confidence"],
                    "raw_score": response.get("raw_score", 0.5),
                },
                fft_result=fft_result,
                image_path=save_path,
            )
            response["explanation"] = explanation
        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}")
            response["explanation"] = None

        # ── Add to history ──────────────────────────────────────
        _add_to_history({
            "label": response["label"],
            "confidence": response["confidence"],
            "model_used": response["model_used"],
            "image_url": image_url,
        })
        
        import history_db
        history_db.log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), response["label"], response["confidence"], response["model_used"])

        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@main_bp.route("/dashboard")
def dashboard():
    """Serve the history dashboard."""
    import history_db
    rows = history_db.get()
    return render_template("dashboard.html", rows=rows)


@main_bp.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for programmatic access.

    Input:  multipart/form-data with 'file' field
    Query:  ?model=ensemble (default) | cnn | resnet | efficientnet

    Returns JSON:
        {
            "prediction": "AI" or "Real",
            "confidence": float,
            "model": string,
            "gradcam_image": string (URL path),
            "fft_analysis": { ... }
        }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send as multipart/form-data with key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "" or not _allowed_file(file.filename):
        return jsonify({"error": "Invalid or missing file."}), 400

    save_path, image_url = _save_upload(file)
    model_name = request.args.get("model", request.form.get("model", "ensemble")).lower()

    try:
        fft_result = extract_fft_features(save_path)
        
        # Always use VisionProbe pipeline
        result = predict_single(save_path, "xgboost")
        
        gradcam_url = None
        try:
            m = model_manager.get("efficientnet")
            gradcam_url = generate_gradcam(save_path, m, "efficientnet")
        except Exception:
            pass

        return jsonify({
            "prediction": "AI" if result["label"] == "AI Generated" else "Real",
            "confidence": result["confidence"],
            "model": model_name,
            "image_url": image_url,
            "gradcam_image": gradcam_url,
            "fft_analysis": fft_result,
        })

    except Exception as e:
        logger.error(f"API prediction failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500



@main_bp.route("/api/history", methods=["GET"])
def api_history():
    """Return recent prediction history as JSON."""
    return jsonify({"history": _history})


@main_bp.route("/api/health", methods=["GET"])
def api_health():
    """
    System health check endpoint.

    Returns model status, memory usage, uptime, and feature availability.
    """
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        models_status = {}
        for name in MODEL_CONFIGS:
            models_status[name] = model_manager.is_loaded(name)

        health = {
            "status": "healthy",
            "uptime_seconds": round(time.time() - _start_time, 1),
            "memory_mb": round(mem_info.rss / 1024 / 1024, 1),
            "models": models_status,
            "features": {
                "meta_ensemble": meta_is_trained(),
                "patch_analysis": True,
            },
            "config": {
                "patch_size": PATCH_SIZE,
            },
        }
        return jsonify(health)

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@main_bp.route("/predict/patch", methods=["POST"])
def predict_patch():
    """UI endpoint for patch-based analysis."""
    return _handle_patch_analysis(request)


@main_bp.route("/api/patch_analysis", methods=["POST"])
def api_patch_analysis():
    """REST API endpoint for patch-based analysis."""
    return _handle_patch_analysis(request)


def _handle_patch_analysis(req):
    """
    Run patch-based AI artifact detection.

    Form fields:
        file       (file)  — The image to analyze
        patch_size (int)   — Patch size (default: config PATCH_SIZE)
        classifier (str)   — Classifier name (default: efficientnet)
    """
    t_start = time.time()

    if "file" not in req.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = req.files["file"]
    if file.filename == "" or not _allowed_file(file.filename):
        return jsonify({"error": "Invalid file."}), 400

    save_path, image_url = _save_upload(file)

    patch_size = int(req.form.get("patch_size", PATCH_SIZE))
    classifier = req.form.get("classifier", "efficientnet")
    stride = req.form.get("stride")
    stride = int(stride) if stride else PATCH_STRIDE

    logger.info(f"Patch analysis: size={patch_size}, classifier={classifier}")

    try:
        result = analyze_patches(
            image_path=save_path,
            patch_size=patch_size,
            classifier_name=classifier,
            stride=stride,
        )

        result["mode"] = "patch"
        result["original_image_url"] = image_url
        result["processing_time_ms"] = round((time.time() - t_start) * 1000, 1)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Patch analysis failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


