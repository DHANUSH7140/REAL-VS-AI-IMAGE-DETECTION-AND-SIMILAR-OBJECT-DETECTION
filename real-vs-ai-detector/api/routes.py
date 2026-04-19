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
                    MODEL_CONFIGS, DEFAULT_SIMILARITY_THRESHOLD,
                    ENABLE_VIT, ENABLE_FAISS, FEATURE_BACKBONE,
                    PATCH_SIZE, PATCH_STRIDE)
from models.loader import model_manager
from services.predictor import predict_single, predict_all
from services.ensemble import weighted_voting
from services.gradcam import generate_gradcam
from services.fft_features import extract_fft_features
from services.combined_visualization import run_combined_pipeline
from config import DETECTOR_MODEL

from services.feature_extractor import extract_embedding, extract_embeddings_batch
from services.similarity_matcher import find_similar_regions
from services.similarity_visualization import (
    generate_similarity_visualization,
    generate_cluster_visualization,
)
from services.dynamic_threshold import adaptive_select
from services.clustering import cluster_embeddings, get_roi_cluster_members
from services.vit_embedder import (
    extract_vit_embedding, extract_vit_embeddings_batch,
    is_available as vit_is_available
)
from services.mobilenet_extractor import (
    extract_mobilenet_embedding, extract_mobilenet_embeddings_batch
)
from services.embedding_fusion import fuse_embeddings, fuse_embeddings_batch

def get_detections(image_path, conf_threshold=0.15, max_detections=100):
    from config import DETECTOR_MODEL
    if DETECTOR_MODEL == "rtdetr":
        from services.rtdetr_detector import detect_objects
        return detect_objects(image_path, conf_threshold, max_detections)
    else:
        from services.yolo_detector import detect_objects
        return detect_objects(image_path, conf_threshold, max_detections)
from services.faiss_engine import search as faiss_search, is_available as faiss_is_available
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
        # ── FFT analysis (run first — used by ensemble + response) ──
        fft_result = extract_fft_features(save_path)

        # ── Ensemble prediction ─────────────────────────────────
        if model_name == "ensemble":
            all_predictions = predict_all(save_path)
            result = weighted_voting(all_predictions, fft_result=fft_result)

            # Generate Grad-CAM for the highest-weight model
            try:
                best_model_name = "efficientnet"
                best_model = model_manager.get(best_model_name)
                gradcam_url = generate_gradcam(save_path, best_model, best_model_name)
            except Exception:
                gradcam_url = None

            # Format individual results for the frontend
            individual = {}
            for name, pred in result.get("individual_results", {}).items():
                cfg = MODEL_CONFIGS.get(name, {})
                individual[name] = {
                    "label": pred.get("label", "Unknown"),
                    "confidence": pred.get("confidence", 0),
                    "display_name": cfg.get("display_name", pred.get("model_name", name)),
                    "icon": cfg.get("icon", "🤖"),
                }

            response = {
                "label": result["label"],
                "confidence": result["confidence"],
                "raw_score": result.get("raw_score", 0),
                "model_used": "ENSEMBLE (Weighted)",
                "method": result.get("method", "weighted_voting"),
                "image_url": image_url,
                "gradcam_url": gradcam_url,
                "individual_results": individual,
            }

        # ── Single model prediction ─────────────────────────────
        else:
            result = predict_single(save_path, model_name)

            # Generate Grad-CAM
            try:
                model = model_manager.get(model_name)
                gradcam_url = generate_gradcam(save_path, model, model_name)
            except Exception as e:
                logger.warning(f"Grad-CAM skipped: {e}")
                gradcam_url = None

            cfg = MODEL_CONFIGS.get(model_name, {})
            response = {
                "label": result["label"],
                "confidence": result["confidence"],
                "raw_score": result.get("raw_score", 0),
                "model_used": cfg.get("display_name", model_name).upper(),
                "image_url": image_url,
                "gradcam_url": gradcam_url,
            }

        # ── Attach FFT to response ──────────────────────────────
        response["fft_analysis"] = fft_result

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

        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


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

        if model_name == "ensemble":
            all_predictions = predict_all(save_path)
            result = weighted_voting(all_predictions, fft_result=fft_result)
            gradcam_url = None
            try:
                m = model_manager.get("efficientnet")
                gradcam_url = generate_gradcam(save_path, m, "efficientnet")
            except Exception:
                pass
        else:
            result = predict_single(save_path, model_name)
            gradcam_url = None
            try:
                m = model_manager.get(model_name)
                gradcam_url = generate_gradcam(save_path, m, model_name)
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


@main_bp.route("/predict/combined", methods=["POST"])
def predict_combined():
    """
    Handle combined YOLO + Classifier pipeline.

    Detects objects via YOLOv8, classifies each region as Real/AI,
    overlays Grad-CAM heatmaps per-region, and returns the annotated image.

    Form fields:
        file  (file)  — The image to analyze
        model (str)   — Classifier to use ('cnn', 'resnet', 'efficientnet')
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
    classifier_name = request.form.get("model", "efficientnet").lower()

    logger.info(f"Combined pipeline request: classifier={classifier_name}, file={file.filename}")

    try:
        result = run_combined_pipeline(
            image_path=save_path,
            classifier_name=classifier_name,
        )

        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        # FFT analysis on the full image
        fft_result = extract_fft_features(save_path)

        response = {
            "mode": "combined",
            "annotated_image_url": result["annotated_image_url"],
            "original_image_url": image_url,
            "regions": result["regions"],
            "total_objects": result["total_objects"],
            "real_count": result["real_count"],
            "ai_count": result["ai_count"],
            "summary": result["summary"],
            "classifier_used": result["classifier_used"],
            "fft_analysis": fft_result,
        }

        # Add to history
        overall_label = "AI Generated" if result["ai_count"] >= result["real_count"] and result["ai_count"] > 0 else "Real Image"
        ai_pct = round(result["ai_count"] / max(result["total_objects"], 1) * 100, 1)
        _add_to_history({
            "label": overall_label,
            "confidence": ai_pct if overall_label == "AI Generated" else round(100 - ai_pct, 1),
            "model_used": f"COMBINED ({result['classifier_used'].upper()})",
            "image_url": image_url,
        })

        return jsonify(response)

    except Exception as e:
        logger.error(f"Combined pipeline failed: {e}", exc_info=True)
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
                "vit_embeddings": vit_is_available() if ENABLE_VIT else False,
                "faiss_search": faiss_is_available() if ENABLE_FAISS else False,
                "meta_ensemble": meta_is_trained(),
                "patch_analysis": True,
            },
            "config": {
                "feature_backbone": FEATURE_BACKBONE,
                "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
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


@main_bp.route("/predict/similar", methods=["POST"])
def predict_similar():
    """
    Handle similarity search from the UI.

    Form fields:
        file       (file)  — The image to analyze
        x1, y1, x2, y2     — ROI coordinates (pixels)
        threshold  (float) — Similarity threshold (default 0.7)
    """
    return _handle_find_similar(request)


@main_bp.route("/api/find_similar", methods=["POST"])
def api_find_similar():
    """
    REST API endpoint for similarity search.

    Input:  multipart/form-data with 'file' + ROI coordinates
    Query:  ?threshold=0.7

    Returns JSON with matched regions and annotated image URL.
    """
    return _handle_find_similar(request)


@main_bp.route("/predict/similar_advanced", methods=["POST"])
def predict_similar_advanced():
    """UI endpoint for advanced similarity with clustering."""
    return _handle_advanced_similar(request)


@main_bp.route("/api/similar_objects_advanced", methods=["POST"])
def api_similar_advanced():
    """REST API endpoint for advanced similarity with clustering."""
    return _handle_advanced_similar(request)


def _handle_find_similar(req):
    """
    Core similarity search logic — upgraded with ViT + FAISS.

    Uses ViT embeddings (primary), EfficientNet (fallback), or fused
    embeddings based on config. FAISS for fast search when available.
    """
    t_start = time.time()

    # ── Validate file ───────────────────────────────────────
    if "file" not in req.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = req.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "error": f"Invalid file format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    # ── Validate ROI coordinates ───────────────────────────
    try:
        x1 = int(req.form.get("x1", 0))
        y1 = int(req.form.get("y1", 0))
        x2 = int(req.form.get("x2", 0))
        y2 = int(req.form.get("y2", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid ROI coordinates."}), 400

    if x1 >= x2 or y1 >= y2:
        return jsonify({"error": "Invalid ROI: x1 < x2 and y1 < y2 required."}), 400

    threshold = float(req.form.get(
        "threshold",
        req.args.get("threshold", DEFAULT_SIMILARITY_THRESHOLD)
    ))

    # ── Save file ─────────────────────────────────────────
    save_path, image_url = _save_upload(file)
    roi_box = [x1, y1, x2, y2]

    logger.info(f"Similarity search: ROI={roi_box}, threshold={threshold}, backbone={FEATURE_BACKBONE}")

    try:
        import cv2

        # ── 1. Load image and extract ROI crop ────────────────
        img = cv2.imread(save_path)
        if img is None:
            return jsonify({"error": "Could not load image."}), 500

        h_img, w_img = img.shape[:2]

        # Clamp ROI to image bounds
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(x1 + 1, min(x2, w_img))
        y2 = max(y1 + 1, min(y2, h_img))
        roi_box = [x1, y1, x2, y2]

        roi_crop = img[y1:y2, x1:x2]

        # ── 2. Run Object Detection to find all objects ─────────────────
        detections = get_detections(save_path, conf_threshold=0.15, max_detections=100)

        # ── 2b. Patch scan fallback if YOLO finds too few ─────
        if len(detections) < 3:
            logger.info(f"YOLO found {len(detections)} — using patch scan")
            roi_w = x2 - x1
            roi_h = y2 - y1
            patch_w = max(roi_w, 32)
            patch_h = max(roi_h, 32)
            stride_x = max(patch_w // 2, 16)
            stride_y = max(patch_h // 2, 16)

            patch_dets = []
            for py in range(0, h_img - patch_h + 1, stride_y):
                for px in range(0, w_img - patch_w + 1, stride_x):
                    overlap_x = max(0, min(px + patch_w, x2) - max(px, x1))
                    overlap_y = max(0, min(py + patch_h, y2) - max(py, y1))
                    if roi_w * roi_h > 0 and (overlap_x * overlap_y) / (roi_w * roi_h) > 0.5:
                        continue
                    patch_dets.append({
                        "bbox": [px, py, px + patch_w, py + patch_h],
                        "class_name": "region",
                        "confidence": 1.0,
                    })

            if len(patch_dets) > 80:
                step = len(patch_dets) // 80
                patch_dets = patch_dets[::step][:80]

            detections = detections + patch_dets

        # ── 3. Extract object crops ───────────────────────────
        object_crops = []
        for det in detections:
            bx1, by1, bx2, by2 = det["bbox"]
            bx1 = max(0, bx1)
            by1 = max(0, by1)
            bx2 = min(w_img, bx2)
            by2 = min(h_img, by2)
            crop = img[by1:by2, bx1:bx2]
            if crop.size > 0:
                object_crops.append(crop)
            else:
                object_crops.append(img[0:10, 0:10])

        # ── 4. Extract embeddings (ViT or EfficientNet) ──────
        embedding_source = "efficientnet"  # fallback

        if FEATURE_BACKBONE == "vit" and ENABLE_VIT and vit_is_available():
            # Use ViT
            roi_embedding = extract_vit_embedding(roi_crop)
            object_embeddings = extract_vit_embeddings_batch(object_crops) if object_crops else __import__('numpy').array([])
            embedding_source = "vit"

        elif FEATURE_BACKBONE == "mobilenetv3":
            # Use MobileNetV3
            from services.mobilenet_extractor import extract_mobilenet_embedding, extract_mobilenet_embeddings_batch
            roi_embedding = extract_mobilenet_embedding(roi_crop)
            object_embeddings = extract_mobilenet_embeddings_batch(object_crops) if object_crops else __import__('numpy').array([])
            embedding_source = "mobilenetv3"

        elif FEATURE_BACKBONE == "vector":
            # Use HOG Vector extractor
            from services.vector_extractor import extract_vector_embedding, extract_vector_embeddings_batch
            roi_embedding = extract_vector_embedding(roi_crop)
            object_embeddings = extract_vector_embeddings_batch(object_crops) if object_crops else __import__('numpy').array([])
            embedding_source = "vector"

        elif FEATURE_BACKBONE == "fused" and ENABLE_VIT and vit_is_available():
            # Fuse ViT + EfficientNet
            import numpy as np
            roi_vit = extract_vit_embedding(roi_crop)
            roi_eff = extract_embedding(roi_crop)
            roi_embedding = fuse_embeddings(vit_embedding=roi_vit, effnet_embedding=roi_eff)

            if object_crops:
                obj_vit = extract_vit_embeddings_batch(object_crops)
                obj_eff = extract_embeddings_batch(object_crops)
                object_embeddings = fuse_embeddings_batch(
                    vit_embeddings=obj_vit, effnet_embeddings=obj_eff
                )
            else:
                object_embeddings = np.array([])
            embedding_source = "fused (vit+efficientnet)"

        else:
            # EfficientNet fallback
            roi_embedding = extract_embedding(roi_crop)
            object_embeddings = extract_embeddings_batch(object_crops) if object_crops else __import__('numpy').array([])
            embedding_source = "efficientnet"

        # ── 5. Find similar regions (FAISS or numpy) ─────────
        import numpy as np
        if object_embeddings.size > 0:
            if ENABLE_FAISS and faiss_is_available():
                # Use FAISS
                search_results = faiss_search(
                    query=roi_embedding,
                    candidates=object_embeddings,
                    threshold=threshold,
                    use_faiss=True,
                )
                matches = []
                for idx, sim in search_results:
                    if idx < len(detections):
                        det = detections[idx]
                        matches.append({
                            "index": idx,
                            "box": det["bbox"],
                            "class_name": det.get("class_name", "unknown"),
                            "confidence": det.get("confidence", 0.0),
                            "similarity": round(sim, 4),
                        })
            else:
                matches = find_similar_regions(
                    roi_embedding=roi_embedding,
                    detections=detections,
                    object_embeddings=object_embeddings,
                    threshold=threshold,
                )
        else:
            matches = []

        # ── 6. Generate annotated visualization ───────────────
        annotated_url = generate_similarity_visualization(
            image_path=save_path,
            roi_box=roi_box,
            detections=detections,
            matches=matches,
        )

        response = {
            "mode": "similar",
            "roi": {"box": roi_box},
            "total_detected": len(detections),
            "match_count": len(matches),
            "matches": matches,
            "threshold": threshold,
            "embedding_source": embedding_source,
            "faiss_used": ENABLE_FAISS and faiss_is_available(),
            "annotated_image_url": annotated_url,
            "original_image_url": image_url,
            "processing_time_ms": round((time.time() - t_start) * 1000, 1),
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Similarity search failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def _handle_advanced_similar(req):
    """
    Advanced similarity search with dynamic thresholding + DBSCAN clustering.

    Form fields:
        file                 — Image file
        x1, y1, x2, y2      — ROI coordinates
        threshold_method     — 'top_k', 'statistical', 'percentile', 'combined' (default: statistical)
        top_k                — Number of top matches (for top_k method)
        multiplier           — Std dev multiplier (for statistical method)
        percentile           — Percentile value (for percentile method)
        eps                  — DBSCAN epsilon (default: 0.35)
        min_samples          — DBSCAN min_samples (default: 2)
        view_mode            — 'similarity' or 'clustering' (default: clustering)
    """
    t_start = time.time()

    # ── Validate file ──────────────────────────────────────────────
    if "file" not in req.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = req.files["file"]
    if file.filename == "" or not _allowed_file(file.filename):
        return jsonify({"error": "Invalid file."}), 400

    # ── Validate ROI ───────────────────────────────────────────────
    try:
        x1 = int(req.form.get("x1", 0))
        y1 = int(req.form.get("y1", 0))
        x2 = int(req.form.get("x2", 0))
        y2 = int(req.form.get("y2", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid ROI coordinates."}), 400
    if x1 >= x2 or y1 >= y2:
        return jsonify({"error": "Invalid ROI: x1 < x2 and y1 < y2 required."}), 400

    # ── Parameters ─────────────────────────────────────────────────
    threshold_method = req.form.get("threshold_method", "statistical")
    top_k_val = int(req.form.get("top_k", 5))
    multiplier = float(req.form.get("multiplier", 0.5))
    pct = float(req.form.get("percentile", 90.0))
    eps = float(req.form.get("eps", 0.35))
    min_samples_val = int(req.form.get("min_samples", 2))
    view_mode = req.form.get("view_mode", "clustering")

    save_path, image_url = _save_upload(file)
    roi_box = [x1, y1, x2, y2]

    logger.info(
        f"Advanced similarity: ROI={roi_box}, method={threshold_method}, "
        f"view={view_mode}, eps={eps}"
    )

    try:
        import cv2
        import numpy as np

        # ── 1. Load image + ROI ────────────────────────────────────
        img = cv2.imread(save_path)
        if img is None:
            return jsonify({"error": "Could not load image."}), 500

        h_img, w_img = img.shape[:2]
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(x1 + 1, min(x2, w_img))
        y2 = max(y1 + 1, min(y2, h_img))
        roi_box = [x1, y1, x2, y2]
        roi_crop = img[y1:y2, x1:x2]

        # 2. Run object detection dynamically ───────────────────────
        detections = get_detections(save_path, conf_threshold=0.15, max_detections=100)

        # ── 2b. Patch-based fallback if YOLO finds too few objects ──
        use_patch_scan = len(detections) < 3
        if use_patch_scan:
            logger.info(f"YOLO found {len(detections)} objects — using patch scan fallback")
            roi_w = x2 - x1
            roi_h = y2 - y1
            # Use ROI dimensions as patch size, stride = half for overlap
            patch_w = max(roi_w, 32)
            patch_h = max(roi_h, 32)
            stride_x = max(patch_w // 2, 16)
            stride_y = max(patch_h // 2, 16)

            patch_detections = []
            for py in range(0, h_img - patch_h + 1, stride_y):
                for px in range(0, w_img - patch_w + 1, stride_x):
                    # Skip if this patch IS the ROI (or mostly overlaps)
                    overlap_x = max(0, min(px + patch_w, x2) - max(px, x1))
                    overlap_y = max(0, min(py + patch_h, y2) - max(py, y1))
                    overlap_area = overlap_x * overlap_y
                    roi_area = roi_w * roi_h
                    if roi_area > 0 and overlap_area / roi_area > 0.5:
                        continue

                    patch_detections.append({
                        "bbox": [px, py, px + patch_w, py + patch_h],
                        "class_name": "region",
                        "confidence": 1.0,
                    })

            # Limit to avoid memory issues
            if len(patch_detections) > 100:
                # Keep evenly sampled patches
                step = len(patch_detections) // 100
                patch_detections = patch_detections[::step][:100]

            # Merge with any YOLO detections
            detections = detections + patch_detections
            logger.info(f"Patch scan generated {len(patch_detections)} regions, total: {len(detections)}")



        if len(detections) == 0:
            # No objects detected — return early
            annotated_url = generate_similarity_visualization(
                save_path, roi_box, detections, []
            )
            return jsonify({
                "mode": "similar_advanced",
                "roi": {"box": roi_box},
                "total_detected": 0,
                "match_count": 0,
                "matches": [],
                "clustering": {"n_clusters": 0, "cluster_summary": {}, "roi_cluster_id": -1},
                "threshold_info": {"method": threshold_method, "threshold": 0.0},
                "annotated_image_url": annotated_url,
                "cluster_image_url": annotated_url,
                "original_image_url": image_url,
                "detector_model": DETECTOR_MODEL,
                "processing_time_ms": round((time.time() - t_start) * 1000, 1),
            })

        # ── 3. Extract crops + embeddings ──────────────────────────
        object_crops = []
        for det in detections:
            bx1, by1, bx2, by2 = det["bbox"]
            crop = img[max(0, by1):min(h_img, by2), max(0, bx1):min(w_img, bx2)]
            if crop.size > 0:
                object_crops.append(crop)
            else:
                object_crops.append(img[0:10, 0:10])

        embedding_source = "efficientnet"
        if FEATURE_BACKBONE == "vit" and ENABLE_VIT and vit_is_available():
            roi_embedding = extract_vit_embedding(roi_crop)
            object_embeddings = extract_vit_embeddings_batch(object_crops) if object_crops else np.array([])
            embedding_source = "vit"
        elif FEATURE_BACKBONE == "mobilenetv3":
            from services.mobilenet_extractor import extract_mobilenet_embedding, extract_mobilenet_embeddings_batch
            roi_embedding = extract_mobilenet_embedding(roi_crop)
            object_embeddings = extract_mobilenet_embeddings_batch(object_crops) if object_crops else __import__('numpy').array([])
            embedding_source = "mobilenetv3"
        elif FEATURE_BACKBONE == "vector":
            from services.vector_extractor import extract_vector_embedding, extract_vector_embeddings_batch
            roi_embedding = extract_vector_embedding(roi_crop)
            object_embeddings = extract_vector_embeddings_batch(object_crops) if object_crops else __import__('numpy').array([])
            embedding_source = "vector"
        elif FEATURE_BACKBONE == "fused" and ENABLE_VIT and vit_is_available():
            roi_vit = extract_vit_embedding(roi_crop)
            roi_eff = extract_embedding(roi_crop)
            roi_embedding = fuse_embeddings(vit_embedding=roi_vit, effnet_embedding=roi_eff)

            if object_crops:
                obj_vit = extract_vit_embeddings_batch(object_crops)
                obj_eff = extract_embeddings_batch(object_crops)
                object_embeddings = fuse_embeddings_batch(
                    vit_embeddings=obj_vit, effnet_embeddings=obj_eff
                )
            else:
                object_embeddings = np.array([])
            embedding_source = "fused (vit+efficientnet)"
        else:
            roi_embedding = extract_embedding(roi_crop)
            object_embeddings = extract_embeddings_batch(object_crops) if object_crops else np.array([])

        # ── 4. Compute ALL similarity scores ───────────────────────
        all_scores = object_embeddings @ roi_embedding  # cosine for L2-normed
        all_scores = np.clip(all_scores, 0, 1)

        # ── 5. Dynamic threshold ───────────────────────────────────
        threshold_result = adaptive_select(
            all_scores,
            method=threshold_method,
            k=top_k_val,
            multiplier=multiplier,
            pct=pct,
        )

        mask = threshold_result["mask"]
        effective_threshold = threshold_result["threshold"]

        # ── 6. DBSCAN clustering ───────────────────────────────────
        clustering_result = cluster_embeddings(
            embeddings=object_embeddings,
            roi_embedding=roi_embedding,
            eps=eps,
            min_samples=min_samples_val,
        )

        roi_cluster_id = clustering_result["roi_cluster_id"]
        roi_cluster_members = get_roi_cluster_members(clustering_result)

        # ── 7. Combine: threshold mask ∩ ROI cluster ───────────────
        # Objects that pass threshold AND belong to ROI's cluster
        selected_indices = set()
        for i in range(len(detections)):
            if mask[i]:
                selected_indices.add(i)

        # Also include ROI cluster members (even if below threshold)
        for idx in roi_cluster_members:
            selected_indices.add(idx)

        # Build matches list
        matches = []
        for i in sorted(selected_indices):
            det = detections[i]
            cid = clustering_result["labels"][i] if i < len(clustering_result["labels"]) else -1
            matches.append({
                "index": i,
                "box": det["bbox"],
                "class_name": det.get("class_name", "unknown"),
                "confidence": det.get("confidence", 0.0),
                "similarity": round(float(all_scores[i]), 4),
                "cluster_id": cid,
            })

        # Sort by similarity descending
        matches.sort(key=lambda m: m["similarity"], reverse=True)

        # ── 8. Generate visualizations ─────────────────────────────
        # Standard similarity view
        sim_viz_url = generate_similarity_visualization(
            save_path, roi_box, detections, matches,
        )

        # Cluster-colored view
        cluster_viz_url = generate_cluster_visualization(
            save_path, roi_box, detections, matches,
            clustering_result=clustering_result,
            all_scores=all_scores.tolist(),
        )

        # Pick which one to show based on view_mode
        primary_url = cluster_viz_url if view_mode == "clustering" else sim_viz_url

        # ── 9. Build response ──────────────────────────────────────
        response = {
            "mode": "similar_advanced",
            "roi": {"box": roi_box},
            "total_detected": len(detections),
            "match_count": len(matches),
            "matches": matches,
            "clustering": {
                "n_clusters": clustering_result["n_clusters"],
                "roi_cluster_id": roi_cluster_id,
                "cluster_summary": clustering_result["cluster_summary"],
                "cluster_colors": clustering_result["cluster_colors"],
                "labels": clustering_result["labels"],
            },
            "threshold_info": {
                "method": threshold_method,
                "threshold": effective_threshold,
                "stats": threshold_result["stats"],
            },
            "all_scores": [round(float(s), 4) for s in all_scores],
            "embedding_source": embedding_source,
            "annotated_image_url": sim_viz_url,
            "cluster_image_url": cluster_viz_url,
            "primary_image_url": primary_url,
            "original_image_url": image_url,
            "view_mode": view_mode,
            "detector_model": DETECTOR_MODEL,
            "processing_time_ms": round((time.time() - t_start) * 1000, 1),
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Advanced similarity failed: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
