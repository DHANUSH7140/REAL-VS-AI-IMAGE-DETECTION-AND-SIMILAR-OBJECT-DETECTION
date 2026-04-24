"""
Full detection pipeline — runs feature extraction, XGBoost classification, and explainability.
"""

import time
import logging
import numpy as np
from PIL import Image

from model_cache import ModelCache
from preprocessing import decode_image, resize_for_processing, extract_exif
from detector.explain import generate_gradcam, generate_text_reasoning
from detector.feature_extractors import FFTFeatureExtractor

logger = logging.getLogger("visionprobe.detector.inference")


async def run_full_pipeline(
    image_bytes: bytes,
    include_gradcam: bool = True,
    include_shap: bool = True,
) -> dict:
    """
    Complete AI vs Real detection pipeline.
    Uses multi-branch features → XGBoost → calibration → explainability.
    """
    t0 = time.time()
    cache = ModelCache.get_instance()

    # 1. Preprocessing
    pil_img, np_img = decode_image(image_bytes)
    original_size = pil_img.size
    pil_img = resize_for_processing(pil_img)
    np_img = np.array(pil_img)
    exif_meta = extract_exif(image_bytes)

    # 2. Run prediction
    if cache.detector is not None:
        result = cache.detector.predict_with_features(pil_img)
        raw_prob = result["raw_probability"]
        label = result["label"]
        confidence = result["confidence"]
        feature_blocks = result.get("feature_blocks", {})
    else:
        # Fallback heuristic
        from detector.predict import AIDetector
        detector = AIDetector()
        detector.load()
        result = detector.predict(pil_img)
        raw_prob = result["raw_probability"]
        label = result["label"]
        confidence = result["confidence"]
        feature_blocks = {}

    # 3. Extract per-branch scores for display
    fft_features = np.array(feature_blocks.get("fft", [0.0] * 8))
    srm_features = np.array(feature_blocks.get("srm", [0.0] * 15))

    # Compute display scores from features
    fft_score = _compute_fft_score(fft_features)
    srm_score = _compute_srm_score(srm_features)
    clip_score = _compute_clip_score(cache, pil_img)
    effnet_score = raw_prob  # Use overall as proxy when model is an ensemble

    # EXIF score
    exif_score = _compute_exif_score(exif_meta)

    # 4. Explainability
    gradcam_b64 = None
    if include_gradcam and cache.effnet_available:
        try:
            gradcam_b64 = generate_gradcam(pil_img, cache.effnet_backbone, cache.device)
        except Exception as e:
            logger.warning(f"GradCAM failed: {e}")

    # Text reasoning
    clip_ext = None
    if cache.detector and cache.detector.extractor and cache.detector.extractor.clip_ext:
        clip_ext = cache.detector.extractor.clip_ext

    reasoning = generate_text_reasoning(
        pil_img, clip_ext, raw_prob, fft_features, srm_features
    )

    # Spectrum data for frontend
    spectrum_data = FFTFeatureExtractor.extract_spectrum_for_display(pil_img)

    # Feature importance (from XGBoost)
    feature_importance = []
    if include_shap:
        from detector.explain import get_feature_importance
        feature_importance = get_feature_importance()

    # 5. Build response (compatible with existing frontend)
    verdict = "AI_GENERATED" if label == "AI" else "REAL"
    processing_time_ms = int((time.time() - t0) * 1000)

    models_used = []
    if cache.clip_available: models_used.append("OpenCLIP-ViT-L/14")
    if cache.effnet_available: models_used.append("EfficientNetV2-L")
    models_used.append("FFT Analysis")
    models_used.append("SRM Noise Analysis")
    if cache.xgb_available: models_used.append("XGBoost Classifier")

    return {
        "verdict": verdict,
        "confidence": round(confidence / 100, 4),  # 0-1 scale for frontend
        "confidence_interval": 0.05 if confidence > 60 else 0.08,
        "scores": {
            "clip": round(clip_score, 4),
            "efficientnet": round(effnet_score, 4),
            "frequency": round(fft_score, 4),
            "srm": round(srm_score, 4),
            "exif": round(exif_score, 4),
            "ensemble": round(raw_prob, 4),
        },
        "explanation": {
            "gradcam_heatmap": gradcam_b64,
            "attention_map": None,
            "frequency_spectrum": spectrum_data.get("spectrum_2d_small", []),
            "frequency_profile": spectrum_data.get("profile_1d", []),
            "shap_features": [
                {"name": f["name"], "value": f["importance"], "direction": "AI" if raw_prob > 0.5 else "Real"}
                for f in feature_importance[:10]
            ],
            "reasoning": reasoning,
        },
        "metadata": {
            "processing_time_ms": processing_time_ms,
            "image_size": list(original_size),
            "file_size_bytes": len(image_bytes),
            "exif_fields_found": len(exif_meta),
            "models_used": models_used,
            "device": str(cache.device),
            "pipeline": "multi-branch-xgboost",
        },
    }


def _compute_fft_score(fft_features: np.ndarray) -> float:
    """Compute FFT-based AI score from features."""
    if len(fft_features) < 8:
        return 0.5
    mid_freq = fft_features[3]
    flatness = fft_features[4]
    periodicity = fft_features[5]

    score = 0.5
    if mid_freq > 0.8: score += 0.15
    elif mid_freq > 0.6: score += 0.08
    if flatness < 0.35 or flatness > 0.65: score += 0.1
    if periodicity > 0.5: score += 0.1
    elif periodicity > 0.3: score += 0.05
    return float(np.clip(score, 0.0, 1.0))


def _compute_srm_score(srm_features: np.ndarray) -> float:
    """Compute SRM-based AI score from features."""
    if len(srm_features) < 15:
        return 0.5
    avg_kurtosis = abs(srm_features[4])
    avg_std = srm_features[2]

    if avg_kurtosis > 6.0:
        k_signal = 0.2
    elif avg_kurtosis > 4.0:
        k_signal = 0.35
    elif avg_kurtosis > 2.0:
        k_signal = 0.55
    else:
        k_signal = 0.75

    if avg_std < 0.1:
        v_signal = 0.8
    elif avg_std < 0.5:
        v_signal = 0.6
    else:
        v_signal = 0.3

    return float(np.clip(0.6 * k_signal + 0.4 * v_signal, 0.0, 1.0))


def _compute_clip_score(cache, pil_img: Image.Image) -> float:
    """Compute CLIP-based AI score."""
    if cache.detector and cache.detector.extractor and cache.detector.extractor.clip_ext:
        try:
            sims = cache.detector.extractor.clip_ext.get_similarities(pil_img)
            ratio = sims.get("ratio", 0.5)
            return float(np.clip((ratio - 0.475) * 40.0, 0.05, 0.95))
        except Exception:
            pass
    return 0.5


def _compute_exif_score(exif_meta: dict) -> float:
    """Compute EXIF-based AI score."""
    if not exif_meta:
        return 0.7  # No EXIF → likely AI

    score = 0.3  # Base: has some EXIF → likely real
    camera_tags = {"Make", "Model", "LensModel", "FocalLength", "ExposureTime", "FNumber", "ISOSpeedRatings"}
    found = sum(1 for t in camera_tags if t in exif_meta)
    if found >= 3:
        score -= 0.15  # Strong camera evidence
    if any(k in exif_meta for k in ["GPSInfo", "GPSLatitude"]):
        score -= 0.1  # GPS is strong real indicator
    software = exif_meta.get("Software", "")
    if any(ai_sw in software.lower() for ai_sw in ["stable diffusion", "midjourney", "dall-e", "comfyui"]):
        score += 0.4  # AI software detected
    return float(np.clip(score, 0.0, 1.0))
