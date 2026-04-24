"""
services/predictor.py — Unified VisionProbe Prediction

Integrates the newly trained VisionProbe XGBoost-based pipeline 
into the legacy Flask application, replacing the old Keras models.
"""

import sys
import os
import time
from PIL import Image
from utils.logger import setup_logger
from config import MODEL_CONFIGS

logger = setup_logger("services.predictor")

# Add the visionprobe backend to the path so we can import it
_visionprobe_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../visionprobe/backend'))
if _visionprobe_path not in sys.path:
    sys.path.insert(0, _visionprobe_path)

try:
    from detector.predict import AIDetector
except ImportError:
    logger.error(f"Could not import AIDetector from {_visionprobe_path}. Make sure the path is correct.")
    AIDetector = None

# Instantiate a global detector
_detector = None

def get_detector():
    global _detector
    if _detector is None and AIDetector is not None:
        logger.info("Initializing VisionProbe XGBoost pipeline...")
        # Path to the weights directory where artifacts are saved
        weights_dir = os.path.join(_visionprobe_path, "weights")
        _detector = AIDetector(weights_dir=weights_dir, device="auto")
        _detector.load()
    return _detector


def predict_single(image_path: str, model_name: str = "xgboost") -> dict:
    """
    Run the unified VisionProbe prediction pipeline.
    
    Args:
        image_path: Absolute path to the image file.
        model_name: Ignored in the new architecture, but kept for compatibility.
        
    Returns:
        Dict with label, confidence, raw_score, model_name, evidence.
    """
    logger.info(f"VisionProbe prediction for: {image_path}")

    try:
        detector = get_detector()
        if not detector:
            raise RuntimeError("AIDetector is not available.")
            
        pil_img = Image.open(image_path).convert("RGB")
        
        # Use predict_with_features to get rich explainability data
        result = detector.predict_with_features(pil_img)
        
        # Format the output to match the legacy API format
        confidence_value = result["confidence"]
        raw_prob = result["raw_probability"]
        label = "AI Generated" if result["label"] == "AI" else "Real Image"
        
        # Build evidence list based on raw_prob
        evidence = []
        if raw_prob > 0.8:
            evidence.append("Strong AI signatures detected across multiple branches (FFT, SRM, CLIP).")
        elif raw_prob > 0.5:
            evidence.append("Moderate AI signatures detected.")
        elif raw_prob < 0.2:
            evidence.append("Strong authentic camera signatures detected.")
        else:
            evidence.append("Image appears mostly real, but exhibits some synthetic characteristics.")
            
        return {
            "label": label,
            "confidence": confidence_value,
            "raw_score": raw_prob,
            "model_name": "VisionProbe (XGBoost 4-Branch)",
            "nn_used": True,
            "nn_raw": raw_prob,
            "fft_score": raw_prob,  # Provide a proxy for legacy UI elements
            "signal_score": raw_prob,
            "key_evidence": evidence,
            "signals": {}, # Legacy signals no longer computed here
            "visionprobe_features": result.get("feature_blocks", {})
        }
    except Exception as e:
        logger.error(f"VisionProbe prediction failed: {e}", exc_info=True)
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "raw_score": 0.5,
            "model_name": "Error",
            "error": str(e)
        }

def predict_all(image_path: str) -> dict:
    """
    Run prediction. Maintained for legacy compatibility.
    """
    # In the new architecture, the ensemble is natively handled by XGBoost
    return {
        "xgboost": predict_single(image_path, "xgboost")
    }
