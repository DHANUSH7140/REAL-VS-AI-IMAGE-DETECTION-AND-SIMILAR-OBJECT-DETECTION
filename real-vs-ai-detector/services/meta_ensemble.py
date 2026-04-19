"""
services/meta_ensemble.py — Meta-model ensemble combining all sub-model outputs.

Combines ResNet, EfficientNet, patch scores, and FFT features
into a unified prediction using Logistic Regression or XGBoost.
Falls back to weighted voting if meta-models are not trained.
"""

import os
import pickle
import numpy as np

from config import BASE_DIR
from utils.logger import setup_logger

logger = setup_logger("services.meta_ensemble")

# ──────────────────────────── MODEL PATHS ───────────────────────
META_MODEL_DIR = os.path.join(BASE_DIR, "model", "meta")
LR_MODEL_PATH = os.path.join(META_MODEL_DIR, "meta_lr.pkl")
XGB_MODEL_PATH = os.path.join(META_MODEL_DIR, "meta_xgb.pkl")

_lr_model = None
_xgb_model = None
_models_loaded = False


def _load_meta_models():
    """Load trained meta-models from disk (if available)."""
    global _lr_model, _xgb_model, _models_loaded

    if _models_loaded:
        return

    os.makedirs(META_MODEL_DIR, exist_ok=True)

    if os.path.isfile(LR_MODEL_PATH):
        try:
            with open(LR_MODEL_PATH, "rb") as f:
                _lr_model = pickle.load(f)
            logger.info("Meta-model (Logistic Regression) loaded.")
        except Exception as e:
            logger.warning(f"Failed to load LR meta-model: {e}")

    if os.path.isfile(XGB_MODEL_PATH):
        try:
            with open(XGB_MODEL_PATH, "rb") as f:
                _xgb_model = pickle.load(f)
            logger.info("Meta-model (XGBoost) loaded.")
        except Exception as e:
            logger.warning(f"Failed to load XGBoost meta-model: {e}")

    if _lr_model is None and _xgb_model is None:
        logger.info(
            "No trained meta-models found. Using weighted voting fallback. "
            "Train with: python scripts/train_meta_ensemble.py"
        )

    _models_loaded = True


def build_feature_vector(predictions: dict) -> np.ndarray:
    """
    Build a feature vector from sub-model outputs for meta-ensemble.

    Expected keys in predictions:
        - resnet_score: float
        - effnet_score: float
        - patch_score: float (average patch-level AI probability)
        - fft_high_freq: float
        - fft_noise_var: float
        - fft_spectral_centroid: float

    Returns:
        1-D numpy array of features.
    """
    features = [
        predictions.get("resnet_score", 0.5),
        predictions.get("effnet_score", 0.5),
        predictions.get("patch_score", 0.5),
        predictions.get("fft_high_freq", 0.0),
        predictions.get("fft_noise_var", 0.0),
        predictions.get("fft_spectral_centroid", 0.0),
    ]
    return np.array(features, dtype=np.float32).reshape(1, -1)


def meta_predict(predictions: dict, method: str = "auto") -> dict:
    """
    Run meta-ensemble prediction combining all sub-model outputs.

    Args:
        predictions: Dict of sub-model raw scores and features.
        method:      'lr', 'xgb', 'auto', or 'voting' (weighted fallback).

    Returns:
        Dict with:
            - label: 'AI Generated' or 'Real Image'
            - confidence: float (0-100)
            - raw_score: float (0-1)
            - method: which meta-model was used
            - model_contributions: breakdown of input features
    """
    _load_meta_models()

    features = build_feature_vector(predictions)
    feature_names = [
        "ResNet50", "EfficientNet",
        "Patch Score", "FFT HF Ratio", "FFT Noise Var", "FFT Centroid"
    ]

    # Select model
    model = None
    method_used = method

    if method == "auto":
        if _xgb_model is not None:
            model = _xgb_model
            method_used = "xgboost"
        elif _lr_model is not None:
            model = _lr_model
            method_used = "logistic_regression"
        else:
            method_used = "weighted_voting"

    elif method == "xgb" and _xgb_model is not None:
        model = _xgb_model
        method_used = "xgboost"
    elif method == "lr" and _lr_model is not None:
        model = _lr_model
        method_used = "logistic_regression"
    else:
        method_used = "weighted_voting"

    # Run prediction
    if model is not None:
        try:
            proba = model.predict_proba(features)[0]
            # Assume class 1 = AI
            raw_score = float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception as e:
            logger.warning(f"Meta-model prediction failed: {e}. Falling back.")
            method_used = "weighted_voting"
            raw_score = _weighted_voting_score(predictions)
    else:
        raw_score = _weighted_voting_score(predictions)

    # Format result
    if raw_score >= 0.5:
        label = "AI Generated"
        confidence = round(raw_score * 100, 2)
    else:
        label = "Real Image"
        confidence = round((1 - raw_score) * 100, 2)

    # Model contributions
    contributions = {}
    for name, val in zip(feature_names, features[0]):
        contributions[name] = round(float(val), 4)

    result = {
        "label": label,
        "confidence": confidence,
        "raw_score": raw_score,
        "method": method_used,
        "model_contributions": contributions,
    }

    logger.info(f"Meta-ensemble [{method_used}]: {label} ({confidence}%)")
    return result


def _weighted_voting_score(predictions: dict) -> float:
    """Fallback weighted voting when meta-models unavailable."""
    weights = {
        "effnet_score": 0.50,
        "resnet_score": 0.35,
        "patch_score": 0.15,
    }

    total_weight = 0.0
    weighted_sum = 0.0

    for key, w in weights.items():
        score = predictions.get(key)
        if score is not None:
            weighted_sum += score * w
            total_weight += w

    if total_weight > 0:
        return weighted_sum / total_weight
    return 0.5


def is_trained() -> bool:
    """Check if any meta-model has been trained."""
    _load_meta_models()
    return _lr_model is not None or _xgb_model is not None
