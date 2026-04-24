"""
config.py — Centralized configuration for Real vs AI Image Detector.

All settings (paths, model configs, ensemble weights) in one place.
Supports .env overrides via python-dotenv.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────── PATHS ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "static", "gradcam")
FFT_FOLDER = os.path.join(BASE_DIR, "static", "fft")
SIMILARITY_FOLDER = os.path.join(BASE_DIR, "static", "similarity")
PATCH_FOLDER = os.path.join(BASE_DIR, "static", "patches")
EVAL_FOLDER = os.path.join(BASE_DIR, "static", "evaluation")
META_MODEL_DIR = os.path.join(BASE_DIR, "model", "meta")
ONNX_MODEL_DIR = os.path.join(BASE_DIR, "model", "onnx")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATASET_DIR = os.path.join(BASE_DIR, "dataset_v2")

# ──────────────────────────── FLASK ─────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"
MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_SIZE", 16 * 1024 * 1024))  # 16 MB

# ──────────────────────────── IMAGE ─────────────────────────────
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif"}

# ──────────────────────────── MODELS ────────────────────────────
MODEL_CONFIGS = {
    "resnet": {
        "path": os.path.join(BASE_DIR, "model_resnet.h5"),
        "img_size": (224, 224),
        "preprocess": "resnet",       # keras resnet50 preprocess_input
        "display_name": "ResNet50",
        "icon": "🧠",
        "gradcam_layer": "conv5_block3_out",
    },
    "efficientnet": {
        "path": os.path.join(BASE_DIR, "model", "efficientnet_trained.h5"),
        "img_size": (224, 224),
        "preprocess": "efficientnet", # keras efficientnet preprocess_input
        "display_name": "EfficientNet",
        "icon": "⚡",
        "gradcam_layer": "top_conv",
    },
}

# ──────────────────────────── ENSEMBLE ──────────────────────────
# Neural networks (85%) + FFT frequency analysis (15%)
ENSEMBLE_WEIGHTS = {
    "efficientnet": 0.45,
    "resnet": 0.40,
    "fft": 0.15,
}

# ──────────────────────────── PATCH ANALYSIS ────────────────────
PATCH_SIZE = 128                   # Default patch size for grid analysis
PATCH_STRIDE = None                # None = same as patch_size (no overlap)

# ──────────────────────────── HISTORY ───────────────────────────
MAX_HISTORY_ITEMS = 50

for _dir in [UPLOAD_FOLDER, GRADCAM_FOLDER, FFT_FOLDER, SIMILARITY_FOLDER,
             PATCH_FOLDER, EVAL_FOLDER, META_MODEL_DIR, ONNX_MODEL_DIR, LOG_DIR]:
    os.makedirs(_dir, exist_ok=True)
