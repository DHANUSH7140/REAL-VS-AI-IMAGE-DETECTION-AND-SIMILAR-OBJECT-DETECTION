"""
utils/preprocessing.py — Image preprocessing for each model type.

Handles loading, resizing, and model-specific normalization.
Fixes the original bug where Image.open() was missing.
"""

import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

from config import MODEL_CONFIGS


def preprocess_image(image_path: str, model_name: str) -> np.ndarray:
    """
    Load an image, resize it, and apply model-specific preprocessing.

    Args:
        image_path: Absolute path to the image file.
        model_name: One of 'cnn', 'resnet', 'efficientnet'.

    Returns:
        NumPy array of shape (1, H, W, 3) ready for model.predict().
    """
    cfg = MODEL_CONFIGS.get(model_name)
    if cfg is None:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_CONFIGS.keys())}")

    # Open and convert to RGB
    img = Image.open(image_path).convert("RGB")
    img = img.resize(cfg["img_size"])

    arr = np.array(img, dtype="float32")

    # Apply model-specific preprocessing
    preprocess_type = cfg["preprocess"]
    if preprocess_type == "resnet":
        arr = resnet_preprocess(arr)
    elif preprocess_type == "efficientnet":
        arr = effnet_preprocess(arr)
    else:
        # Custom CNN: simple rescale to [0, 1]
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0)  # shape (1, H, W, 3)
