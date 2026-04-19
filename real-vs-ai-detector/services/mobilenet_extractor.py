"""
services/mobilenet_extractor.py — MobileNetV3 feature embedding extraction.

Uses MobileNetV3Large (headless) to extract 960-d feature vectors from image regions.
Thread-safe singleton pattern with lazy initialization.
"""

import threading
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

from utils.logger import setup_logger

logger = setup_logger("services.mobilenet_extractor")

# ──────────────────────────── SINGLETON ─────────────────────────
_mobilenet_model = None
_mobilenet_lock = threading.Lock()
_IMG_SIZE = (224, 224)


def _get_mobilenet_model():
    """
    Build and cache a headless MobileNetV3Large model.
    Produces a 960-d normalized embedding vector.
    """
    global _mobilenet_model
    if _mobilenet_model is not None:
        return _mobilenet_model

    with _mobilenet_lock:
        if _mobilenet_model is not None:
            return _mobilenet_model

        logger.info("Building feature extraction model from MobileNetV3Large…")

        # Load headless MobileNetV3Large with GlobalAveragePooling built-in.
        base_model = tf.keras.applications.MobileNetV3Large(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )
        
        # L2 normalization for cosine similarity
        outputs = tf.keras.layers.Lambda(
            lambda v: tf.math.l2_normalize(v, axis=1),
            name="l2_norm"
        )(base_model.output)

        _mobilenet_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=outputs,
            name="mobilenet_v3_features"
        )

        embedding_dim = _mobilenet_model.output_shape[-1]
        logger.info(f"MobileNetV3 model ready. Embedding dim: {embedding_dim}")

    return _mobilenet_model


def _preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess a BGR crop for MobileNetV3 feature extraction.
    Forces grayscale conversion to match shape/vectors while ignoring colors.

    Args:
        crop_bgr: Cropped region as BGR numpy array (H, W, 3).

    Returns:
        Preprocessed array of shape (1, 224, 224, 3).
    """
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

    # Strip color to force shape/texture matching only
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    pil_img = Image.fromarray(gray_3c).resize(_IMG_SIZE)
    arr = np.array(pil_img, dtype="float32")
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def extract_mobilenet_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Extract a normalized feature vector from a single BGR image crop.

    Args:
        crop_bgr: Cropped region as BGR numpy array (H, W, 3).

    Returns:
        1-D numpy array of shape (960,), L2-normalized.
    """
    model = _get_mobilenet_model()
    img_array = _preprocess_crop(crop_bgr)
    embedding = model.predict(img_array, verbose=0)[0]
    return embedding


def extract_mobilenet_embeddings_batch(crops: list) -> np.ndarray:
    """
    Extract normalized feature vectors from multiple BGR crops in one batch.

    Args:
        crops: List of BGR numpy arrays, each (H, W, 3).

    Returns:
        2-D numpy array of shape (N, 960), L2-normalized rows.
    """
    if not crops:
        return np.array([])

    model = _get_mobilenet_model()

    # Build batch
    batch = np.concatenate(
        [_preprocess_crop(crop) for crop in crops],
        axis=0
    )

    embeddings = model.predict(batch, verbose=0)
    return embeddings
