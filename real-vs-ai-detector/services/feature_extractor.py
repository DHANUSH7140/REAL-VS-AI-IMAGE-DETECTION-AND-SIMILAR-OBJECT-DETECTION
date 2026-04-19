"""
services/feature_extractor.py — Deep feature embedding extraction.

Uses the existing EfficientNet model (with classification head removed)
to extract 1280-d feature vectors from image regions for similarity matching.
Thread-safe singleton pattern with lazy initialization.
"""

import threading
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

from config import MODEL_CONFIGS
from models.loader import model_manager
from utils.logger import setup_logger

logger = setup_logger("services.feature_extractor")

# ──────────────────────────── SINGLETON ─────────────────────────
_feature_model = None
_feature_lock = threading.Lock()
_IMG_SIZE = (224, 224)


def _get_feature_model():
    """
    Build and cache a headless feature extraction model from EfficientNet.

    Strips the classification head and adds GlobalAveragePooling2D
    to produce a 1280-d embedding vector per input image.
    """
    global _feature_model
    if _feature_model is not None:
        return _feature_model

    with _feature_lock:
        if _feature_model is not None:
            return _feature_model

        logger.info("Building feature extraction model from EfficientNet…")

        # Load the full classifier
        full_model = model_manager.get("efficientnet")

        # Find the last Conv / pooling output before the dense head.
        # EfficientNet architecture: ... → top_conv → top_bn → top_activation → (head)
        # We want the output right after global pooling.
        # Strategy: build a new model that outputs the penultimate dense-ready features.
        target_layer = None

        # Try to find 'top_activation' or 'top_bn' in the model or sub-models
        for layer_name_candidate in ["top_activation", "top_bn", "top_conv"]:
            try:
                target_layer = full_model.get_layer(layer_name_candidate)
                break
            except ValueError:
                # Try inside nested models (transfer learning wrappers)
                for layer in full_model.layers:
                    if hasattr(layer, "get_layer"):
                        try:
                            target_layer = layer.get_layer(layer_name_candidate)
                            break
                        except ValueError:
                            continue
                if target_layer is not None:
                    break

        if target_layer is None:
            # Fallback: find the last non-Dense, non-Dropout layer
            for layer in reversed(full_model.layers):
                if not isinstance(layer, (tf.keras.layers.Dense,
                                          tf.keras.layers.Dropout,
                                          tf.keras.layers.Flatten)):
                    target_layer = layer
                    break

        if target_layer is None:
            raise RuntimeError("Could not find a suitable feature layer in EfficientNet.")

        # Build feature model: input → target_layer → GlobalAveragePooling → L2 normalize
        x = target_layer.output
        # Add global average pooling if the output is spatial (4D)
        if len(x.shape) == 4:
            x = tf.keras.layers.GlobalAveragePooling2D(name="feature_gap")(x)
        # L2 normalization for cosine similarity
        x = tf.keras.layers.Lambda(
            lambda v: tf.math.l2_normalize(v, axis=1),
            name="l2_norm"
        )(x)

        _feature_model = tf.keras.Model(
            inputs=full_model.input,
            outputs=x,
            name="efficientnet_features"
        )

        embedding_dim = _feature_model.output_shape[-1]
        logger.info(f"Feature model ready. Embedding dim: {embedding_dim}")

    return _feature_model


def _preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess a BGR crop for EfficientNet feature extraction.

    Args:
        crop_bgr: Cropped region as BGR numpy array (H, W, 3).

    Returns:
        Preprocessed array of shape (1, 224, 224, 3).
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb).resize(_IMG_SIZE)
    arr = np.array(pil_img, dtype="float32")
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def extract_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Extract a normalized feature vector from a single BGR image crop.

    Args:
        crop_bgr: Cropped region as BGR numpy array (H, W, 3).

    Returns:
        1-D numpy array of shape (embedding_dim,), L2-normalized.
    """
    model = _get_feature_model()
    img_array = _preprocess_crop(crop_bgr)
    embedding = model.predict(img_array, verbose=0)[0]
    return embedding


def extract_embeddings_batch(crops: list) -> np.ndarray:
    """
    Extract normalized feature vectors from multiple BGR crops in one batch.

    Args:
        crops: List of BGR numpy arrays, each (H, W, 3).

    Returns:
        2-D numpy array of shape (N, embedding_dim), L2-normalized rows.
    """
    if not crops:
        return np.array([])

    model = _get_feature_model()

    # Build batch
    batch = np.concatenate(
        [_preprocess_crop(crop) for crop in crops],
        axis=0
    )

    embeddings = model.predict(batch, verbose=0)
    return embeddings
