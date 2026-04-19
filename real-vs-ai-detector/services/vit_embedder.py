"""
services/vit_embedder.py — Vision Transformer (ViT) embedding extraction.

Uses google/vit-base-patch16-224 from HuggingFace to extract
768-d CLS token embeddings for semantic similarity matching.
Thread-safe singleton with lazy initialization.
"""

import threading
import numpy as np
import cv2
from PIL import Image

from utils.logger import setup_logger

logger = setup_logger("services.vit_embedder")

# ──────────────────────────── SINGLETON ─────────────────────────
_vit_model = None
_vit_processor = None
_vit_lock = threading.Lock()
_VIT_MODEL_NAME = "google/vit-base-patch16-224"
_VIT_AVAILABLE = True


def _load_vit():
    """Load ViT model and processor from HuggingFace (lazy, thread-safe)."""
    global _vit_model, _vit_processor, _VIT_AVAILABLE

    if _vit_model is not None:
        return

    with _vit_lock:
        if _vit_model is not None:
            return

        try:
            import torch
            from transformers import ViTModel, ViTImageProcessor

            logger.info(f"Loading ViT model: {_VIT_MODEL_NAME}...")
            _vit_processor = ViTImageProcessor.from_pretrained(_VIT_MODEL_NAME)
            _vit_model = ViTModel.from_pretrained(_VIT_MODEL_NAME)
            _vit_model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                _vit_model = _vit_model.cuda()
                logger.info("ViT model loaded on GPU.")
            else:
                logger.info("ViT model loaded on CPU.")

            logger.info(f"ViT embedding dim: 768")

        except ImportError:
            _VIT_AVAILABLE = False
            logger.warning(
                "HuggingFace transformers not installed. "
                "ViT embeddings disabled. Install: pip install transformers torch"
            )
        except Exception as e:
            _VIT_AVAILABLE = False
            logger.error(f"Failed to load ViT model: {e}")


def is_available() -> bool:
    """Check if ViT embeddings are available."""
    if not _VIT_AVAILABLE:
        return False
    _load_vit()
    return _vit_model is not None


def _prepare_image(crop_bgr: np.ndarray) -> "Image.Image":
    """Convert BGR numpy array to PIL RGB image."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def extract_vit_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Extract CLS token embedding from a single BGR image crop.

    Args:
        crop_bgr: Cropped region as BGR numpy array (H, W, 3).

    Returns:
        1-D numpy array of shape (768,), L2-normalized.

    Raises:
        RuntimeError: If ViT model is not available.
    """
    import torch

    _load_vit()
    if _vit_model is None:
        raise RuntimeError("ViT model not available.")

    pil_img = _prepare_image(crop_bgr)

    # Preprocess
    inputs = _vit_processor(images=pil_img, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Extract CLS token
    with torch.no_grad():
        outputs = _vit_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, 768)

    # L2 normalize
    embedding = cls_embedding.cpu().numpy()[0]
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def extract_vit_embeddings_batch(crops: list) -> np.ndarray:
    """
    Extract CLS token embeddings from multiple BGR crops in one batch.

    Args:
        crops: List of BGR numpy arrays, each (H, W, 3).

    Returns:
        2-D numpy array of shape (N, 768), L2-normalized rows.
    """
    import torch

    if not crops:
        return np.array([])

    _load_vit()
    if _vit_model is None:
        raise RuntimeError("ViT model not available.")

    pil_images = [_prepare_image(crop) for crop in crops]

    # Batch preprocess
    inputs = _vit_processor(images=pil_images, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Extract CLS tokens
    with torch.no_grad():
        outputs = _vit_model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (N, 768)

    embeddings = cls_embeddings.cpu().numpy()

    # L2 normalize each row
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    embeddings = embeddings / norms

    return embeddings
