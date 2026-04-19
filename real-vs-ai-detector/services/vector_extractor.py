"""
services/vector_extractor.py — Vector/HOG Geometry embedding extraction.

Utilizes Histogram of Oriented Gradients (HOG) to extract pure geometrical structure pipelines (edges, angles, lines) to match shapes mathematically independently of colors and generic neural network classes. 
"""

import cv2
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("services.vector_extractor")

# ──────────────────────────── SINGLETON ─────────────────────────
_hog_descriptor = None

def _get_hog_descriptor():
    """Builds and caches a tuned OpenCV HOG configuration."""
    global _hog_descriptor
    if _hog_descriptor is None:
        logger.info("Initializing HOG geometric vector extractor…")
        win_size = (224, 224)
        block_size = (32, 32)
        block_stride = (16, 16)
        cell_size = (16, 16)
        nbins = 9
        _hog_descriptor = cv2.HOGDescriptor(
            win_size, block_size, block_stride, cell_size, nbins
        )
    return _hog_descriptor


def extract_vector_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Computes a mathematical geometric vector track of the shapes in the image.
    
    Args:
        crop_bgr: Cropped region as BGR numpy array.
        
    Returns:
        1-D numpy array of shape (6084,), L2-normalized.
    """
    hog = _get_hog_descriptor()
    
    # 1. Immediately drop color via Grayscale 
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize to standardization bounds so big/small shapes scale identically
    resized = cv2.resize(crop_gray, (224, 224))
    
    # 3. Compute Orientation Histogram
    embedding = hog.compute(resized).flatten()
    
    # 4. L2-Normalize for perfect cosine similarity scaling
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
        
    return embedding


def extract_vector_embeddings_batch(crops: list) -> np.ndarray:
    """
    Extracts L2-normalized geometric structure vectors from multiple image crops.
    
    Args:
        crops: List of BGR numpy arrays.
        
    Returns:
        2-D numpy array of shape (N, 6084).
    """
    if not crops:
        return np.array([])
        
    embeddings = [extract_vector_embedding(c) for c in crops]
    return np.array(embeddings)
