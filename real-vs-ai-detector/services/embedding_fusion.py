"""
services/embedding_fusion.py — Combine embeddings from multiple sources.

Fuses ViT (768-d), EfficientNet (1792-d), and RF-DETR embeddings
via concatenation + L2 normalization for unified similarity search.
"""

import numpy as np

from utils.logger import setup_logger

logger = setup_logger("services.embedding_fusion")


def fuse_embeddings(
    vit_embedding: np.ndarray = None,
    effnet_embedding: np.ndarray = None,
    rfdetr_embedding: np.ndarray = None,
    weights: dict = None,
) -> np.ndarray:
    """
    Fuse embeddings from multiple sources into a single vector.

    Strategy: concatenate available embeddings, apply optional weighting,
    then L2-normalize the result for cosine similarity.

    Args:
        vit_embedding:    768-d ViT CLS token vector (or None).
        effnet_embedding: 1792-d EfficientNet feature vector (or None).
        rfdetr_embedding: RF-DETR encoder features (or None).
        weights:          Optional dict mapping source → weight multiplier.

    Returns:
        L2-normalized fused embedding vector.

    Raises:
        ValueError: If no embeddings are provided.
    """
    if weights is None:
        weights = {"vit": 1.0, "effnet": 0.6, "rfdetr": 0.8}

    parts = []

    if vit_embedding is not None:
        parts.append(vit_embedding * weights.get("vit", 1.0))

    if effnet_embedding is not None:
        parts.append(effnet_embedding * weights.get("effnet", 0.6))

    if rfdetr_embedding is not None:
        parts.append(rfdetr_embedding * weights.get("rfdetr", 0.8))

    if not parts:
        raise ValueError("No embeddings provided for fusion.")

    fused = np.concatenate(parts)

    # L2 normalize
    norm = np.linalg.norm(fused)
    if norm > 0:
        fused = fused / norm

    return fused


def fuse_embeddings_batch(
    vit_embeddings: np.ndarray = None,
    effnet_embeddings: np.ndarray = None,
    weights: dict = None,
) -> np.ndarray:
    """
    Fuse batches of embeddings (ViT + EfficientNet).

    Args:
        vit_embeddings:    (N, 768) array or None.
        effnet_embeddings: (N, 1792) array or None.
        weights:           Optional weight dict.

    Returns:
        (N, D) array of L2-normalized fused embeddings.
    """
    if weights is None:
        weights = {"vit": 1.0, "effnet": 0.6}

    parts = []

    if vit_embeddings is not None and vit_embeddings.size > 0:
        parts.append(vit_embeddings * weights.get("vit", 1.0))

    if effnet_embeddings is not None and effnet_embeddings.size > 0:
        parts.append(effnet_embeddings * weights.get("effnet", 0.6))

    if not parts:
        return np.array([])

    fused = np.concatenate(parts, axis=1)

    # L2 normalize each row
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    fused = fused / norms

    logger.debug(f"Fused embedding shape: {fused.shape}")
    return fused
