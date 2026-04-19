"""
services/similarity_matcher.py — Cosine similarity matching between image regions.

Compares a user-selected ROI embedding against all detected object embeddings
to find visually similar regions within the same image.
"""

import numpy as np

from utils.logger import setup_logger

logger = setup_logger("services.similarity_matcher")


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Both vectors are assumed to be L2-normalized (from feature_extractor),
    so similarity = dot product.

    Args:
        vec_a: 1-D numpy array.
        vec_b: 1-D numpy array.

    Returns:
        Cosine similarity in range [-1, 1].
    """
    dot = np.dot(vec_a, vec_b)
    # Clamp to valid range (handles floating-point edge cases)
    return float(np.clip(dot, -1.0, 1.0))


def cosine_similarity_batch(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between one query and many candidates.

    Both are assumed L2-normalized.

    Args:
        query:       1-D array of shape (D,).
        candidates:  2-D array of shape (N, D).

    Returns:
        1-D array of shape (N,) with similarity scores.
    """
    if candidates.size == 0:
        return np.array([])

    scores = candidates @ query  # (N, D) @ (D,) → (N,)
    return np.clip(scores, -1.0, 1.0)


def find_similar_regions(
    roi_embedding: np.ndarray,
    detections: list,
    object_embeddings: np.ndarray,
    threshold: float = 0.7,
) -> list:
    """
    Find detected objects that are visually similar to the user-selected ROI.

    Args:
        roi_embedding:     1-D feature vector for the ROI crop.
        detections:        List of detection dicts (each with 'bbox', 'class_name', etc.).
        object_embeddings: 2-D array of shape (N, D), one row per detection.
        threshold:         Minimum cosine similarity to consider a match.

    Returns:
        List of match dicts sorted by similarity (descending), each with:
            - index:       Detection index
            - box:         [x1, y1, x2, y2]
            - class_name:  Detected object class
            - confidence:  YOLO detection confidence
            - similarity:  Cosine similarity score (0–1)
    """
    if object_embeddings.size == 0:
        logger.info("No object embeddings to compare against.")
        return []

    # Compute all similarities at once
    similarities = cosine_similarity_batch(roi_embedding, object_embeddings)

    matches = []
    for i, (det, sim) in enumerate(zip(detections, similarities)):
        sim_val = float(sim)
        if sim_val >= threshold:
            matches.append({
                "index": i,
                "box": det["bbox"],
                "class_name": det.get("class_name", "unknown"),
                "confidence": det.get("confidence", 0.0),
                "similarity": round(sim_val, 4),
            })

    # Sort by similarity descending
    matches.sort(key=lambda m: m["similarity"], reverse=True)

    logger.info(
        f"Similarity matching: {len(matches)}/{len(detections)} regions "
        f"above threshold {threshold}"
    )

    return matches
