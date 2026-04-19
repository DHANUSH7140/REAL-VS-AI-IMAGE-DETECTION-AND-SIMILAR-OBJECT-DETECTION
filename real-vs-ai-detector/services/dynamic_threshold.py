"""
services/dynamic_threshold.py — Adaptive threshold selection for similarity matching.

Provides multiple strategies to dynamically determine which objects
are "similar enough" to the user's ROI, beyond a simple fixed threshold.
"""

import numpy as np

from utils.logger import setup_logger

logger = setup_logger("services.dynamic_threshold")


def top_k(scores: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Select the top-K most similar objects.

    Args:
        scores: 1-D array of similarity scores.
        k:      Number of top matches to return.

    Returns:
        Boolean mask of selected indices.
    """
    if len(scores) == 0:
        return np.array([], dtype=bool)

    k = min(k, len(scores))
    top_indices = np.argsort(scores)[-k:][::-1]

    mask = np.zeros(len(scores), dtype=bool)
    mask[top_indices] = True

    logger.debug(f"Top-K (k={k}): selected {mask.sum()} objects")
    return mask


def statistical(scores: np.ndarray, multiplier: float = 0.5) -> np.ndarray:
    """
    Select objects above (mean + multiplier * std) threshold.

    This adapts to the distribution of similarity scores in the image.

    Args:
        scores:     1-D array of similarity scores.
        multiplier: Standard deviation multiplier (0.5 = moderate, 1.0 = strict).

    Returns:
        Boolean mask of selected indices.
    """
    if len(scores) == 0:
        return np.array([], dtype=bool)

    mean = np.mean(scores)
    std = np.std(scores)
    threshold = mean + multiplier * std

    mask = scores >= threshold

    logger.debug(
        f"Statistical (mean={mean:.3f}, std={std:.3f}, "
        f"threshold={threshold:.3f}): selected {mask.sum()} objects"
    )
    return mask


def percentile(scores: np.ndarray, pct: float = 90.0) -> np.ndarray:
    """
    Select objects in the top percentile of similarity scores.

    Args:
        scores: 1-D array of similarity scores.
        pct:    Percentile threshold (90 = top 10%).

    Returns:
        Boolean mask of selected indices.
    """
    if len(scores) == 0:
        return np.array([], dtype=bool)

    threshold = np.percentile(scores, pct)
    mask = scores >= threshold

    logger.debug(
        f"Percentile (p{pct:.0f}, threshold={threshold:.3f}): "
        f"selected {mask.sum()} objects"
    )
    return mask


def adaptive_select(
    scores: np.ndarray,
    method: str = "statistical",
    **kwargs,
) -> dict:
    """
    Apply adaptive threshold selection and return results.

    Args:
        scores: 1-D array of cosine similarity scores for all detections.
        method: 'top_k', 'statistical', 'percentile', or 'combined'.
        **kwargs: Method-specific parameters.

    Returns:
        Dict with:
            - mask:        Boolean array of selected indices
            - threshold:   The effective threshold value
            - method:      Method name used
            - stats:       Additional statistics
    """
    if len(scores) == 0:
        return {
            "mask": np.array([], dtype=bool),
            "threshold": 0.0,
            "method": method,
            "stats": {},
        }

    if method == "top_k":
        k = kwargs.get("k", 5)
        mask = top_k(scores, k=k)
        threshold = float(np.min(scores[mask])) if mask.any() else 0.0

    elif method == "percentile":
        pct = kwargs.get("pct", 90.0)
        threshold = float(np.percentile(scores, pct))
        mask = scores >= threshold

    elif method == "combined":
        # Intersection of statistical + percentile for robust selection
        stat_mask = statistical(scores, multiplier=kwargs.get("multiplier", 0.5))
        pct_mask = percentile(scores, pct=kwargs.get("pct", 85.0))
        mask = stat_mask & pct_mask
        threshold = float(np.min(scores[mask])) if mask.any() else 0.0

    else:  # statistical (default)
        multiplier = kwargs.get("multiplier", 0.5)
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        threshold = mean + multiplier * std
        mask = scores >= threshold

    # Compute stats
    stats = {
        "mean": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores)), 4),
        "min": round(float(np.min(scores)), 4),
        "max": round(float(np.max(scores)), 4),
        "selected_count": int(mask.sum()),
        "total_count": len(scores),
    }

    logger.info(
        f"Adaptive threshold [{method}]: selected {mask.sum()}/{len(scores)} "
        f"(threshold={threshold:.3f})"
    )

    return {
        "mask": mask,
        "threshold": round(threshold, 4),
        "method": method,
        "stats": stats,
    }
