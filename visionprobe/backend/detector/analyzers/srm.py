"""
SRM (Spatial Rich Model) residual noise analysis for AI vs Real detection.
Real cameras leave specific noise signatures; AI images have different residual patterns.
"""

import logging
import os
import pickle

import numpy as np
import cv2
from scipy.stats import kurtosis, skew

from model_cache import ModelCache

logger = logging.getLogger("visionprobe.detector.analyzers.srm")

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")

# SRM filter kernels (3×3)
SRM_FILTERS = [
    np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=np.float64),   # horizontal
    np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float64),   # vertical
    np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]], dtype=np.float64),  # diagonal
]

# Truncation threshold
SRM_TRUNCATION = 2.0


def compute_srm_residuals(np_img: np.ndarray) -> np.ndarray:
    """
    Apply 3 SRM residual filters to each of 3 RGB channels.
    
    Returns:
      np.ndarray of shape (H, W, 9) — 9 residual channels
      Truncated to [-T, T] with T=2.0
    """
    if np_img.ndim != 3 or np_img.shape[2] < 3:
        # Convert grayscale to 3-channel
        if np_img.ndim == 2:
            np_img = np.stack([np_img] * 3, axis=-1)
        else:
            np_img = np.concatenate([np_img] * 3, axis=-1)[:, :, :3]

    H, W, C = np_img.shape
    residuals = np.zeros((H, W, 9), dtype=np.float64)

    for f_idx, filt in enumerate(SRM_FILTERS):
        for c_idx in range(3):
            channel = np_img[:, :, c_idx].astype(np.float64)
            residual = cv2.filter2D(channel, -1, filt)
            residuals[:, :, f_idx * 3 + c_idx] = residual

    # Truncate to [-T, T]
    residuals = np.clip(residuals, -SRM_TRUNCATION, SRM_TRUNCATION)
    return residuals


def analyze_srm(residuals: np.ndarray) -> dict:
    """
    Compute statistical features from SRM residuals and classify.
    
    For each of 9 residual channels: mean, variance, kurtosis, skewness → 36 features.
    
    Returns dict with:
      - score: float (0=real, 1=AI)
      - noise_stats: list of 12 floats (summary for meta-learner)
      - channel_stats: list of 36 floats (full stats)
    """
    try:
        num_channels = residuals.shape[2] if residuals.ndim == 3 else 1
        channel_stats = []

        for c in range(min(num_channels, 9)):
            if residuals.ndim == 3:
                ch = residuals[:, :, c].flatten()
            else:
                ch = residuals.flatten()

            ch_mean = float(np.mean(ch))
            ch_var = float(np.var(ch))
            ch_kurt = float(kurtosis(ch, fisher=True))
            ch_skew = float(skew(ch))
            channel_stats.extend([ch_mean, ch_var, ch_kurt, ch_skew])

        # Pad to 36 if fewer channels
        while len(channel_stats) < 36:
            channel_stats.append(0.0)
        channel_stats = channel_stats[:36]

        # Summary stats for meta-learner (12 dims: aggregate across channels)
        means = [channel_stats[i * 4] for i in range(9)]
        variances = [channel_stats[i * 4 + 1] for i in range(9)]
        kurtoses = [channel_stats[i * 4 + 2] for i in range(9)]

        noise_stats = [
            float(np.mean(means)),
            float(np.std(means)),
            float(np.mean(variances)),
            float(np.std(variances)),
            float(np.mean(kurtoses)),
            float(np.std(kurtoses)),
            float(np.max(kurtoses)),
            float(np.min(kurtoses)),
            float(np.median(variances)),
            float(np.max(variances)),
            float(np.mean([abs(s) for s in channel_stats[3::4]])),  # mean abs skewness
            float(np.std([abs(s) for s in channel_stats[3::4]])),   # std abs skewness
        ]

        # Classification
        cache = ModelCache.get_instance()
        if cache.srm_available and cache.srm_classifier is not None:
            try:
                features_arr = np.array(channel_stats).reshape(1, -1)
                probs = cache.srm_classifier.predict_proba(features_arr)[0]
                srm_ai_score = float(probs[1])  # class 1 = AI
            except Exception as e:
                logger.warning(f"SRM classifier failed, using heuristic: {e}")
                srm_ai_score = _heuristic_srm_score(kurtoses, variances)
        else:
            srm_ai_score = _heuristic_srm_score(kurtoses, variances)

        return {
            "score": srm_ai_score,
            "noise_stats": noise_stats,
            "channel_stats": channel_stats,
        }

    except Exception as e:
        logger.error(f"SRM analysis failed: {e}", exc_info=True)
        return {
            "score": 0.5,
            "noise_stats": [0.0] * 12,
            "channel_stats": [0.0] * 36,
        }


def _heuristic_srm_score(kurtoses: list, variances: list) -> float:
    """
    Heuristic SRM scoring when classifier is unavailable.
    Real cameras: higher kurtosis from sensor noise patterns.
    AI images: residuals closer to Gaussian (lower kurtosis).
    """
    mean_abs_kurtosis = float(np.mean([abs(k) for k in kurtoses]))
    mean_variance = float(np.mean(variances))

    # Higher kurtosis → more likely real (camera noise has heavy tails)
    if mean_abs_kurtosis > 6.0:
        kurtosis_signal = 0.2  # very likely real
    elif mean_abs_kurtosis > 4.0:
        kurtosis_signal = 0.35
    elif mean_abs_kurtosis > 2.0:
        kurtosis_signal = 0.55
    else:
        kurtosis_signal = 0.75  # low kurtosis → likely AI

    # Very low variance in residuals → likely AI (too smooth)
    if mean_variance < 0.1:
        variance_signal = 0.8
    elif mean_variance < 0.5:
        variance_signal = 0.6
    else:
        variance_signal = 0.3

    return float(np.clip(0.6 * kurtosis_signal + 0.4 * variance_signal, 0.0, 1.0))
