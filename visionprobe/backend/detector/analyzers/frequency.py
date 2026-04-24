"""
DCT Frequency Spectrum Analysis for AI vs Real detection.
Detects mid-range periodicity from diffusion models and spectral anomalies from GANs.
"""

import logging
import numpy as np
import cv2
from scipy import fft as scipy_fft
from scipy.stats import gmean

logger = logging.getLogger("visionprobe.detector.analyzers.frequency")


def analyze_frequency(np_img: np.ndarray) -> dict:
    """
    Compute DCT frequency spectrum and extract features for AI detection.
    
    Returns dict with:
      - score: float (0=real, 1=AI)
      - features: list of 6 floats
      - spectrum_2d_small: list of lists (64x64 downsampled for frontend)
      - profile_1d: list of floats (azimuthal average)
      - feature_names: list of feature name strings
    """
    try:
        # 1. Convert to grayscale
        if np_img.ndim == 3:
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = np_img.copy()

        # Resize for consistent analysis
        target_size = 512
        gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # 2. 2D DCT
        dct2d = scipy_fft.dctn(gray.astype(np.float64), norm="ortho")

        # 3. Log magnitude
        log_mag = np.log(np.abs(dct2d) + 1e-8)

        # 4. Shift zero-frequency to center
        shifted = np.fft.fftshift(log_mag)

        # 5. Azimuthal average
        H, W = shifted.shape
        cy, cx = H // 2, W // 2
        Y, X = np.mgrid[0:H, 0:W]
        R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        max_r = int(min(cx, cy))
        profile = []
        for r in range(0, max_r, 2):
            ring_mask = (R >= r) & (R < r + 2)
            if ring_mask.any():
                ring_mean = float(np.mean(shifted[ring_mask]))
            else:
                ring_mean = 0.0
            profile.append(ring_mean)

        profile = np.array(profile)

        # 6. Compute features
        eps = 1e-8
        low_band = profile[:10] if len(profile) >= 10 else profile
        mid_band = profile[10:30] if len(profile) >= 30 else profile[10:]
        high_band = profile[40:] if len(profile) >= 40 else profile[-10:]

        low_mean = float(np.mean(low_band)) if len(low_band) > 0 else eps
        mid_mean = float(np.mean(mid_band)) if len(mid_band) > 0 else 0.0
        high_mean = float(np.mean(high_band)) if len(high_band) > 0 else 0.0

        # Mid-frequency energy (normalized)
        mid_freq_energy = mid_mean / (abs(low_mean) + eps)

        # High-frequency ratio
        high_freq_ratio = high_mean / (abs(low_mean) + eps)

        # Spectral flatness: geometric mean / arithmetic mean
        profile_positive = np.clip(np.exp(profile), eps, None)
        arith_mean = float(np.mean(profile_positive))
        geo_mean = float(gmean(profile_positive)) if len(profile_positive) > 0 else eps
        spectral_flatness = geo_mean / (arith_mean + eps)

        # Peak prominence
        peak_prominence = float(np.max(profile) - np.mean(profile))

        # Periodicity score via autocorrelation
        if len(profile) > 2:
            centered = profile - np.mean(profile)
            autocorr = np.correlate(centered, centered, mode="full")
            autocorr = autocorr[len(autocorr) // 2:]  # positive lags only
            if autocorr[0] != 0:
                autocorr = autocorr / autocorr[0]
            # Skip lag 0, find max in lags 2-20
            search_range = autocorr[2:min(20, len(autocorr))]
            periodicity_score = float(np.max(search_range)) if len(search_range) > 0 else 0.0
        else:
            periodicity_score = 0.0

        # Mid-to-low ratio
        mid_to_low_ratio = mid_mean / (abs(low_mean) + eps)

        features = [
            mid_freq_energy,
            high_freq_ratio,
            spectral_flatness,
            peak_prominence,
            periodicity_score,
            mid_to_low_ratio,
        ]

        # 7. Compute AI score from features
        # Heuristic scoring based on known patterns
        ai_indicators = 0.0
        total_checks = 0.0

        # Diffusion models show elevated mid-frequency energy
        if mid_freq_energy > 0.8:
            ai_indicators += 1.0
        elif mid_freq_energy > 0.6:
            ai_indicators += 0.5
        total_checks += 1.0

        # GANs: spectral flatness outside natural range
        if spectral_flatness < 0.35 or spectral_flatness > 0.65:
            ai_indicators += 1.0
        elif spectral_flatness < 0.40 or spectral_flatness > 0.60:
            ai_indicators += 0.4
        total_checks += 1.0

        # High periodicity → AI
        if periodicity_score > 0.5:
            ai_indicators += 0.8
        elif periodicity_score > 0.3:
            ai_indicators += 0.3
        total_checks += 1.0

        # Low high-frequency content → AI (diffusion models have smooth textures)
        if abs(high_freq_ratio) < 0.3:
            ai_indicators += 0.6
        total_checks += 1.0

        freq_ai_score = float(np.clip(ai_indicators / total_checks, 0.0, 1.0))

        # 8. Downsample spectrum for frontend (64×64)
        small_size = 64
        spectrum_small = cv2.resize(shifted, (small_size, small_size), interpolation=cv2.INTER_AREA)
        spectrum_2d_small = spectrum_small.tolist()

        return {
            "score": freq_ai_score,
            "features": features,
            "spectrum_2d_small": spectrum_2d_small,
            "profile_1d": profile.tolist(),
            "feature_names": [
                "mid_freq_energy",
                "high_freq_ratio",
                "spectral_flatness",
                "peak_prominence",
                "periodicity_score",
                "mid_to_low_ratio",
            ],
        }

    except Exception as e:
        logger.error(f"Frequency analysis failed: {e}", exc_info=True)
        return {
            "score": 0.5,
            "features": [0.0] * 6,
            "spectrum_2d_small": [],
            "profile_1d": [],
            "feature_names": [],
        }
