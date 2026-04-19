"""
services/fft_features.py — FFT-based frequency domain analysis.

Extracts frequency features from images to detect subtle AI-generation
artifacts that are invisible in the spatial domain.
"""

import os
import uuid
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FFT_FOLDER
from utils.logger import setup_logger

logger = setup_logger("services.fft_features")


def extract_fft_features(image_path: str) -> dict:
    """
    Perform FFT analysis on an image and extract frequency-domain features.

    AI-generated images often show different frequency distribution patterns:
    - Lower high-frequency energy (smoother textures)
    - Different spectral centroids
    - Lower noise variance (too-clean appearance)

    Args:
        image_path: Absolute path to the image file.

    Returns:
        Dict with:
            - high_freq_ratio:  Ratio of high-frequency energy to total energy
            - spectral_centroid: Center of mass of the frequency spectrum
            - noise_variance:    Variance of high-frequency noise
            - spectrum_url:      URL to the saved spectrum visualization
            - analysis_summary:  Human-readable analysis string
    """
    try:
        # Load image in grayscale for frequency analysis
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"error": "Could not load image for FFT analysis."}

        # Resize to standard size for consistent analysis
        img = cv2.resize(img, (256, 256))

        # ── Compute 2D FFT ──────────────────────────────────────────
        f_transform = np.fft.fft2(img.astype(np.float64))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        magnitude_log = np.log1p(magnitude)

        # ── Extract features ────────────────────────────────────────
        rows, cols = img.shape
        crow, ccol = rows // 2, cols // 2

        # Create masks for low and high frequency regions
        radius_low = min(rows, cols) // 8   # Inner radius (low freq)
        radius_high = min(rows, cols) // 4  # Mid radius

        # Total energy
        total_energy = np.sum(magnitude ** 2)

        # Low frequency energy (center of spectrum)
        y, x = np.ogrid[:rows, :cols]
        low_mask = (x - ccol) ** 2 + (y - crow) ** 2 <= radius_low ** 2
        low_energy = np.sum(magnitude[low_mask] ** 2)

        # High frequency energy (outside mid radius)
        high_mask = (x - ccol) ** 2 + (y - crow) ** 2 > radius_high ** 2
        high_energy = np.sum(magnitude[high_mask] ** 2)

        high_freq_ratio = float(high_energy / (total_energy + 1e-10))

        # Spectral centroid (weighted average of frequency positions)
        freq_x = np.arange(cols) - ccol
        freq_y = np.arange(rows) - crow
        freq_xx, freq_yy = np.meshgrid(freq_x, freq_y)
        freq_radius = np.sqrt(freq_xx ** 2 + freq_yy ** 2)
        spectral_centroid = float(
            np.sum(freq_radius * magnitude) / (np.sum(magnitude) + 1e-10)
        )

        # Noise variance (Laplacian-based)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        noise_variance = float(laplacian.var())

        # ── Generate spectrum visualization ─────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor('#0b0f19')

        # Original image
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original (Grayscale)', color='white', fontsize=10)
        axes[0].axis('off')

        # FFT Magnitude Spectrum
        axes[1].imshow(magnitude_log, cmap='inferno')
        axes[1].set_title('FFT Magnitude Spectrum', color='white', fontsize=10)
        axes[1].axis('off')

        # Azimuthal average (radial power spectrum)
        max_radius = int(np.sqrt(crow ** 2 + ccol ** 2))
        radial_profile = np.zeros(max_radius)
        count = np.zeros(max_radius)
        for iy in range(rows):
            for ix in range(cols):
                r = int(np.sqrt((iy - crow) ** 2 + (ix - ccol) ** 2))
                if r < max_radius:
                    radial_profile[r] += magnitude[iy, ix]
                    count[r] += 1
        count[count == 0] = 1
        radial_profile /= count

        axes[2].plot(radial_profile[:max_radius // 2], color='#818cf8', linewidth=1.5)
        axes[2].fill_between(
            range(max_radius // 2),
            radial_profile[:max_radius // 2],
            alpha=0.3, color='#6366f1'
        )
        axes[2].set_title('Radial Power Spectrum', color='white', fontsize=10)
        axes[2].set_xlabel('Frequency', color='#94a3b8', fontsize=9)
        axes[2].set_ylabel('Magnitude', color='#94a3b8', fontsize=9)
        axes[2].set_facecolor('#0b0f19')
        axes[2].tick_params(colors='#94a3b8')
        axes[2].spines['bottom'].set_color('#333')
        axes[2].spines['left'].set_color('#333')
        axes[2].spines['top'].set_visible(False)
        axes[2].spines['right'].set_visible(False)

        plt.tight_layout()

        filename = f"fft_{uuid.uuid4().hex[:12]}.png"
        save_path = os.path.join(FFT_FOLDER, filename)
        plt.savefig(save_path, dpi=100, facecolor='#0b0f19', bbox_inches='tight')
        plt.close()

        spectrum_url = f"/static/fft/{filename}"

        # ── Human-readable analysis ─────────────────────────────────
        if high_freq_ratio < 0.15:
            freq_assessment = "Low high-frequency content — may indicate AI smoothing"
        elif high_freq_ratio > 0.35:
            freq_assessment = "Rich high-frequency detail — consistent with real photos"
        else:
            freq_assessment = "Moderate frequency distribution — inconclusive"

        if noise_variance < 50:
            noise_assessment = "Very low noise — possibly AI-generated (too clean)"
        elif noise_variance > 500:
            noise_assessment = "High noise variance — consistent with real camera sensor"
        else:
            noise_assessment = "Normal noise levels"

        analysis_summary = f"{freq_assessment}. {noise_assessment}."

        result = {
            "high_freq_ratio": round(high_freq_ratio, 4),
            "spectral_centroid": round(spectral_centroid, 2),
            "noise_variance": round(noise_variance, 2),
            "spectrum_url": spectrum_url,
            "analysis_summary": analysis_summary,
        }

        logger.info(f"FFT analysis complete: HF={high_freq_ratio:.4f}, "
                     f"SC={spectral_centroid:.2f}, NV={noise_variance:.2f}")
        return result

    except Exception as e:
        logger.error(f"FFT analysis failed: {e}")
        return {"error": str(e)}
