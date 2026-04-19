"""
services/patch_analyzer.py — Patch-based AI artifact detection.

Splits an image into a grid of patches, runs each through the
classifier, and produces a heatmap of per-patch AI probability.
Useful for detecting localized AI artifacts in composite images.
"""

import os
import uuid
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import MODEL_CONFIGS, PATCH_FOLDER
from models.loader import model_manager
from utils.logger import setup_logger

logger = setup_logger("services.patch_analyzer")


def _classify_patch(patch_bgr: np.ndarray, model, model_name: str) -> float:
    """
    Classify a single patch and return the raw AI probability.

    Args:
        patch_bgr:  Patch as BGR numpy array.
        model:      Loaded Keras classifier.
        model_name: Model identifier for preprocessing.

    Returns:
        Raw AI probability score (0 = real, 1 = AI).
    """
    cfg = MODEL_CONFIGS.get(model_name, {})
    img_size = cfg.get("img_size", (224, 224))

    crop_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb).resize(img_size)
    arr = np.array(pil_img, dtype="float32")

    preprocess_type = cfg.get("preprocess", "rescale")
    if preprocess_type == "resnet":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        arr = preprocess_input(arr)
    elif preprocess_type == "efficientnet":
        from tensorflow.keras.applications.efficientnet import preprocess_input
        arr = preprocess_input(arr)
    else:
        arr = arr / 255.0

    img_array = np.expand_dims(arr, axis=0)
    raw_score = float(model.predict(img_array, verbose=0)[0][0])
    return raw_score


def analyze_patches(
    image_path: str,
    patch_size: int = 128,
    classifier_name: str = "efficientnet",
    stride: int = None,
) -> dict:
    """
    Split image into patches, classify each, and generate analysis.

    Args:
        image_path:      Path to the input image.
        patch_size:      Size of each square patch in pixels.
        classifier_name: Classifier to use for each patch.
        stride:          Step size between patches (default = patch_size).

    Returns:
        Dict with:
            - patch_scores:    2D list of raw AI probability per patch
            - heatmap_url:     URL to saved heatmap visualization
            - artifact_regions: list of suspicious patch coordinates
            - overall_score:   average AI probability across all patches
            - grid_size:       (rows, cols) of the patch grid
            - patch_count:     total number of patches analyzed
    """
    if stride is None:
        stride = patch_size

    logger.info(f"Patch analysis: size={patch_size}, model={classifier_name}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Could not load image."}

    h, w = img.shape[:2]

    # Load classifier
    try:
        classifier = model_manager.get(classifier_name)
    except Exception as e:
        return {"error": f"Failed to load classifier: {e}"}

    # Generate patches
    rows = max(1, (h - patch_size) // stride + 1)
    cols = max(1, (w - patch_size) // stride + 1)

    patch_scores = []
    artifact_regions = []

    for r in range(rows):
        row_scores = []
        for c in range(cols):
            y1 = r * stride
            x1 = c * stride
            y2 = min(y1 + patch_size, h)
            x2 = min(x1 + patch_size, w)

            patch = img[y1:y2, x1:x2]

            # Skip tiny patches at edges
            if patch.shape[0] < 16 or patch.shape[1] < 16:
                row_scores.append(0.0)
                continue

            score = _classify_patch(patch, classifier, classifier_name)
            row_scores.append(round(score, 4))

            # Flag as suspicious if AI probability > 0.65
            if score > 0.65:
                artifact_regions.append({
                    "patch_row": r,
                    "patch_col": c,
                    "bbox": [x1, y1, x2, y2],
                    "ai_probability": round(score, 4),
                })

        patch_scores.append(row_scores)

    # Compute overall score
    all_scores = [s for row in patch_scores for s in row]
    overall_score = round(np.mean(all_scores), 4) if all_scores else 0.0

    # Generate visualization
    heatmap_url = _generate_patch_heatmap(
        image_path, patch_scores, patch_size, stride
    )

    result = {
        "patch_scores": patch_scores,
        "heatmap_url": heatmap_url,
        "artifact_regions": artifact_regions,
        "overall_score": overall_score,
        "grid_size": [rows, cols],
        "patch_count": len(all_scores),
        "classifier_used": classifier_name,
    }

    # Human-readable summary
    if overall_score > 0.7:
        result["summary"] = f"High AI probability ({overall_score*100:.1f}%). Most patches show AI characteristics."
    elif overall_score > 0.4:
        result["summary"] = f"Mixed signals ({overall_score*100:.1f}%). Some regions may be AI-generated."
    else:
        result["summary"] = f"Low AI probability ({overall_score*100:.1f}%). Image appears mostly authentic."

    if artifact_regions:
        result["summary"] += f" {len(artifact_regions)} suspicious region(s) detected."

    logger.info(
        f"Patch analysis: {rows}x{cols} grid, overall={overall_score:.4f}, "
        f"artifacts={len(artifact_regions)}"
    )

    return result


def _generate_patch_heatmap(
    image_path: str,
    patch_scores: list,
    patch_size: int,
    stride: int,
) -> str:
    """Generate and save a heatmap overlay of patch-level AI probabilities."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create score array
    scores_arr = np.array(patch_scores)
    rows, cols = scores_arr.shape

    # Create heatmap at image resolution
    heatmap_full = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            y1 = r * stride
            x1 = c * stride
            y2 = min(y1 + patch_size, h)
            x2 = min(x1 + patch_size, w)
            heatmap_full[y1:y2, x1:x2] += scores_arr[r, c]
            count_map[y1:y2, x1:x2] += 1.0

    count_map = np.where(count_map > 0, count_map, 1.0)
    heatmap_full = heatmap_full / count_map

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0b0f19')

    # Original
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image', color='white', fontsize=11)
    axes[0].axis('off')

    # Heatmap
    im = axes[1].imshow(heatmap_full, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1].set_title('AI Probability Heatmap', color='white', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(img_rgb)
    axes[2].imshow(heatmap_full, cmap='RdYlGn_r', alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title('Overlay', color='white', fontsize=11)
    axes[2].axis('off')

    plt.tight_layout()

    filename = f"patch_{uuid.uuid4().hex[:12]}.png"
    os.makedirs(PATCH_FOLDER, exist_ok=True)
    save_path = os.path.join(PATCH_FOLDER, filename)
    plt.savefig(save_path, dpi=100, facecolor='#0b0f19', bbox_inches='tight')
    plt.close()

    url = f"/static/patches/{filename}"
    logger.info(f"Patch heatmap saved: {url}")
    return url
