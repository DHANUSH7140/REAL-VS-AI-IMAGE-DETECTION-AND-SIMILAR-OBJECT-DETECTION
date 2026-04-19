"""
services/similarity_visualization.py — Annotated image generation for similarity results.

Draws ROI, matched objects, and non-matched objects with distinct styling
on the original image to visualize similarity search results.
"""

import os
import uuid
import numpy as np
import cv2

from config import SIMILARITY_FOLDER
from utils.logger import setup_logger

logger = setup_logger("services.similarity_viz")

# ──────────────────────────── COLORS (BGR) ──────────────────────
COLOR_ROI       = (0, 0, 255)       # Red — user's selected region
COLOR_MATCH     = (0, 230, 118)     # Green — similar objects
COLOR_NO_MATCH  = (120, 120, 120)   # Gray — non-similar objects
COLOR_TEXT_BG   = (0, 0, 0)
FONT            = cv2.FONT_HERSHEY_SIMPLEX


def _draw_dashed_rect(img, pt1, pt2, color, thickness=2, dash_len=12):
    """Draw a dashed rectangle on an image."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Top edge
    _draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness, dash_len)
    # Bottom edge
    _draw_dashed_line(img, (x1, y2), (x2, y2), color, thickness, dash_len)
    # Left edge
    _draw_dashed_line(img, (x1, y1), (x1, y2), color, thickness, dash_len)
    # Right edge
    _draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness, dash_len)


def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_len):
    """Draw a dashed line between two points."""
    x1, y1 = pt1
    x2, y2 = pt2

    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if dist < 1:
        return

    num_dashes = int(dist / dash_len)
    for i in range(0, num_dashes, 2):
        t_start = i / num_dashes
        t_end = min((i + 1) / num_dashes, 1.0)

        start = (int(x1 + (x2 - x1) * t_start), int(y1 + (y2 - y1) * t_start))
        end = (int(x1 + (x2 - x1) * t_end), int(y1 + (y2 - y1) * t_end))

        cv2.line(img, start, end, color, thickness)


def _draw_label(img, text, x, y, bg_color, text_color=(255, 255, 255)):
    """Draw a text label with background at the given position."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, 0.5, 1)
    label_y = max(y - 8, th + 8)

    # Background rectangle
    cv2.rectangle(
        img,
        (x, label_y - th - 6),
        (x + tw + 8, label_y + 4),
        bg_color, -1
    )
    # Text
    cv2.putText(
        img, text,
        (x + 4, label_y - 2),
        FONT, 0.5, text_color, 1, cv2.LINE_AA
    )


def _similarity_to_color(similarity: float) -> tuple:
    """
    Map similarity score to a color gradient (green for high, yellow for low).

    Args:
        similarity: Score in [0, 1].

    Returns:
        BGR color tuple.
    """
    # Interpolate from yellow (0, 255, 255) to bright green (0, 230, 0)
    t = min(max(similarity, 0), 1)
    b = 0
    g = int(200 + t * 55)       # 200 → 255
    r = int(255 * (1 - t))      # 255 → 0
    return (b, g, r)


def generate_similarity_visualization(
    image_path: str,
    roi_box: list,
    detections: list,
    matches: list,
) -> str:
    """
    Generate an annotated image showing similarity search results.

    Args:
        image_path:  Path to the original image.
        roi_box:     User's ROI as [x1, y1, x2, y2].
        detections:  List of all YOLO detections (dicts with 'bbox').
        matches:     List of match dicts from similarity_matcher.

    Returns:
        URL path to the saved annotated image.
    """
    original = cv2.imread(image_path)
    if original is None:
        logger.error(f"Could not load image: {image_path}")
        return None

    annotated = original.copy()

    # Build set of matched detection indices for quick lookup
    matched_indices = {m["index"] for m in matches}

    # ── 1. Draw non-matched detections (dimmed gray) ────────────
    for i, det in enumerate(detections):
        if i in matched_indices:
            continue
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_NO_MATCH, 1)
        _draw_label(annotated, det.get("class_name", "?"), x1, y1, COLOR_NO_MATCH)

    # ── 2. Draw matched detections (colored by similarity) ──────
    for match in matches:
        x1, y1, x2, y2 = match["box"]
        sim = match["similarity"]
        color = _similarity_to_color(sim)

        # Thicker border for matches
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Semi-transparent fill
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

        # Label with similarity percentage
        label = f"{match['class_name']}: {sim * 100:.1f}% match"
        _draw_label(annotated, label, x1, y1, color)

    # ── 3. Draw ROI (cyan dashed border) ────────────────────────
    rx1, ry1, rx2, ry2 = roi_box
    _draw_dashed_rect(annotated, (rx1, ry1), (rx2, ry2), COLOR_ROI, thickness=3)

    # ROI label
    _draw_label(annotated, "YOUR SELECTION (ROI)", rx1, ry1, COLOR_ROI, (0, 0, 0))

    # ── 4. Summary overlay (top-left) ──────────────────────────
    summary = f"ROI Search: {len(matches)} similar objects found"
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (len(summary) * 11 + 20, 36), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated) # Make background translucent
    cv2.putText(
        annotated, summary, (10, 25),
        FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA
    )

    # ── 5. Save annotated image ─────────────────────────────────
    filename = f"similar_{uuid.uuid4().hex[:12]}.png"
    os.makedirs(SIMILARITY_FOLDER, exist_ok=True)
    save_path = os.path.join(SIMILARITY_FOLDER, filename)
    cv2.imwrite(save_path, annotated)

    url = f"/static/similarity/{filename}"
    logger.info(f"Similarity visualization saved: {url}")
    return url


def generate_cluster_visualization(
    image_path: str,
    roi_box: list,
    detections: list,
    matches: list,
    clustering_result: dict = None,
    all_scores: list = None,
) -> str:
    """
    Generate an annotated image with cluster-colored bounding boxes.

    Args:
        image_path:        Path to the original image.
        roi_box:           User's ROI as [x1, y1, x2, y2].
        detections:        All YOLO detections.
        matches:           Matched objects with similarity + cluster_id.
        clustering_result: Output from clustering.cluster_embeddings().
        all_scores:        Similarity scores for all detections.

    Returns:
        URL path to the saved annotated image.
    """
    from services.clustering import get_cluster_color_bgr, NOISE_COLOR_BGR

    original = cv2.imread(image_path)
    if original is None:
        logger.error(f"Could not load image: {image_path}")
        return None

    annotated = original.copy()

    labels = clustering_result.get("labels", []) if clustering_result else []
    roi_cluster = clustering_result.get("roi_cluster_id", -1) if clustering_result else -1
    cluster_summary = clustering_result.get("cluster_summary", {}) if clustering_result else {}
    n_clusters = clustering_result.get("n_clusters", 0) if clustering_result else 0

    matched_indices = {m["index"] for m in matches}

    # ── 1. Draw ALL detections (cluster-colored) ─────────────────
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]

        cid = labels[i] if i < len(labels) else -1
        is_match = i in matched_indices
        color = get_cluster_color_bgr(cid)

        if is_match:
            # Thick border + semi-transparent fill for matches
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

            # Score label
            sim = 0.0
            for m in matches:
                if m["index"] == i:
                    sim = m.get("similarity", 0.0)
                    break

            label_text = f"C{cid} | {sim*100:.0f}%"
            _draw_label(annotated, label_text, x1, y1, color)
        else:
            # Thin gray border for unmatched
            cv2.rectangle(annotated, (x1, y1), (x2, y2), NOISE_COLOR_BGR, 1)

    # ── 2. Draw ROI (cyan dashed, thicker) ───────────────────────
    rx1, ry1, rx2, ry2 = roi_box
    _draw_dashed_rect(annotated, (rx1, ry1), (rx2, ry2), COLOR_ROI, thickness=3)
    roi_label = f"ROI (Cluster {roi_cluster})" if roi_cluster >= 0 else "ROI"
    _draw_label(annotated, roi_label, rx1, ry1, COLOR_ROI, (0, 0, 0))

    # ── 3. Top-left summary overlay ──────────────────────────────
    h_img, w_img = annotated.shape[:2]
    lines = [
        f"Similar Objects Found: {len(matches)}",
        f"Clusters: {n_clusters} | ROI Cluster: {roi_cluster}",
    ]

    # Per-cluster counts
    for cid_str, count in sorted(cluster_summary.items(), key=lambda x: int(x[0])):
        cid_int = int(cid_str)
        if cid_int == -1:
            lines.append(f"  Noise: {count} objects")
        else:
            marker = " <-- ROI" if cid_int == roi_cluster else ""
            lines.append(f"  Cluster {cid_int}: {count} objects{marker}")

    # Draw overlay background
    max_line_width = max(len(l) for l in lines) * 10 + 30
    overlay_h = 30 + len(lines) * 22
    
    # Use overlay for translucency
    overlay = annotated.copy()
    cv2.rectangle(
        overlay, (0, 0),
        (min(max_line_width, w_img), min(overlay_h, h_img)),
        (0, 0, 0), -1,
    )
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated) # Make background translucent
    
    cv2.rectangle(
        annotated, (0, 0),
        (min(max_line_width, w_img), min(overlay_h, h_img)),
        COLOR_ROI, 1,
    )

    for i, line in enumerate(lines):
        y_pos = 22 + i * 22
        cv2.putText(
            annotated, line, (10, y_pos),
            FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # ── 4. Save ──────────────────────────────────────────────────
    filename = f"cluster_{uuid.uuid4().hex[:12]}.png"
    os.makedirs(SIMILARITY_FOLDER, exist_ok=True)
    save_path = os.path.join(SIMILARITY_FOLDER, filename)
    cv2.imwrite(save_path, annotated)

    url = f"/static/similarity/{filename}"
    logger.info(f"Cluster visualization saved: {url}")
    return url

