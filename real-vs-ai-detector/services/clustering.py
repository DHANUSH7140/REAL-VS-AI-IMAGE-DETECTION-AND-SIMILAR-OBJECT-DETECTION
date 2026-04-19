"""
services/clustering.py — DBSCAN-based clustering of object embeddings.

Groups detected objects into semantic clusters based on their
embedding similarity. Identifies which cluster the user's ROI belongs to.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from utils.logger import setup_logger

logger = setup_logger("services.clustering")

# ── Cluster color palette (BGR for OpenCV) ─────────────────────
CLUSTER_COLORS_BGR = [
    (255, 165, 0),    # Cyan-blue
    (0, 230, 118),    # Green
    (0, 140, 255),    # Orange
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (255, 100, 100),  # Light blue
    (100, 255, 100),  # Light green
    (147, 20, 255),   # Pink
    (255, 255, 0),    # Cyan
    (0, 165, 255),    # Gold
]

# CSS-friendly hex colors for the UI legend
CLUSTER_COLORS_HEX = [
    "#00a5ff", "#22c55e", "#ff8c00", "#ff00ff", "#ffff00",
    "#6464ff", "#64ff64", "#ff1493", "#00ffff", "#ffa500",
]

NOISE_COLOR_BGR = (120, 120, 120)  # Gray for noise/unclustered
NOISE_COLOR_HEX = "#787878"


def cluster_embeddings(
    embeddings: np.ndarray,
    roi_embedding: np.ndarray = None,
    eps: float = 0.35,
    min_samples: int = 2,
) -> dict:
    """
    Cluster object embeddings using DBSCAN with cosine distance.

    Args:
        embeddings:     (N, D) array of L2-normalized embeddings.
        roi_embedding:  1-D ROI embedding to identify its cluster.
        eps:            DBSCAN epsilon (max cosine distance within cluster).
        min_samples:    Minimum points per cluster.

    Returns:
        Dict with:
            - labels:           Array of cluster IDs per object (-1 = noise)
            - n_clusters:       Number of clusters found
            - roi_cluster_id:   Cluster ID the ROI belongs to (-1 = noise)
            - cluster_summary:  Dict mapping cluster_id -> count
            - cluster_members:  Dict mapping cluster_id -> list of indices
            - cluster_colors:   Dict mapping cluster_id -> hex color
    """
    if embeddings.size == 0 or len(embeddings) == 0:
        return _empty_result()

    # Compute cosine distance matrix
    dist_matrix = cosine_distances(embeddings)

    # Run DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = dbscan.fit_predict(dist_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Build cluster summary
    cluster_summary = {}
    cluster_members = {}
    for i, label in enumerate(labels):
        lid = int(label)
        cluster_summary[lid] = cluster_summary.get(lid, 0) + 1
        if lid not in cluster_members:
            cluster_members[lid] = []
        cluster_members[lid].append(i)

    # Assign colors
    cluster_colors = {}
    unique_labels = sorted(set(labels))
    color_idx = 0
    for lid in unique_labels:
        if lid == -1:
            cluster_colors[-1] = NOISE_COLOR_HEX
        else:
            cluster_colors[int(lid)] = CLUSTER_COLORS_HEX[
                color_idx % len(CLUSTER_COLORS_HEX)
            ]
            color_idx += 1

    # Determine ROI cluster
    roi_cluster_id = -1
    if roi_embedding is not None and len(embeddings) > 0:
        # Find which cluster the ROI is closest to
        roi_distances = cosine_distances(
            roi_embedding.reshape(1, -1), embeddings
        )[0]
        nearest_idx = int(np.argmin(roi_distances))
        roi_cluster_id = int(labels[nearest_idx])

    logger.info(
        f"DBSCAN clustering: {n_clusters} clusters from {len(embeddings)} objects "
        f"(eps={eps}, min_samples={min_samples}), ROI cluster={roi_cluster_id}"
    )

    return {
        "labels": labels.tolist(),
        "n_clusters": n_clusters,
        "roi_cluster_id": roi_cluster_id,
        "cluster_summary": {str(k): v for k, v in sorted(cluster_summary.items())},
        "cluster_members": {str(k): v for k, v in sorted(cluster_members.items())},
        "cluster_colors": cluster_colors,
    }


def get_roi_cluster_members(clustering_result: dict) -> list:
    """
    Get indices of objects in the same cluster as the ROI.

    Args:
        clustering_result: Output from cluster_embeddings().

    Returns:
        List of detection indices belonging to the ROI's cluster.
    """
    roi_cid = clustering_result.get("roi_cluster_id", -1)
    if roi_cid == -1:
        return []

    members = clustering_result.get("cluster_members", {})
    return members.get(str(roi_cid), [])


def get_cluster_color_bgr(cluster_id: int) -> tuple:
    """Get BGR color for a cluster ID (for OpenCV drawing)."""
    if cluster_id == -1:
        return NOISE_COLOR_BGR
    return CLUSTER_COLORS_BGR[cluster_id % len(CLUSTER_COLORS_BGR)]


def _empty_result():
    return {
        "labels": [],
        "n_clusters": 0,
        "roi_cluster_id": -1,
        "cluster_summary": {},
        "cluster_members": {},
        "cluster_colors": {},
    }
