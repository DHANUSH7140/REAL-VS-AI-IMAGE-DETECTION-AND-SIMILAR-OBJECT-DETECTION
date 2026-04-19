"""
services/faiss_engine.py — FAISS-accelerated similarity search.

Uses FAISS IndexFlatIP (inner product on L2-normalized vectors = cosine)
for fast nearest-neighbor search. Falls back to numpy for small sets.
"""

import numpy as np

from utils.logger import setup_logger

logger = setup_logger("services.faiss_engine")

_FAISS_AVAILABLE = True

try:
    import faiss
except ImportError:
    _FAISS_AVAILABLE = False
    logger.info("faiss-cpu not installed — using numpy fallback for similarity.")


def is_available() -> bool:
    """Check if FAISS is available."""
    return _FAISS_AVAILABLE


def build_index(embeddings: np.ndarray) -> "faiss.Index":
    """
    Build a FAISS inner-product index from L2-normalized embeddings.

    Args:
        embeddings: (N, D) float32 array of L2-normalized vectors.

    Returns:
        FAISS IndexFlatIP instance.
    """
    if not _FAISS_AVAILABLE:
        raise RuntimeError("FAISS not installed.")

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.debug(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def search_similar(
    index: "faiss.Index",
    query: np.ndarray,
    k: int = 50,
    threshold: float = 0.7,
) -> list:
    """
    Search for the top-k most similar vectors using FAISS.

    Args:
        index:     FAISS index built from candidate embeddings.
        query:     1-D query vector (L2-normalized).
        k:         Number of neighbors to search.
        threshold: Minimum similarity score to include.

    Returns:
        List of (index, similarity) tuples, sorted by similarity descending.
    """
    if not _FAISS_AVAILABLE:
        raise RuntimeError("FAISS not installed.")

    query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
    k = min(k, index.ntotal)

    if k == 0:
        return []

    distances, indices = index.search(query, k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx >= 0 and float(score) >= threshold:
            results.append((int(idx), float(score)))

    # Already sorted by FAISS (descending similarity for IP)
    logger.debug(f"FAISS search: {len(results)} matches above {threshold}")
    return results


def numpy_cosine_search(
    query: np.ndarray,
    candidates: np.ndarray,
    threshold: float = 0.7,
) -> list:
    """
    Fallback numpy-based cosine search when FAISS unavailable.

    Args:
        query:      1-D query vector (L2-normalized).
        candidates: (N, D) array (L2-normalized).
        threshold:  Minimum similarity score.

    Returns:
        List of (index, similarity) tuples, sorted descending.
    """
    if candidates.size == 0:
        return []

    scores = candidates @ query  # dot product = cosine for L2-normed
    results = []
    for i, score in enumerate(scores):
        if float(score) >= threshold:
            results.append((i, float(score)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def search(
    query: np.ndarray,
    candidates: np.ndarray,
    threshold: float = 0.7,
    use_faiss: bool = True,
) -> list:
    """
    Unified search interface — uses FAISS when available and beneficial.

    Args:
        query:      1-D query vector (L2-normalized).
        candidates: (N, D) array (L2-normalized).
        threshold:  Minimum similarity score.
        use_faiss:  Whether to prefer FAISS (auto-fallback to numpy).

    Returns:
        List of (index, similarity) tuples.
    """
    n = candidates.shape[0] if candidates.ndim == 2 else 0

    # Use FAISS for larger sets (overhead is justified at >10 objects)
    if use_faiss and _FAISS_AVAILABLE and n > 10:
        index = build_index(candidates)
        return search_similar(index, query, k=n, threshold=threshold)

    return numpy_cosine_search(query, candidates, threshold)
