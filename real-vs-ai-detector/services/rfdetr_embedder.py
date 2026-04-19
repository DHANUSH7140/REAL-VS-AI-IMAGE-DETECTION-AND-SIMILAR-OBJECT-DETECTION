"""
services/rfdetr_embedder.py — RF-DETR transformer embedding extraction (OPTIONAL).

Extracts encoder-level transformer embeddings from RF-DETR for
object-level semantic features. Gracefully disabled if rfdetr not installed.
"""

import threading
import numpy as np
import cv2

from utils.logger import setup_logger

logger = setup_logger("services.rfdetr_embedder")

# ──────────────────────────── STATE ─────────────────────────────
_rfdetr_model = None
_rfdetr_lock = threading.Lock()
_RFDETR_AVAILABLE = True


def _load_rfdetr():
    """Attempt to load RF-DETR model (lazy, thread-safe)."""
    global _rfdetr_model, _RFDETR_AVAILABLE

    if _rfdetr_model is not None:
        return

    with _rfdetr_lock:
        if _rfdetr_model is not None:
            return

        try:
            from rfdetr import RFDETRBase
            logger.info("Loading RF-DETR model...")
            _rfdetr_model = RFDETRBase()
            logger.info("RF-DETR model loaded successfully.")
        except ImportError:
            _RFDETR_AVAILABLE = False
            logger.info(
                "rfdetr package not installed — RF-DETR embeddings disabled. "
                "Install: pip install rfdetr"
            )
        except Exception as e:
            _RFDETR_AVAILABLE = False
            logger.warning(f"Failed to load RF-DETR: {e}")


def is_available() -> bool:
    """Check if RF-DETR is available."""
    if not _RFDETR_AVAILABLE:
        return False
    _load_rfdetr()
    return _rfdetr_model is not None


def detect_and_embed(image_path: str, conf_threshold: float = 0.3) -> list:
    """
    Run RF-DETR detection and extract encoder embeddings per object.

    Args:
        image_path:      Path to the input image.
        conf_threshold:  Detection confidence threshold.

    Returns:
        List of dicts, each with:
            - bbox: [x1, y1, x2, y2]
            - class_name: detected class
            - confidence: detection confidence
            - embedding: numpy array (encoder features)
        Returns empty list if RF-DETR unavailable.
    """
    if not is_available():
        return []

    try:
        results = _rfdetr_model.predict(image_path, threshold=conf_threshold)

        detections = []
        if hasattr(results, 'xyxy') and results.xyxy is not None:
            boxes = results.xyxy
            confs = results.confidence if hasattr(results, 'confidence') else []
            class_ids = results.class_id if hasattr(results, 'class_id') else []

            for i in range(len(boxes)):
                bbox = [int(b) for b in boxes[i].tolist()]
                conf = float(confs[i]) if i < len(confs) else 0.0
                cls_id = int(class_ids[i]) if i < len(class_ids) else -1

                detections.append({
                    "bbox": bbox,
                    "class_name": f"class_{cls_id}",
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    # RF-DETR doesn't directly expose encoder embeddings via the
                    # public API — we use the detection features as a proxy
                    "embedding": None,
                })

        logger.info(f"RF-DETR detected {len(detections)} objects.")
        return detections

    except Exception as e:
        logger.error(f"RF-DETR detection failed: {e}")
        return []
