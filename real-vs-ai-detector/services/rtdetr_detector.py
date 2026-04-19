"""
services/rtdetr_detector.py — RT-DETR object detection service.

Detects objects using Ultralytics RT-DETR (Vision Transformer), returning bounding boxes,
class labels, and confidence scores. Designed for high accuracy similarity ROI bounds.
"""

import threading
from ultralytics import RTDETR
from utils.logger import setup_logger

logger = setup_logger("services.rtdetr_detector")

# ──────────────────────────── RT-DETR MODEL ─────────────────────
_rtdetr_model = None
_rtdetr_lock = threading.Lock()

def _get_rtdetr_model():
    """Lazy-load RT-DETR model (singleton)."""
    global _rtdetr_model
    if _rtdetr_model is None:
        with _rtdetr_lock:
            if _rtdetr_model is None:
                logger.info("Loading RT-DETR model…")
                _rtdetr_model = RTDETR("rtdetr-l.pt")
                logger.info("RT-DETR model loaded successfully.")
    return _rtdetr_model

def detect_objects(image_path: str, conf_threshold: float = 0.35, max_detections: int = 15) -> list:
    """
    Run RT-DETR object detection on an image.

    Args:
        image_path:      Absolute path to the input image.
        conf_threshold:  Minimum confidence threshold for detections.
        max_detections:  Maximum number of detections to return.

    Returns:
        List of dicts, each with:
            - bbox: [x1, y1, x2, y2] pixel coordinates
            - class_name: detected object class (e.g., 'person', 'car')
            - class_id: COCO class ID
            - confidence: detection confidence (0–1)
    """
    model = _get_rtdetr_model()

    logger.info(f"Running RT-DETR detection on: {image_path}")
    results = model(image_path, conf=conf_threshold, verbose=False)

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = result.names[cls_id]

            detections.append({
                "bbox": [int(b) for b in bbox],
                "class_name": cls_name,
                "class_id": cls_id,
                "confidence": round(conf, 4),
            })

    # Sort by confidence, limit to max_detections
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    detections = detections[:max_detections]

    logger.info(f"Detected {len(detections)} objects with RT-DETR.")
    return detections
