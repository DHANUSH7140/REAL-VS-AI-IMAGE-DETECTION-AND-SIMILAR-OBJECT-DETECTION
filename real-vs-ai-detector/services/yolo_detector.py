"""
services/yolo_detector.py — YOLOv8 object detection service.

Detects objects in images using YOLOv8, returns bounding boxes,
class labels, and confidence scores for each detected region.
"""

import os
import threading
from ultralytics import YOLO

from utils.logger import setup_logger

logger = setup_logger("services.yolo_detector")

# ──────────────────────────── YOLO MODEL ────────────────────────
_yolo_model = None
_yolo_lock = threading.Lock()


def _get_yolo_model():
    """Lazy-load YOLOv8 model (singleton)."""
    global _yolo_model
    if _yolo_model is None:
        with _yolo_lock:
            if _yolo_model is None:
                logger.info("Loading YOLOv8n model…")
                _yolo_model = YOLO("yolov8n.pt")
                logger.info("YOLOv8n model loaded successfully.")
    return _yolo_model


def detect_objects(image_path: str, conf_threshold: float = 0.35, max_detections: int = 15) -> list:
    """
    Run YOLOv8 object detection on an image.

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
    model = _get_yolo_model()

    logger.info(f"Running YOLOv8 detection on: {image_path}")
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

    logger.info(f"Detected {len(detections)} objects.")
    return detections
