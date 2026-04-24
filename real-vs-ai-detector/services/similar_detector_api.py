"""
Wrapper API for Got-Chu Similar Object Detection Module.
Provides lazy loading, timeout protection, and ROI validation
without polluting existing dependencies.
"""
import os
import sys
import time
import base64
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import cv2
import numpy as np

logger = logging.getLogger("services.similar_detector")

# Absolute path to the cloned module
MODULE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../modules/similar_object_detection'))

# Lazy loaded module reference
_detector_module = None

def load_model():
    """Lazily load the external module by injecting its path."""
    global _detector_module
    if _detector_module is None:
        try:
            logger.info("Initializing Similar Object Detection module...")
            if MODULE_DIR not in sys.path:
                sys.path.insert(0, MODULE_DIR)
            
            import detector
            _detector_module = detector
            logger.info("Similar Object Detection module loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load similar detector: {e}")
            raise
    return _detector_module

def _validate_roi(image, x, y, w, h):
    """
    Validate that the ROI is not completely empty, blank, or featureless.
    Returns True if valid, False otherwise.
    """
    if w < 10 or h < 10:
        return False
        
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return False
        
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)
    
    # Check if the region is too uniform (e.g., solid color background)
    if variance < 10.0:
        logger.warning(f"ROI rejected: variance too low ({variance:.2f})")
        return False
        
    return True

def _run_detection_task(image_path, box):
    """Inner function to run detection within a thread."""
    det = load_model()
    # The detector's find_similar returns: img_disp, count, final_boxes, image_hash, sample_hash
    # We disable cache to ensure we test exactly what's provided
    img_disp, count, final_boxes, _, _ = det.find_similar(image_path, box, use_cache=False)
    return img_disp, count, final_boxes

def detect_similar_objects(image_path: str, roi_bbox: list):
    """
    Detect similar objects in the image based on the ROI.
    
    Args:
        image_path: Path to the full image.
        roi_bbox: List/tuple of [x1, y1, x2, y2]
        
    Returns:
        dict: {
            "boxes": [[x1, y1, x2, y2], ...],
            "count": int,
            "annotated_image_b64": str (base64 encoded jpeg),
            "error": str (if any)
        }
    """
    try:
        if not os.path.exists(image_path):
            return {"error": "Image not found"}
            
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image"}
            
        x1, y1, x2, y2 = [int(v) for v in roi_bbox]
        
        # Ensure coordinates are within image bounds
        h_img, w_img = img.shape[:2]
        x1, x2 = max(0, min(x1, x2)), min(w_img, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h_img, max(y1, y2))
        
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        
        if not _validate_roi(img, x, y, w, h):
            return {"error": "Invalid ROI: The selected region is too small or lacks features (e.g., blank background)."}
        
        # Execute with timeout protection to prevent hanging the main Flask worker
        # 10 seconds timeout for SIFT/Template execution
        timeout_seconds = 10.0
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_detection_task, image_path, (x, y, w, h))
            try:
                img_disp, count, final_boxes = future.result(timeout=timeout_seconds)
            except TimeoutError:
                logger.error(f"Detection timed out after {timeout_seconds}s")
                return {"error": f"Detection timed out (took >{timeout_seconds}s). The object might be too complex."}
        
        # If the detector returns an image, encode it to base64
        b64_img = None
        if img_disp is not None:
            # We add our own Red bounding box for the ROI just to be clear
            cv2.rectangle(img_disp, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img_disp, "ROI", (x, max(10, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            _, buffer = cv2.imencode('.jpg', img_disp, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            b64_img = base64.b64encode(buffer).decode('utf-8')
            b64_img = f"data:image/jpeg;base64,{b64_img}"
            
        return {
            "boxes": final_boxes,
            "count": count,
            "annotated_image_b64": b64_img,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Error in similar object detection: {e}", exc_info=True)
        return {"error": str(e)}
