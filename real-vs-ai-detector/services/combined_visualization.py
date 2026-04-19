"""
services/combined_visualization.py — YOLO + Classifier + Grad-CAM combined pipeline.

Pipeline:
    1. Run YOLOv8 to detect objects in the image
    2. Crop each detected region
    3. Run AI vs Real classifier on each crop
    4. Generate Grad-CAM heatmap for each crop
    5. Overlay heatmaps inside bounding boxes on the original image
    6. Draw bounding boxes with classification labels + confidence
    7. Return final annotated image + per-region predictions
"""

import os
import uuid
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

from config import MODEL_CONFIGS, UPLOAD_FOLDER
from models.loader import model_manager
from services.yolo_detector import detect_objects
from utils.preprocessing import preprocess_image as preprocess_for_model
from utils.logger import setup_logger

logger = setup_logger("services.combined_viz")

# ──────────────────────────── COLORS ────────────────────────────
COLOR_REAL = (46, 204, 113)     # Green (BGR)
COLOR_AI   = (71, 71, 239)     # Red (BGR)
COLOR_TEXT_BG = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _find_last_conv_layer(model) -> str:
    """Auto-detect the last Conv2D layer in a Keras model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
    raise ValueError("No Conv2D layer found.")


def _get_gradcam_heatmap(model, img_array, layer_name: str) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a preprocessed image.

    Args:
        model:      Loaded Keras model.
        img_array:  Preprocessed image array (1, H, W, 3).
        layer_name: Target conv layer name.

    Returns:
        Heatmap as numpy array (H, W) with values in [0, 1].
    """
    try:
        # Find the layer in the model or a sub-model
        try:
            target_layer = model.get_layer(layer_name)
        except ValueError:
            target_layer = None
            for layer in model.layers:
                if hasattr(layer, 'get_layer'):
                    try:
                        target_layer = layer.get_layer(layer_name)
                        break
                    except ValueError:
                        continue
            if target_layer is None:
                raise ValueError(f"Layer '{layer_name}' not found.")

        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[target_layer.output, model.output]
        )

        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

        return heatmap.numpy()

    except Exception as e:
        logger.warning(f"Grad-CAM heatmap generation failed: {e}")
        return None


def _classify_crop(crop_bgr: np.ndarray, model, model_name: str) -> dict:
    """
    Classify a cropped image region as Real or AI.

    Args:
        crop_bgr:   Cropped region as BGR numpy array.
        model:      Loaded Keras model.
        model_name: Model identifier for preprocessing config.

    Returns:
        Dict with label, confidence, raw_score.
    """
    cfg = MODEL_CONFIGS.get(model_name, {})
    img_size = cfg.get("img_size", (224, 224))

    # Convert BGR to RGB and resize
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb).resize(img_size)
    arr = np.array(pil_img, dtype="float32")

    # Apply preprocessing
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

    if raw_score >= 0.5:
        label = "AI Generated"
        confidence = round(raw_score * 100, 2)
    else:
        label = "Real Image"
        confidence = round((1 - raw_score) * 100, 2)

    return {
        "label": label,
        "confidence": confidence,
        "raw_score": raw_score,
        "img_array": img_array,  # Keep for Grad-CAM
    }


def run_combined_pipeline(
    image_path: str,
    classifier_name: str = "efficientnet",
    yolo_conf: float = 0.35,
) -> dict:
    """
    Run the full YOLO + Classifier + Grad-CAM combined pipeline.

    Args:
        image_path:      Path to the input image.
        classifier_name: Which classifier to use ('cnn', 'resnet', 'efficientnet').
        yolo_conf:       YOLO detection confidence threshold.

    Returns:
        Dict with:
            - annotated_image_url:  URL to the final annotated image
            - regions:              List of per-region prediction results
            - total_objects:        Number of objects detected
            - summary:              Overall assessment text
    """
    logger.info(f"Starting combined pipeline: image={image_path}, model={classifier_name}")

    # ── 1. Load original image ──────────────────────────────────
    original = cv2.imread(image_path)
    if original is None:
        return {"error": "Could not load image."}
    annotated = original.copy()
    h_img, w_img = original.shape[:2]

    # ── 2. Run YOLOv8 detection ─────────────────────────────────
    detections = detect_objects(image_path, conf_threshold=yolo_conf)

    if not detections:
        logger.info("No objects detected by YOLO.")
        # Still classify the full image
        detections = [{
            "bbox": [0, 0, w_img, h_img],
            "class_name": "full_image",
            "class_id": -1,
            "confidence": 1.0,
        }]

    # ── 3. Load classifier model ────────────────────────────────
    try:
        classifier = model_manager.get(classifier_name)
    except Exception as e:
        return {"error": f"Failed to load classifier: {e}"}

    # Determine Grad-CAM layer
    cfg = MODEL_CONFIGS.get(classifier_name, {})
    gradcam_layer = cfg.get("gradcam_layer")
    if gradcam_layer is None:
        try:
            gradcam_layer = _find_last_conv_layer(classifier)
        except ValueError:
            gradcam_layer = None

    # ── 4. Process each detected region ─────────────────────────
    regions = []
    ai_count = 0
    real_count = 0

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        # Ensure valid crop bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)

        crop_w = x2 - x1
        crop_h = y2 - y1

        # Skip tiny regions
        if crop_w < 20 or crop_h < 20:
            continue

        # Crop the region
        crop = original[y1:y2, x1:x2]

        # ── Classify the crop ───────────────────────────────────
        result = _classify_crop(crop, classifier, classifier_name)
        is_ai = result["label"] == "AI Generated"

        if is_ai:
            ai_count += 1
        else:
            real_count += 1

        # ── Generate Grad-CAM for the crop ──────────────────────
        heatmap = None
        if gradcam_layer is not None:
            heatmap = _get_gradcam_heatmap(classifier, result["img_array"], gradcam_layer)

        # ── Overlay heatmap inside bounding box ─────────────────
        if heatmap is not None:
            heatmap_resized = cv2.resize(heatmap, (crop_w, crop_h))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # Blend heatmap with the region in the annotated image
            region = annotated[y1:y2, x1:x2]
            blended = cv2.addWeighted(region, 0.6, heatmap_colored, 0.4, 0)
            annotated[y1:y2, x1:x2] = blended

        # ── Draw bounding box ───────────────────────────────────
        color = COLOR_AI if is_ai else COLOR_REAL
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # ── Draw label background ───────────────────────────────
        label_text = f"{det['class_name']}: {result['label']} ({result['confidence']}%)"
        (tw, th), baseline = cv2.getTextSize(label_text, FONT, 0.5, 1)

        label_y = max(y1 - 8, th + 8)
        cv2.rectangle(annotated,
                       (x1, label_y - th - 6),
                       (x1 + tw + 8, label_y + 4),
                       color, -1)
        cv2.putText(annotated, label_text,
                     (x1 + 4, label_y - 2),
                     FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Store region result (without the numpy array)
        regions.append({
            "bbox": det["bbox"],
            "object_class": det["class_name"],
            "yolo_confidence": det["confidence"],
            "classification": result["label"],
            "classification_confidence": result["confidence"],
            "has_gradcam": heatmap is not None,
        })

    # ── 5. Add global summary overlay ───────────────────────────
    summary_text = f"Objects: {len(regions)} | Real: {real_count} | AI: {ai_count}"
    cv2.rectangle(annotated, (0, 0), (len(summary_text) * 11 + 20, 36), (0, 0, 0), -1)
    cv2.putText(annotated, summary_text, (10, 25),
                FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # ── 6. Save annotated image ─────────────────────────────────
    filename = f"combined_{uuid.uuid4().hex[:12]}.png"
    save_dir = os.path.join(os.path.dirname(UPLOAD_FOLDER), "combined")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, annotated)

    annotated_url = f"/static/combined/{filename}"
    logger.info(f"Combined visualization saved: {annotated_url}")

    # ── 7. Generate summary assessment ──────────────────────────
    if ai_count == 0 and real_count > 0:
        summary = "All detected regions appear to be from a real photograph."
    elif real_count == 0 and ai_count > 0:
        summary = "All detected regions appear to be AI-generated."
    elif ai_count > real_count:
        summary = f"Majority of regions ({ai_count}/{len(regions)}) classified as AI-generated. Image is likely AI-generated."
    elif real_count > ai_count:
        summary = f"Majority of regions ({real_count}/{len(regions)}) classified as real. Image is likely authentic."
    else:
        summary = "Mixed results — some regions appear real, others AI-generated."

    return {
        "annotated_image_url": annotated_url,
        "regions": regions,
        "total_objects": len(regions),
        "real_count": real_count,
        "ai_count": ai_count,
        "summary": summary,
        "classifier_used": classifier_name,
    }
