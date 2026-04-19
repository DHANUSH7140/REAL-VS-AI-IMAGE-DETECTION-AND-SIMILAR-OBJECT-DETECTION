"""
services/gradcam.py — Grad-CAM heatmap generation for model explainability.

Generates class activation heatmaps showing which image regions
the model focused on when making its prediction.
"""

import os
import uuid
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

from config import MODEL_CONFIGS, GRADCAM_FOLDER
from utils.preprocessing import preprocess_image
from utils.logger import setup_logger

logger = setup_logger("services.gradcam")


def _find_last_conv_layer(model) -> str:
    """
    Auto-detect the last convolutional layer in a Keras model.

    Args:
        model: Loaded Keras model.

    Returns:
        Name of the last Conv2D layer found.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # Also check inside nested models (transfer learning)
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
    raise ValueError("No Conv2D layer found in the model.")


def _get_gradcam_model(model, layer_name: str):
    """
    Build a sub-model that outputs both the conv layer activations and predictions.

    Args:
        model:      Loaded Keras model.
        layer_name: Name of the target convolutional layer.

    Returns:
        tf.keras.Model that outputs [conv_output, predictions].
    """
    # Try to find the layer directly in the model
    try:
        target_layer = model.get_layer(layer_name)
    except ValueError:
        # Layer might be inside a nested model (e.g., resnet50 base)
        target_layer = None
        for layer in model.layers:
            if hasattr(layer, 'get_layer'):
                try:
                    target_layer = layer.get_layer(layer_name)
                    break
                except ValueError:
                    continue

        if target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model or sub-models.")

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output]
    )
    return grad_model


def generate_gradcam(image_path: str, model, model_name: str) -> str:
    """
    Generate a Grad-CAM heatmap overlay for the given image and model.

    Args:
        image_path: Absolute path to the input image.
        model:      Loaded Keras model.
        model_name: Model identifier for config lookup.

    Returns:
        Relative URL path to the saved heatmap overlay image.
    """
    cfg = MODEL_CONFIGS.get(model_name, {})

    # Determine target conv layer
    layer_name = cfg.get("gradcam_layer")
    if layer_name is None:
        try:
            layer_name = _find_last_conv_layer(model)
            logger.info(f"Auto-detected conv layer for '{model_name}': {layer_name}")
        except ValueError as e:
            logger.error(f"Grad-CAM failed for '{model_name}': {e}")
            return None

    try:
        # Build gradient model
        grad_model = _get_gradcam_model(model, layer_name)

        # Preprocess image
        img_array = preprocess_image(image_path, model_name)
        img_tensor = tf.cast(img_array, tf.float32)

        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = grad_model(img_tensor)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the conv output channels by gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Load original image for overlay
        original = cv2.imread(image_path)
        original = cv2.resize(original, (original.shape[1], original.shape[0]))

        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Overlay heatmap on original image
        overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

        # Save overlay
        filename = f"gradcam_{uuid.uuid4().hex[:12]}.png"
        save_path = os.path.join(GRADCAM_FOLDER, filename)
        cv2.imwrite(save_path, overlay)

        gradcam_url = f"/static/gradcam/{filename}"
        logger.info(f"Grad-CAM saved: {gradcam_url}")
        return gradcam_url

    except Exception as e:
        logger.error(f"Grad-CAM generation failed for '{model_name}': {e}")
        return None
