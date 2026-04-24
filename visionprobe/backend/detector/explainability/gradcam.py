"""
GradCAM++ heatmap generation on EfficientNetV2-L for visual explanation.
"""

import logging
import io
import base64

import numpy as np
import torch
import cv2
from PIL import Image

from model_cache import ModelCache
from preprocessing import image_to_tensor

logger = logging.getLogger("visionprobe.detector.explainability.gradcam")


def generate_gradcam(pil_img: Image.Image) -> str:
    """
    Generate GradCAM++ heatmap on EfficientNetV2-L's last conv block.
    
    Returns base64-encoded JPEG string, or None if generation fails.
    """
    cache = ModelCache.get_instance()

    if not cache.efficientnet_available:
        logger.info("EfficientNet unavailable — skipping GradCAM.")
        return None

    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        logger.warning("grad-cam package not installed — skipping GradCAM.")
        return None

    try:
        device = cache.device
        backbone = cache.efficientnet_backbone

        # Build a wrapper that includes backbone + head for full forward pass
        class FullModel(torch.nn.Module):
            def __init__(self, bb, head):
                super().__init__()
                self.backbone = bb
                self.head = head

            def forward(self, x):
                features = self.backbone.forward_features(x)
                pooled = self.backbone.global_pool(features)
                if hasattr(self.backbone, 'flatten'):
                    pooled = pooled.flatten(1)
                else:
                    pooled = pooled.view(pooled.size(0), -1)
                return self.head(pooled)

        full_model = FullModel(backbone, cache.efficientnet_head)
        full_model.eval()

        # Target layer: last block of EfficientNet
        # For timm EfficientNetV2, the blocks are in backbone.blocks
        target_layers = None
        if hasattr(backbone, "blocks"):
            target_layers = [backbone.blocks[-1]]
        elif hasattr(backbone, "features"):
            target_layers = [backbone.features[-1]]
        else:
            # Fallback: try to get the last named child
            children = list(backbone.children())
            if children:
                target_layers = [children[-2] if len(children) > 1 else children[-1]]

        if target_layers is None:
            logger.warning("Could not identify target layer for GradCAM.")
            return None

        # Prepare input
        tensor = image_to_tensor(pil_img, size=480).to(device)

        # GradCAM++
        cam = GradCAMPlusPlus(model=full_model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=tensor, targets=None)  # predicted class

        # Prepare original image for overlay
        img_resized = pil_img.resize((480, 480), Image.LANCZOS)
        img_np = np.array(img_resized).astype(np.float32) / 255.0

        # Overlay heatmap
        visualization = show_cam_on_image(
            img_np,
            grayscale_cam[0],
            use_rgb=True,
            colormap=cv2.COLORMAP_JET,
            image_weight=0.55,
        )

        # Encode to base64 JPEG
        result_img = Image.fromarray(visualization)
        buf = io.BytesIO()
        result_img.save(buf, format="JPEG", quality=85)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Clean up cam
        del cam

        return b64_str

    except Exception as e:
        logger.warning(f"GradCAM generation failed: {e}", exc_info=True)
        return None
