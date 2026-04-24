"""
Explainability module for AI vs Real image detection.

Provides:
  1. Visual explanation — Grad-CAM++ heatmap on EfficientNet
  2. Text reasoning — CLIP similarity-based analysis
  3. Feature importance — XGBoost native feature importance
"""

import logging
import io
import base64
import os
import pickle

import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger("visionprobe.detector.explain")


# ===== VISUAL: Grad-CAM++ ==================================================

def generate_gradcam(pil_img: Image.Image, effnet_backbone, device) -> str:
    """
    Generate GradCAM++ heatmap on EfficientNet's last conv block.
    
    Returns base64-encoded JPEG string, or None if generation fails.
    """
    if effnet_backbone is None:
        logger.info("EfficientNet unavailable — skipping GradCAM.")
        return None

    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        logger.warning("grad-cam package not installed — skipping GradCAM.")
        return None

    try:
        import torch
        import torch.nn as nn
        from torchvision import transforms

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        # Wrapper that uses forward_features for proper GradCAM targeting
        class GradCAMWrapper(nn.Module):
            def __init__(self, bb):
                super().__init__()
                self.backbone = bb
                # Add a simple binary head for gradient flow
                self.head = nn.Linear(1280, 2).to(device)

            def forward(self, x):
                features = self.backbone.forward_features(x)
                pooled = self.backbone.global_pool(features)
                if pooled.dim() > 2:
                    pooled = pooled.flatten(1)
                return self.head(pooled)

        wrapper = GradCAMWrapper(effnet_backbone)
        wrapper.eval()

        # Target layer: last block of EfficientNet
        target_layers = None
        if hasattr(effnet_backbone, "blocks"):
            target_layers = [effnet_backbone.blocks[-1]]
        elif hasattr(effnet_backbone, "features"):
            target_layers = [effnet_backbone.features[-1]]
        else:
            children = list(effnet_backbone.children())
            if children:
                target_layers = [children[-2] if len(children) > 1 else children[-1]]

        if target_layers is None:
            logger.warning("Could not identify target layer for GradCAM.")
            return None

        # Prepare input
        transform = transforms.Compose([
            transforms.Resize((480, 480), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        tensor = transform(pil_img).unsqueeze(0).to(device)

        # GradCAM++
        cam = GradCAMPlusPlus(model=wrapper, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=tensor, targets=None)

        # Overlay on original
        img_resized = pil_img.resize((480, 480), Image.LANCZOS)
        img_np = np.array(img_resized).astype(np.float32) / 255.0

        visualization = show_cam_on_image(
            img_np, grayscale_cam[0],
            use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=0.55,
        )

        result_img = Image.fromarray(visualization)
        buf = io.BytesIO()
        result_img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        del cam
        return b64

    except Exception as e:
        logger.warning(f"GradCAM generation failed: {e}", exc_info=True)
        return None


# ===== TEXT REASONING: CLIP Similarity =====================================

def generate_text_reasoning(
    pil_img: Image.Image,
    clip_extractor,
    raw_probability: float,
    fft_features: np.ndarray,
    srm_features: np.ndarray,
) -> list:
    """
    Generate human-readable text reasoning for the prediction.
    
    Combines CLIP similarity analysis with FFT and SRM feature interpretation.
    
    Returns list of reasoning strings.
    """
    reasons = []

    # 1. CLIP-based reasoning
    if clip_extractor is not None:
        try:
            sims = clip_extractor.get_similarities(pil_img)
            ratio = sims.get("ratio", 0.5)
            real_sim = sims.get("real_similarity", 0.5)
            ai_sim = sims.get("ai_similarity", 0.5)

            if ratio > 0.52:
                reasons.append(
                    f"CLIP semantic analysis shows {ratio:.0%} alignment with AI-generated "
                    f"image descriptions vs {1-ratio:.0%} with real photograph descriptions."
                )
            elif ratio < 0.48:
                reasons.append(
                    f"CLIP semantic analysis shows {1-ratio:.0%} alignment with real photograph "
                    f"descriptions — consistent with authentic imagery."
                )

            # Check for specific AI patterns
            raw_sims = sims.get("raw_sims", {})
            for prompt, sim_val in raw_sims.items():
                if "texture" in prompt.lower() and sim_val > 0.28:
                    reasons.append("Texture patterns show potential inconsistencies detected by CLIP.")
                    break
        except Exception as e:
            logger.warning(f"CLIP reasoning failed: {e}")

    # 2. FFT-based reasoning
    if fft_features is not None and len(fft_features) >= 8:
        mid_freq = fft_features[3]   # mid_freq_energy
        flatness = fft_features[4]   # spectral_flatness
        periodicity = fft_features[5]  # periodicity

        if mid_freq > 0.8:
            reasons.append(
                "High-frequency artifacts detected — elevated mid-frequency energy is "
                "characteristic of diffusion model upsampling."
            )
        elif mid_freq > 0.6:
            reasons.append(
                "Moderate mid-frequency energy elevation — possible signs of computational image generation."
            )

        if flatness < 0.35 or flatness > 0.65:
            reasons.append(
                f"Spectral flatness ({flatness:.3f}) outside natural range — "
                "GAN-generated images often show abnormal frequency distribution."
            )

        if periodicity > 0.5:
            reasons.append(
                "Periodic patterns detected in frequency domain — "
                "consistent with AI generation artifacts."
            )

    # 3. SRM-based reasoning
    if srm_features is not None and len(srm_features) >= 15:
        avg_kurtosis = abs(srm_features[4])
        avg_std = srm_features[2]
        avg_entropy = srm_features[10]

        if avg_kurtosis < 2.0:
            reasons.append(
                "Noise pattern inconsistent with real camera — SRM residuals show "
                "low kurtosis (smooth noise), typical of AI-generated images."
            )
        elif avg_kurtosis > 6.0:
            reasons.append(
                "SRM noise residuals show camera sensor patterns (high kurtosis) "
                "consistent with real photography."
            )

        if avg_std < 0.1:
            reasons.append(
                "Very low noise variance in residuals — image appears unusually smooth, "
                "a common trait of AI-generated content."
            )

    # 4. Overall confidence reasoning
    if raw_probability > 0.85:
        reasons.append(
            f"Overall confidence is very high ({raw_probability:.0%}) — "
            "multiple indicators strongly suggest AI generation."
        )
    elif raw_probability > 0.5:
        reasons.append(
            f"Overall analysis leans toward AI-generated ({raw_probability:.0%}) "
            "with moderate confidence."
        )
    elif raw_probability > 0.15:
        reasons.append(
            f"Overall analysis leans toward real photograph ({1-raw_probability:.0%}) "
            "with moderate confidence."
        )
    else:
        reasons.append(
            f"Overall confidence is very high ({1-raw_probability:.0%}) — "
            "image is almost certainly a real photograph."
        )

    # Ensure at least one reason
    if not reasons:
        reasons.append("Analysis complete — see individual feature scores for details.")

    return reasons


# ===== FEATURE IMPORTANCE ==================================================

def get_feature_importance(weights_dir: str = None) -> list:
    """
    Get XGBoost native feature importance.
    
    Returns list of dicts: [{"name": str, "importance": float, "block": str}]
    """
    if weights_dir is None:
        weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")

    model_path = os.path.join(weights_dir, "xgboost_model.json")
    if not os.path.exists(model_path):
        return []

    try:
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(model_path)

        importance = model.feature_importances_
        n_features = len(importance)

        # Map feature indices to names/blocks
        results = []
        for idx, imp in enumerate(importance):
            if imp < 0.001:
                continue

            if idx < 128:
                block = "CLIP"
                name = f"CLIP feature {idx}"
            elif idx < 256:
                block = "EfficientNet"
                name = f"EfficientNet feature {idx - 128}"
            elif idx < 264:
                fft_names = [
                    "Mean magnitude", "Variance", "High-freq energy",
                    "Mid-freq energy", "Spectral flatness", "Periodicity",
                    "High/Low ratio", "Peak prominence"
                ]
                block = "FFT"
                name = fft_names[idx - 256] if (idx - 256) < len(fft_names) else f"FFT feature {idx - 256}"
            else:
                srm_names = [
                    "Avg mean", "Std of means", "Avg std", "Std of stds",
                    "Avg kurtosis", "Std kurtosis", "Max kurtosis", "Min kurtosis",
                    "Avg skewness", "Std skewness", "Avg entropy", "Std entropy",
                    "Median std", "Max std", "Mean abs skewness"
                ]
                block = "SRM"
                si = idx - 264
                name = srm_names[si] if si < len(srm_names) else f"SRM feature {si}"

            results.append({
                "name": name,
                "importance": round(float(imp), 4),
                "block": block,
            })

        results.sort(key=lambda x: x["importance"], reverse=True)
        return results[:15]

    except Exception as e:
        logger.warning(f"Feature importance extraction failed: {e}")
        return []
