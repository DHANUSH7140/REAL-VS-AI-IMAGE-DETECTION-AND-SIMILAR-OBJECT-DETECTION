"""
Inference module for AI vs Real image detection.

Loads trained model artifacts and runs the full prediction pipeline on a single image.
Can be used as a library or as a standalone CLI:
    python predict.py --image path/to/image.jpg
"""

import argparse
import logging
import os
import sys
import pickle
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("visionprobe.detector.predict")

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")


class AIDetector:
    """Encapsulates the full detection pipeline: feature extraction → XGBoost → calibration."""

    def __init__(self, weights_dir: str = None, device: str = "auto"):
        self.weights_dir = weights_dir or WEIGHTS_DIR
        self.device = None
        self.extractor = None
        self.model = None
        self.calibrated_model = None
        self._loaded = False
        self._device_str = device

    def load(self):
        """Load all model artifacts."""
        import torch

        if self._device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self._device_str)

        logger.info(f"Loading detector on {self.device}")

        # 1. Load OpenCLIP
        clip_ext = None
        try:
            import open_clip
            from detector.feature_extractors import CLIPFeatureExtractor
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
            )
            model = model.to(self.device).eval()
            tokenizer = open_clip.get_tokenizer("ViT-L-14")
            clip_ext = CLIPFeatureExtractor(model, preprocess, tokenizer, self.device)
            logger.info("  OpenCLIP ViT-L/14 loaded")
        except Exception as e:
            logger.warning(f"  OpenCLIP failed: {e}")

        # 2. Load EfficientNet
        effnet_ext = None
        try:
            import timm
            from detector.feature_extractors import EfficientNetFeatureExtractor
            backbone = timm.create_model(
                "tf_efficientnetv2_l.in21k_ft_in1k",
                pretrained=True, num_classes=0
            ).to(self.device).eval()
            effnet_ext = EfficientNetFeatureExtractor(backbone, self.device)
            logger.info("  EfficientNetV2-L loaded")
        except Exception as e:
            logger.warning(f"  EfficientNet failed: {e}")

        # 3. Build extractor
        from detector.feature_extractors import MultiFeatureExtractor
        self.extractor = MultiFeatureExtractor(
            clip_extractor=clip_ext,
            effnet_extractor=effnet_ext,
        )

        # 4. Load feature processor (scalers + PCA)
        proc_path = os.path.join(self.weights_dir, "feature_processor.pkl")
        if os.path.exists(proc_path):
            self.extractor.load(proc_path)
            logger.info("  Feature processor loaded")
        else:
            logger.warning("  No feature processor found — running without normalization/PCA")

        # 5. Load calibrated model
        cal_path = os.path.join(self.weights_dir, "calibrated_model.pkl")
        if os.path.exists(cal_path):
            with open(cal_path, "rb") as f:
                self.calibrated_model = pickle.load(f)
            logger.info("  Calibrated XGBoost model loaded")
        else:
            # Try raw XGBoost
            model_path = os.path.join(self.weights_dir, "xgboost_model.json")
            if os.path.exists(model_path):
                import xgboost as xgb
                self.model = xgb.XGBClassifier()
                self.model.load_model(model_path)
                logger.info("  Raw XGBoost model loaded (no calibration)")
            else:
                logger.warning("  No trained model found! Run train.py first.")

        self._loaded = True
        logger.info("Detector ready.")

    def predict(self, pil_img: Image.Image) -> dict:
        """
        Run full pipeline on a single image.
        
        Returns:
            {
                "label": "REAL" | "AI",
                "confidence": float (0-100%),
                "raw_probability": float (0-1, probability of AI),
                "features": dict of per-branch raw features,
            }
        """
        if not self._loaded:
            self.load()

        pil_img = pil_img.convert("RGB")

        # Standardize resolution to match training exactly (512x512)
        pil_img = pil_img.resize((512, 512), Image.LANCZOS)

        # Extract features
        t0 = time.time()
        feature_vec = self.extractor.extract(pil_img)
        extract_time = time.time() - t0

        # Classify
        t1 = time.time()
        if self.calibrated_model is not None:
            prob = float(self.calibrated_model.predict_proba(feature_vec.reshape(1, -1))[0][1])
        elif self.model is not None:
            prob = float(self.model.predict_proba(feature_vec.reshape(1, -1))[0][1])
        else:
            prob = self._heuristic_prediction(pil_img)
        classify_time = time.time() - t1

        # Rely on the calibrated classifier's genuine probability
        threshold = 0.50
        label = "AI" if prob >= threshold else "REAL"
        
        if label == "AI":
            confidence = prob * 100.0
        else:
            confidence = (1.0 - prob) * 100.0

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "raw_probability": round(prob, 4),
            "timings": {
                "extraction_ms": int(extract_time * 1000),
                "classification_ms": int(classify_time * 1000),
            },
        }

    def predict_with_features(self, pil_img: Image.Image) -> dict:
        """Predict and also return raw per-branch features for explainability."""
        if not self._loaded:
            self.load()

        pil_img = pil_img.convert("RGB")
        # Standardize resolution to match training exactly (512x512)
        pil_img = pil_img.resize((512, 512), Image.LANCZOS)

        # Extract raw blocks
        blocks = self.extractor.extract_raw(pil_img)

        # Process and classify
        feature_vec = self.extractor._process_blocks(blocks)

        if self.calibrated_model is not None:
            prob = float(self.calibrated_model.predict_proba(feature_vec.reshape(1, -1))[0][1])
        elif self.model is not None:
            prob = float(self.model.predict_proba(feature_vec.reshape(1, -1))[0][1])
        else:
            prob = self._heuristic_prediction(pil_img)

        # Rely on the calibrated classifier's genuine probability
        threshold = 0.50
        label = "AI" if prob >= threshold else "REAL"
        
        if label == "AI":
            confidence = prob * 100.0
        else:
            confidence = (1.0 - prob) * 100.0

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "raw_probability": round(prob, 4),
            "feature_blocks": {k: v.tolist() for k, v in blocks.items()},
            "feature_vector": feature_vec.tolist(),
        }

    def _heuristic_prediction(self, pil_img: Image.Image) -> float:
        """Fallback heuristic when no trained model is available.
        Uses FFT + SRM scores with simple thresholds."""
        from detector.feature_extractors import FFTFeatureExtractor, SRMFeatureExtractor

        fft_feat = FFTFeatureExtractor.extract(pil_img)
        srm_feat = SRMFeatureExtractor.extract(pil_img)

        # FFT heuristic: mid-frequency energy and spectral flatness
        mid_freq = fft_feat[3]  # mid_freq_energy
        flatness = fft_feat[4]  # spectral_flatness

        fft_score = 0.5
        if mid_freq > 0.8:
            fft_score = 0.7
        elif mid_freq > 0.6:
            fft_score = 0.6
        if flatness < 0.35 or flatness > 0.65:
            fft_score += 0.1

        # SRM heuristic: kurtosis
        avg_kurtosis = abs(srm_feat[4])  # avg kurtosis
        srm_score = 0.5
        if avg_kurtosis > 6.0:
            srm_score = 0.25  # likely real
        elif avg_kurtosis > 4.0:
            srm_score = 0.35
        elif avg_kurtosis < 2.0:
            srm_score = 0.7   # likely AI

        # CLIP zero-shot if available
        clip_score = 0.5
        if self.extractor and self.extractor.clip_ext:
            sims = self.extractor.clip_ext.get_similarities(pil_img)
            ratio = sims.get("ratio", 0.5)
            clip_score = float(np.clip((ratio - 0.475) * 40.0, 0.05, 0.95))

        return float(np.clip(0.4 * clip_score + 0.35 * fft_score + 0.25 * srm_score, 0.0, 1.0))


def main():
    parser = argparse.ArgumentParser(description="Predict AI vs Real for a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--weights-dir", type=str, default=None, help="Model weights directory")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    detector = AIDetector(weights_dir=args.weights_dir, device=args.device)
    detector.load()

    img = Image.open(args.image).convert("RGB")
    result = detector.predict(img)

    print(f"\n{'='*50}")
    print(f"  Image:      {args.image}")
    print(f"  Prediction: {result['label']}")
    print(f"  Confidence: {result['confidence']:.1f}%")
    print(f"  AI Prob:    {result['raw_probability']:.4f}")
    print(f"  Time:       {result['timings']['extraction_ms'] + result['timings']['classification_ms']}ms")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
