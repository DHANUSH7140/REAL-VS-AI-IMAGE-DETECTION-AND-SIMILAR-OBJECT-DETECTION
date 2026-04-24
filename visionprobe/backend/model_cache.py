"""
ModelCache — thread-safe singleton that loads models for the detector on startup.
Graceful degradation: if any model fails, the rest continue working.
"""

import os
import sys
import time
import threading
import traceback
import logging
import pickle

import numpy as np
import torch

logger = logging.getLogger("visionprobe.model_cache")

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


class ModelCache:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = torch.device("cpu")
        self.all_loaded = False

        # OpenCLIP
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.clip_available = False

        # EfficientNet backbone (intermediate features, no classification head)
        self.effnet_backbone = None
        self.effnet_available = False

        # XGBoost classifier + calibrator
        self.xgb_model = None
        self.calibrated_model = None
        self.xgb_available = False

        # Feature processor (scalers + PCA)
        self.feature_processor = None
        self.processor_available = False

        # Full detector (lazy init)
        self.detector = None

    async def load_all_models(self):
        """Load all models on startup."""
        env_device = os.environ.get("DEVICE", "auto")
        if env_device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif env_device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(env_device)

        logger.info(f"Loading all models on {self.device}")
        os.makedirs(WEIGHTS_DIR, exist_ok=True)

        await self._load_clip()
        await self._load_efficientnet()
        await self._load_xgboost()
        self._init_detector()

        self.all_loaded = True
        loaded = []
        if self.clip_available: loaded.append("OpenCLIP")
        if self.effnet_available: loaded.append("EfficientNet")
        if self.xgb_available: loaded.append("XGBoost")
        logger.info(f"All models loaded. Available: {', '.join(loaded) or 'none (using heuristics)'}")

    async def _load_clip(self):
        """Load OpenCLIP ViT-L/14."""
        try:
            logger.info("Loading OpenCLIP ViT-L/14...")
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
            )
            model = model.to(self.device).eval()
            tokenizer = open_clip.get_tokenizer("ViT-L-14")

            # Warmup
            with torch.inference_mode():
                dummy = torch.randn(1, 3, 224, 224).to(self.device)
                model.encode_image(dummy)

            self.clip_model = model
            self.clip_preprocess = preprocess
            self.clip_tokenizer = tokenizer
            self.clip_available = True
            logger.info("  OpenCLIP ViT-L/14 loaded successfully.")
        except Exception:
            logger.error(f"  OpenCLIP failed to load:\n{traceback.format_exc()}")
            self.clip_available = False

    async def _load_efficientnet(self):
        """Load EfficientNetV2-L backbone for feature extraction."""
        try:
            logger.info("Loading EfficientNetV2-L backbone...")
            import timm
            self.effnet_backbone = timm.create_model(
                "tf_efficientnetv2_l.in21k_ft_in1k",
                pretrained=True, num_classes=0  # No classification head, returns 1280-dim
            ).to(self.device).eval()

            # Warmup
            with torch.inference_mode():
                dummy = torch.randn(1, 3, 480, 480).to(self.device)
                self.effnet_backbone(dummy)

            self.effnet_available = True
            logger.info("  EfficientNetV2-L loaded successfully.")
        except Exception:
            logger.error(f"  EfficientNet failed to load:\n{traceback.format_exc()}")
            self.effnet_available = False

    async def _load_xgboost(self):
        """Load XGBoost model and calibrator."""
        try:
            logger.info("Loading XGBoost classifier...")

            # Try calibrated model first
            cal_path = os.path.join(WEIGHTS_DIR, "calibrated_model.pkl")
            if os.path.exists(cal_path):
                with open(cal_path, "rb") as f:
                    self.calibrated_model = pickle.load(f)
                self.xgb_available = True
                logger.info("  Calibrated XGBoost model loaded.")
            else:
                # Try raw model
                model_path = os.path.join(WEIGHTS_DIR, "xgboost_model.json")
                if os.path.exists(model_path):
                    import xgboost as xgb
                    self.xgb_model = xgb.XGBClassifier()
                    self.xgb_model.load_model(model_path)
                    self.xgb_available = True
                    logger.info("  Raw XGBoost model loaded (no calibration).")
                else:
                    logger.info("  No XGBoost model found — will use heuristic fallback.")
                    self.xgb_available = False

            # Load feature processor
            proc_path = os.path.join(WEIGHTS_DIR, "feature_processor.pkl")
            if os.path.exists(proc_path):
                with open(proc_path, "rb") as f:
                    self.feature_processor = pickle.load(f)
                self.processor_available = True
                logger.info("  Feature processor (scalers + PCA) loaded.")
            else:
                logger.info("  No feature processor found — features will not be normalized.")

        except Exception:
            logger.error(f"  XGBoost failed to load:\n{traceback.format_exc()}")
            self.xgb_available = False

    def _init_detector(self):
        """Initialize the full AIDetector with loaded models."""
        try:
            from detector.feature_extractors import (
                CLIPFeatureExtractor, EfficientNetFeatureExtractor, MultiFeatureExtractor
            )
            from detector.predict import AIDetector

            # Build feature extractors from cached models
            clip_ext = None
            if self.clip_available:
                clip_ext = CLIPFeatureExtractor(
                    self.clip_model, self.clip_preprocess,
                    self.clip_tokenizer, self.device
                )

            effnet_ext = None
            if self.effnet_available:
                effnet_ext = EfficientNetFeatureExtractor(self.effnet_backbone, self.device)

            # Create detector
            self.detector = AIDetector.__new__(AIDetector)
            self.detector.weights_dir = WEIGHTS_DIR
            self.detector.device = self.device
            self.detector.model = self.xgb_model
            self.detector.calibrated_model = self.calibrated_model
            self.detector._loaded = True

            # Build extractor
            self.detector.extractor = MultiFeatureExtractor(
                clip_extractor=clip_ext,
                effnet_extractor=effnet_ext,
            )

            # Load feature processor if available
            if self.processor_available and self.feature_processor is not None:
                self.detector.extractor.scalers = self.feature_processor.get("scalers", {})
                self.detector.extractor.pca_models = self.feature_processor.get("pca_models", {})
                self.detector.extractor.use_pca = self.feature_processor.get("use_pca", True)
                self.detector.extractor.pca_dims = self.feature_processor.get("pca_dims", {"clip": 128, "effnet": 128})

            logger.info("  Full AIDetector initialized.")

        except Exception:
            logger.error(f"  Detector init failed:\n{traceback.format_exc()}")
            self.detector = None
