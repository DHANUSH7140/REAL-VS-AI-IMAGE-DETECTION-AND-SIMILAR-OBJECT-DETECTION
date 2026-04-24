#!/usr/bin/env python3
"""
=============================================================================
 REAL vs AI Image Detection — Complete Training Pipeline
=============================================================================

 Multi-branch feature extraction + XGBoost classifier.

 Architecture:
   Branch 1: CLIP ViT-L/14 (semantic, 768-dim, frozen)
   Branch 2: EfficientNetV2-L (artifact, 1280-dim, frozen feature extractor)
   Branch 3: FFT (frequency statistics, 8-dim)
   Branch 4: SRM (noise residuals, 15-dim)
        ↓
   StandardScaler + PCA (per-branch)
        ↓
   Concatenated feature vector (279-dim)
        ↓
   XGBoost binary classifier + Isotonic calibration

 Usage:
   # Auto-download datasets + train
   python train_pipeline.py --auto-download --samples-per-class 1000

   # Use local directories
   python train_pipeline.py --real-dir ./data/real --ai-dir ./data/ai

   # Use existing dataset_v2
   python train_pipeline.py \\
       --real-dir "../real-vs-ai-detector/dataset_v2/real" \\
       --ai-dir "../real-vs-ai-detector/dataset_v2/ai" \\
       --samples-per-class 2000

=============================================================================
"""

import argparse
import hashlib
import io
import json
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance
from scipy.stats import kurtosis, skew, entropy as sp_entropy, gmean
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_pipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
TARGET_SIZE = 512  # Changed from 224 to 512 to preserve high-frequency artifacts for SRM/FFT

SRM_FILTERS = [
    np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=np.float64),
    np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float64),
    np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]], dtype=np.float64),
]
SRM_TRUNCATION = 2.0


# =============================================================================
# SECTION 1: DATASET DOWNLOAD & PREPARATION
# =============================================================================

def collect_images(directory: str, max_count: int = None, shuffle: bool = True) -> list:
    """Recursively collect image paths from a directory."""
    paths = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if Path(f).suffix.lower() in VALID_EXTENSIONS:
                paths.append(os.path.join(root, f))

    if shuffle:
        rng = np.random.RandomState(42)
        rng.shuffle(paths)

    if max_count and len(paths) > max_count:
        paths = paths[:max_count]

    logger.info(f"  Collected {len(paths)} images from {directory}")
    return paths


def download_real_images(output_dir: str, count: int) -> str:
    """Download real images from COCO or ImageNet subsets via HuggingFace."""
    real_dir = os.path.join(output_dir, "real")
    os.makedirs(real_dir, exist_ok=True)

    existing = len(collect_images(real_dir, shuffle=False))
    if existing >= count:
        logger.info(f"  Real images already exist ({existing} found)")
        return real_dir

    needed = count - existing
    downloaded = 0

    # Source 1: COCO validation set (no trust_remote_code)
    try:
        from datasets import load_dataset

        logger.info("  Downloading real images from COCO val set...")
        ds = load_dataset("detection-datasets/coco", split="val", streaming=True, trust_remote_code=True)
        for i, item in enumerate(ds):
            if downloaded >= needed // 2:
                break
            try:
                img = item.get("image")
                if img is not None:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
                    save_path = os.path.join(real_dir, f"coco_{existing + downloaded:05d}.jpg")
                    img.save(save_path, "JPEG", quality=95)
                    downloaded += 1
                    if downloaded % 100 == 0:
                        logger.info(f"    Downloaded {downloaded}/{needed} real images")
            except Exception:
                continue

        logger.info(f"  Downloaded {downloaded} real images from COCO")
    except Exception as e:
        logger.warning(f"  COCO download failed: {e}")

    # Source 2: ImageNet subset (if still need more)
    if downloaded < needed:
        try:
            from datasets import load_dataset

            logger.info("  Downloading real images from ImageNet subset...")
            remaining = needed - downloaded
            ds = load_dataset(
                "mrm8488/ImageNet1K-val", split="train", streaming=True, trust_remote_code=True
            )
            for i, item in enumerate(ds):
                if remaining <= 0:
                    break
                try:
                    img = item.get("image")
                    if img is not None:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
                        save_path = os.path.join(real_dir, f"imagenet_{downloaded:05d}.jpg")
                        img.save(save_path, "JPEG", quality=95)
                        downloaded += 1
                        remaining -= 1
                except Exception:
                    continue

            logger.info(f"  Total real images downloaded: {downloaded}")
        except Exception as e:
            logger.warning(f"  ImageNet download failed: {e}")

    return real_dir


def download_ai_images(output_dir: str, count: int) -> str:
    """Download AI-generated images from DiffusionDB directly via Zip files."""
    import requests
    import zipfile
    import io
    
    ai_dir = os.path.join(output_dir, "ai")
    os.makedirs(ai_dir, exist_ok=True)

    existing = len(collect_images(ai_dir, shuffle=False))
    if existing >= count:
        logger.info(f"  AI images already exist ({existing} found)")
        return ai_dir

    needed = count - existing
    downloaded = 0
    
    # DiffusionDB has parts from 000001 to 002000. Each part has 1000 images.
    part = 1
    
    while downloaded < needed and part <= 2000:
        url = f"https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-{part:06d}.zip"
        logger.info(f"  Downloading DiffusionDB part {part}...")
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    for filename in z.namelist():
                        if downloaded >= needed:
                            break
                        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                            try:
                                with z.open(filename) as f_img:
                                    img = Image.open(f_img).convert("RGB")
                                    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
                                    save_path = os.path.join(ai_dir, f"diffusion_{existing + downloaded:06d}.jpg")
                                    img.save(save_path, "JPEG", quality=95)
                                    downloaded += 1
                                    if downloaded % 100 == 0:
                                        logger.info(f"    Downloaded {downloaded}/{needed} AI images")
                            except Exception as e:
                                continue
            else:
                logger.warning(f"  Failed to download part {part}: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"  Error downloading part {part}: {e}")
        
        part += 1

    logger.info(f"  Total AI images downloaded: {downloaded}")
    return ai_dir


# =============================================================================
# SECTION 2: DATA AUGMENTATION (REAL IMAGES ONLY)
# =============================================================================

class RealImageAugmenter:
    """
    Augmentation for REAL images only.
    AI images are NOT augmented to preserve their generative fingerprints.
    """

    def __init__(self, p_flip=0.5, p_color=0.3, p_noise=0.2):
        self.p_flip = p_flip
        self.p_color = p_color
        self.p_noise = p_noise

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply random augmentations to a real image."""
        # Random horizontal flip
        if np.random.random() < self.p_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Color jitter (brightness, contrast, saturation)
        if np.random.random() < self.p_color:
            factor = np.random.uniform(0.85, 1.15)
            img = ImageEnhance.Brightness(img).enhance(factor)
            factor = np.random.uniform(0.85, 1.15)
            img = ImageEnhance.Contrast(img).enhance(factor)
            factor = np.random.uniform(0.9, 1.1)
            img = ImageEnhance.Color(img).enhance(factor)

        # Gaussian noise
        if np.random.random() < self.p_noise:
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 3.0, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        return img


def load_and_preprocess(
    path: str,
    target_size: int = TARGET_SIZE,
    augmenter: Optional[RealImageAugmenter] = None,
) -> Optional[Image.Image]:
    """Load image, resize to target_size, optionally augment."""
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((target_size, target_size), Image.LANCZOS)
        if augmenter is not None:
            img = augmenter(img)
        return img
    except Exception as e:
        logger.warning(f"  Failed to load {path}: {e}")
        return None


# =============================================================================
# SECTION 3: FEATURE EXTRACTION (ALL 4 BRANCHES)
# =============================================================================

class CLIPExtractor:
    """Branch 1: CLIP ViT-L/14 semantic features (frozen, 768-dim)."""

    def __init__(self, device: torch.device):
        import open_clip

        logger.info("  Loading OpenCLIP ViT-L/14 (frozen)...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
        )
        self.model = self.model.to(device).eval()
        self.device = device
        self.dim = 768

        # Freeze all parameters — no training
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info(f"    CLIP loaded on {device} (768-dim, frozen)")

    @torch.inference_mode()
    def extract(self, pil_img: Image.Image) -> np.ndarray:
        try:
            tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            features = self.model.encode_image(tensor)
            features = F.normalize(features, dim=-1)
            return features.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception:
            return np.zeros(self.dim, dtype=np.float32)


class EfficientNetExtractor:
    """Branch 2: EfficientNetV2-L intermediate features (frozen backbone, 1280-dim)."""

    def __init__(self, device: torch.device):
        import timm

        logger.info("  Loading EfficientNetV2-L backbone (feature extractor)...")
        self.backbone = timm.create_model(
            "tf_efficientnetv2_l.in21k_ft_in1k",
            pretrained=True,
            num_classes=0,  # Remove classification head, returns 1280-dim
        ).to(device).eval()
        self.device = device
        self.dim = 1280

        # Freeze all parameters — use as feature extractor only
        for param in self.backbone.parameters():
            param.requires_grad = False

        logger.info(f"    EfficientNetV2-L loaded on {device} (1280-dim, frozen)")

    @torch.inference_mode()
    def extract(self, pil_img: Image.Image) -> np.ndarray:
        try:
            tensor = self._preprocess(pil_img)
            features = self.backbone(tensor)
            return features.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception:
            return np.zeros(self.dim, dtype=np.float32)

    def _preprocess(self, pil_img: Image.Image) -> torch.Tensor:
        img = pil_img.resize((480, 480), Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)


class FFTExtractor:
    """Branch 3: FFT frequency statistics (8-dim)."""

    dim = 8

    @staticmethod
    def extract(pil_img: Image.Image) -> np.ndarray:
        try:
            gray = np.array(pil_img.convert("L"))
            gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)

            # 2D FFT
            f_transform = np.fft.fft2(gray.astype(np.float64))
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift) + 1e-8
            log_mag = np.log(magnitude)

            # Radial profile (azimuthal average)
            H, W = log_mag.shape
            cy, cx = H // 2, W // 2
            Y, X = np.mgrid[0:H, 0:W]
            R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            max_r = int(min(cx, cy))
            profile = []
            for r in range(0, max_r, 2):
                mask = (R >= r) & (R < r + 2)
                if mask.any():
                    profile.append(float(np.mean(log_mag[mask])))
            profile = np.array(profile) if profile else np.zeros(10)

            # Band decomposition
            n = len(profile)
            low = profile[: max(1, n // 5)]
            mid = profile[n // 5 : 3 * n // 5]
            high = profile[3 * n // 5 :]

            eps = 1e-8
            # Feature 0: mean log magnitude
            mean_mag = float(np.mean(log_mag))
            # Feature 1: variance of log magnitude
            var_mag = float(np.var(log_mag))
            # Feature 2: high-frequency energy
            high_freq_energy = float(np.mean(high)) if len(high) > 0 else 0.0
            # Feature 3: mid-to-low frequency ratio
            mid_freq_ratio = (
                float(np.mean(mid)) / (abs(float(np.mean(low))) + eps)
                if len(mid) > 0
                else 0.0
            )
            # Feature 4: spectral flatness
            profile_pos = np.clip(np.exp(profile), eps, None)
            geo = float(gmean(profile_pos)) if len(profile_pos) > 0 else eps
            arith = float(np.mean(profile_pos))
            spectral_flatness = geo / (arith + eps)
            # Feature 5: periodicity (autocorrelation peak)
            if len(profile) > 4:
                centered = profile - np.mean(profile)
                ac = np.correlate(centered, centered, mode="full")
                ac = ac[len(ac) // 2 :]
                if ac[0] != 0:
                    ac = ac / ac[0]
                search = ac[2 : min(20, len(ac))]
                periodicity = float(np.max(search)) if len(search) > 0 else 0.0
            else:
                periodicity = 0.0
            # Feature 6: high-to-low ratio
            high_low_ratio = (
                float(np.mean(high)) / (abs(float(np.mean(low))) + eps)
                if len(high) > 0 and len(low) > 0
                else 0.0
            )
            # Feature 7: peak prominence
            peak_prom = float(np.max(profile) - np.mean(profile))

            return np.array(
                [mean_mag, var_mag, high_freq_energy, mid_freq_ratio,
                 spectral_flatness, periodicity, high_low_ratio, peak_prom],
                dtype=np.float32,
            )
        except Exception:
            return np.zeros(8, dtype=np.float32)


class SRMExtractor:
    """Branch 4: SRM noise residual statistics (15-dim)."""

    dim = 15

    @staticmethod
    def extract(pil_img: Image.Image) -> np.ndarray:
        try:
            np_img = np.array(pil_img.convert("RGB"))
            np_img = cv2.resize(np_img, (512, 512), interpolation=cv2.INTER_AREA)

            # Apply 3 SRM filters x 3 channels -> (H, W, 9) residuals
            H, W, C = np_img.shape
            residuals = np.zeros((H, W, 9), dtype=np.float64)
            for f_idx, filt in enumerate(SRM_FILTERS):
                for c_idx in range(3):
                    channel = np_img[:, :, c_idx].astype(np.float64)
                    residual = cv2.filter2D(channel, -1, filt)
                    residuals[:, :, f_idx * 3 + c_idx] = residual
            residuals = np.clip(residuals, -SRM_TRUNCATION, SRM_TRUNCATION)

            # Compute statistical features per channel
            means, stds, kurts, skews, entropies = [], [], [], [], []
            for c in range(9):
                ch = residuals[:, :, c].flatten()
                means.append(float(np.mean(ch)))
                stds.append(float(np.std(ch)))
                kurts.append(float(kurtosis(ch, fisher=True)))
                skews.append(float(skew(ch)))
                hist, _ = np.histogram(ch, bins=64, density=True)
                hist = hist + 1e-10
                entropies.append(float(sp_entropy(hist)))

            # Aggregate into 15 summary features
            return np.array(
                [
                    float(np.mean(means)),      # 0: avg mean
                    float(np.std(means)),        # 1: std of means
                    float(np.mean(stds)),        # 2: avg std
                    float(np.std(stds)),         # 3: std of stds
                    float(np.mean(kurts)),       # 4: avg kurtosis
                    float(np.std(kurts)),        # 5: std kurtosis
                    float(np.max(kurts)),        # 6: max kurtosis
                    float(np.min(kurts)),        # 7: min kurtosis
                    float(np.mean(skews)),       # 8: avg skewness
                    float(np.std(skews)),        # 9: std skewness
                    float(np.mean(entropies)),   # 10: avg entropy
                    float(np.std(entropies)),    # 11: std entropy
                    float(np.median(stds)),      # 12: median std
                    float(np.max(stds)),         # 13: max std
                    float(np.mean([abs(s) for s in skews])),  # 14: mean |skewness|
                ],
                dtype=np.float32,
            )
        except Exception:
            return np.zeros(15, dtype=np.float32)


# =============================================================================
# SECTION 4: MULTI-BRANCH FEATURE PIPELINE
# =============================================================================

class FeaturePipeline:
    """
    Orchestrates all 4 branches:
      1. Extract raw features per branch
      2. Fit StandardScaler + PCA per branch (on training set)
      3. Transform + concatenate into a single feature vector
    """

    BRANCH_NAMES = ["clip", "effnet", "fft", "srm"]
    PCA_DIMS = {"clip": 128, "effnet": 128}  # PCA only for high-dim branches

    def __init__(self, device: torch.device):
        self.clip_ext = CLIPExtractor(device)
        self.effnet_ext = EfficientNetExtractor(device)
        self.fft_ext = FFTExtractor()
        self.srm_ext = SRMExtractor()
        self.scalers = {}
        self.pca_models = {}

    def extract_raw(self, pil_img: Image.Image) -> dict:
        """Extract raw features from all 4 branches."""
        return {
            "clip": self.clip_ext.extract(pil_img),
            "effnet": self.effnet_ext.extract(pil_img),
            "fft": self.fft_ext.extract(pil_img),
            "srm": self.srm_ext.extract(pil_img),
        }

    def extract_all_raw(
        self,
        image_paths: list,
        labels: list,
        augmenter: Optional[RealImageAugmenter] = None,
        cache_path: Optional[str] = None,
    ) -> tuple:
        """Extract raw features for all images, with caching."""
        # Check cache
        if cache_path and os.path.exists(cache_path):
            logger.info(f"  Loading cached features from {cache_path}")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data["blocks"], data["labels"]

        all_blocks = []
        valid_labels = []
        total = len(image_paths)

        for i, (path, label) in enumerate(zip(image_paths, labels)):
            if (i + 1) % 50 == 0 or i == 0:
                logger.info(f"    Extracting features: {i + 1}/{total}")

            # Apply augmentation only for REAL images (label=0)
            aug = augmenter if label == 0 else None
            img = load_and_preprocess(path, augmenter=aug)
            if img is None:
                continue

            blocks = self.extract_raw(img)
            all_blocks.append(blocks)
            valid_labels.append(label)

        logger.info(f"    Extracted {len(all_blocks)}/{total} features successfully")

        # Save cache
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"blocks": all_blocks, "labels": valid_labels}, f)
            logger.info(f"    Cached features to {cache_path}")

        return all_blocks, valid_labels

    def fit_scalers_and_pca(self, all_blocks: list):
        """Fit StandardScaler + PCA on training data (per branch)."""
        logger.info("  Fitting per-branch scalers and PCA...")
        for name in self.BRANCH_NAMES:
            features = np.stack([b[name] for b in all_blocks])

            # StandardScaler
            scaler = StandardScaler()
            scaler.fit(features)
            self.scalers[name] = scaler

            # PCA only for high-dim branches
            if name in self.PCA_DIMS:
                n_components = min(
                    self.PCA_DIMS[name], features.shape[1], features.shape[0]
                )
                pca = PCA(n_components=n_components, random_state=42)
                scaled = scaler.transform(features)
                pca.fit(scaled)
                self.pca_models[name] = pca
                explained = pca.explained_variance_ratio_.sum()
                logger.info(
                    f"    {name}: {features.shape[1]} -> {n_components} dims "
                    f"(explained variance: {explained:.3f})"
                )
            else:
                logger.info(f"    {name}: {features.shape[1]} dims (no PCA)")

    def transform(self, blocks: dict) -> np.ndarray:
        """Transform a single sample's raw blocks -> concatenated vector."""
        parts = []
        for name in self.BRANCH_NAMES:
            vec = blocks[name].reshape(1, -1)
            if name in self.scalers:
                vec = self.scalers[name].transform(vec)
            if name in self.pca_models:
                vec = self.pca_models[name].transform(vec)
            parts.append(vec.flatten())
        return np.concatenate(parts).astype(np.float32)

    def transform_batch(self, all_blocks: list) -> np.ndarray:
        """Transform a batch of raw blocks -> (N, D) matrix."""
        return np.stack([self.transform(b) for b in all_blocks])

    def get_feature_dim(self) -> int:
        clip_d = self.PCA_DIMS.get("clip", 768) if "clip" in self.pca_models else 768
        effnet_d = self.PCA_DIMS.get("effnet", 1280) if "effnet" in self.pca_models else 1280
        return clip_d + effnet_d + FFTExtractor.dim + SRMExtractor.dim

    def save(self, path: str):
        data = {
            "scalers": self.scalers,
            "pca_models": self.pca_models,
            "pca_dims": self.PCA_DIMS,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"  Saved feature processor -> {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scalers = data["scalers"]
        self.pca_models = data["pca_models"]
        self.PCA_DIMS = data.get("pca_dims", self.PCA_DIMS)


# =============================================================================
# SECTION 5: XGBOOST TRAINING
# =============================================================================

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict = None,
) -> xgb.XGBClassifier:
    """Train XGBoost binary classifier with early stopping."""

    default_params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=25,
    )

    return model


# =============================================================================
# SECTION 6: CALIBRATION
# =============================================================================

def calibrate_model(
    model: xgb.XGBClassifier,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "isotonic",
):
    """Apply Platt Scaling or Isotonic Regression calibration."""
    logger.info(f"  Calibrating with {method} regression...")

    # Try cv='prefit' first (older sklearn), fall back to cross-val
    try:
        calibrated = CalibratedClassifierCV(
            estimator=model, method=method, cv="prefit"
        )
        calibrated.fit(X_cal, y_cal)
        return calibrated
    except Exception:
        logger.info("    cv='prefit' not available, using 3-fold CV...")
        calibrated = CalibratedClassifierCV(
            estimator=model, method=method, cv=3
        )
        calibrated.fit(X_cal, y_cal)
        return calibrated


# =============================================================================
# SECTION 7: EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, label="Test") -> dict:
    """Comprehensive evaluation with all required metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    report_str = classification_report(y_test, y_pred, target_names=["REAL", "AI"])

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  {label} EVALUATION RESULTS")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Accuracy:   {acc:.4f}")
    logger.info(f"  AUC-ROC:    {auc:.4f}")
    logger.info(f"  Precision:  {prec:.4f}")
    logger.info(f"  Recall:     {rec:.4f}")
    logger.info(f"  F1 Score:   {f1:.4f}")
    logger.info(f"  Confusion Matrix:")
    logger.info(f"    TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
    logger.info(f"    FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")
    logger.info(f"\n{report_str}")
    logger.info(f"{'=' * 60}\n")

    # Check for overfitting signals
    return {
        "accuracy": round(acc, 4),
        "auc_roc": round(auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_str,
    }


def check_overfitting(train_metrics: dict, val_metrics: dict):
    """Warn if there are signs of overfitting."""
    train_acc = train_metrics["accuracy"]
    val_acc = val_metrics["accuracy"]
    gap = train_acc - val_acc

    if gap > 0.10:
        logger.warning(
            f"  OVERFITTING DETECTED: train_acc={train_acc:.4f}, "
            f"val_acc={val_acc:.4f}, gap={gap:.4f}"
        )
        logger.warning("  Recommendations:")
        logger.warning("    - Reduce max_depth (currently 6)")
        logger.warning("    - Increase reg_alpha/reg_lambda")
        logger.warning("    - Add more real images")
        logger.warning("    - Reduce n_estimators")
    elif gap > 0.05:
        logger.info(
            f"  Slight overfitting: gap={gap:.4f} (acceptable)"
        )
    else:
        logger.info(f"  No overfitting detected: gap={gap:.4f}")


# =============================================================================
# SECTION 8: REPORT GENERATION
# =============================================================================

def generate_report(
    output_dir: str,
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
    config: dict,
):
    """Generate a comprehensive evaluation report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "pipeline": "multi-branch-xgboost",
        "architecture": {
            "branches": [
                {"name": "CLIP ViT-L/14", "dim": 768, "pca_dim": 128, "frozen": True},
                {"name": "EfficientNetV2-L", "dim": 1280, "pca_dim": 128, "frozen": True},
                {"name": "FFT frequency", "dim": 8, "pca_dim": None, "frozen": True},
                {"name": "SRM noise", "dim": 15, "pca_dim": None, "frozen": True},
            ],
            "total_features_before_pca": 2071,
            "total_features_after_pca": 279,
            "classifier": "XGBoost",
            "calibration": config.get("calibration", "isotonic"),
        },
        "dataset": {
            "samples_per_class": config.get("samples_per_class", "unknown"),
            "total_images": config.get("total_images", "unknown"),
            "split": "70/15/15 (train/val/test)",
            "augmentation": "Real images only (flip, color jitter, gaussian noise)",
        },
        "results": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
            "overfitting_gap": round(
                train_metrics["accuracy"] - val_metrics["accuracy"], 4
            ),
        },
        "xgboost_params": config.get("xgb_params", {}),
    }

    # Save JSON report
    report_path = os.path.join(output_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"  Saved report -> {report_path}")

    # Save human-readable text report
    txt_path = os.path.join(output_dir, "training_report.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  REAL vs AI IMAGE DETECTION - TRAINING REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"  Date: {report['timestamp']}\n")
        f.write(f"  Pipeline: {report['pipeline']}\n\n")

        f.write("  ARCHITECTURE:\n")
        for b in report["architecture"]["branches"]:
            pca = f" -> PCA({b['pca_dim']})" if b["pca_dim"] else ""
            f.write(f"    {b['name']}: {b['dim']}-dim{pca}\n")
        f.write(f"    Total features: {report['architecture']['total_features_after_pca']}\n")
        f.write(f"    Classifier: {report['architecture']['classifier']}\n")
        f.write(f"    Calibration: {report['architecture']['calibration']}\n\n")

        f.write("  DATASET:\n")
        f.write(f"    Samples/class: {report['dataset']['samples_per_class']}\n")
        f.write(f"    Split: {report['dataset']['split']}\n")
        f.write(f"    Augmentation: {report['dataset']['augmentation']}\n\n")

        for split_name in ["train", "validation", "test"]:
            m = report["results"][split_name]
            f.write(f"  {split_name.upper()} RESULTS:\n")
            f.write(f"    Accuracy:  {m['accuracy']:.4f}\n")
            f.write(f"    AUC-ROC:   {m['auc_roc']:.4f}\n")
            f.write(f"    Precision: {m['precision']:.4f}\n")
            f.write(f"    Recall:    {m['recall']:.4f}\n")
            f.write(f"    F1 Score:  {m['f1']:.4f}\n\n")

        gap = report["results"]["overfitting_gap"]
        f.write(f"  OVERFITTING GAP: {gap:.4f}\n")
        if gap > 0.10:
            f.write("  STATUS: OVERFITTING DETECTED - needs tuning\n")
        elif gap > 0.05:
            f.write("  STATUS: SLIGHT OVERFITTING - acceptable\n")
        else:
            f.write("  STATUS: NO OVERFITTING - good\n")

        f.write("\n" + "=" * 70 + "\n")

    logger.info(f"  Saved text report -> {txt_path}")
    return report


# =============================================================================
# SECTION 9: MAIN TRAINING ORCHESTRATOR
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train REAL vs AI image detector (multi-branch XGBoost)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data source
    data_group = parser.add_argument_group("Data Source")
    data_group.add_argument("--real-dir", type=str, help="Directory of real images")
    data_group.add_argument("--ai-dir", type=str, help="Directory of AI-generated images")
    data_group.add_argument("--auto-download", action="store_true",
                            help="Auto-download datasets from HuggingFace")
    data_group.add_argument("--data-dir", type=str, default="./data",
                            help="Root dir for downloaded data")
    data_group.add_argument("--samples-per-class", type=int, default=1000,
                            help="Max samples per class (default: 1000)")

    # Training
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--max-depth", type=int, default=6)
    train_group.add_argument("--learning-rate", type=float, default=0.1)
    train_group.add_argument("--n-estimators", type=int, default=200)
    train_group.add_argument("--subsample", type=float, default=0.8)
    train_group.add_argument("--calibration", type=str, default="isotonic",
                             choices=["isotonic", "sigmoid"])
    train_group.add_argument("--no-pca", action="store_true", help="Disable PCA")
    train_group.add_argument("--no-augment", action="store_true",
                             help="Disable real image augmentation")

    # Output
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--output-dir", type=str, default=None,
                           help="Save models here (default: ./weights)")
    out_group.add_argument("--no-cache", action="store_true",
                           help="Don't cache features")

    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "weights"
    )
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()

    # ===================================================================
    # STEP 1: DATASET COLLECTION
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  STEP 1: DATASET COLLECTION")
    logger.info("=" * 70)

    if args.real_dir and args.ai_dir:
        real_dir, ai_dir = args.real_dir, args.ai_dir
    elif args.auto_download:
        real_dir = download_real_images(args.data_dir, args.samples_per_class)
        ai_dir = download_ai_images(args.data_dir, args.samples_per_class)
    else:
        # Try local dataset first
        local_v2 = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "real-vs-ai-detector", "dataset_v2",
        )
        if os.path.isdir(local_v2):
            real_dir = os.path.join(local_v2, "real")
            ai_dir = os.path.join(local_v2, "ai")
            logger.info(f"  Found local dataset_v2")
        else:
            real_dir = download_real_images(args.data_dir, args.samples_per_class)
            ai_dir = download_ai_images(args.data_dir, args.samples_per_class)

    real_paths = collect_images(real_dir, args.samples_per_class)
    ai_paths = collect_images(ai_dir, args.samples_per_class)

    if len(real_paths) == 0 or len(ai_paths) == 0:
        logger.error("No images found! Provide --real-dir and --ai-dir")
        logger.error(f"  Real dir: {real_dir} ({len(real_paths)} images)")
        logger.error(f"  AI dir:   {ai_dir} ({len(ai_paths)} images)")
        sys.exit(1)

    # Strict 50:50 balance
    min_count = min(len(real_paths), len(ai_paths))
    real_paths = real_paths[:min_count]
    ai_paths = ai_paths[:min_count]
    logger.info(
        f"  Balanced dataset: {min_count} real + {min_count} AI = {min_count * 2} total"
    )

    all_paths = real_paths + ai_paths
    all_labels = [0] * len(real_paths) + [1] * len(ai_paths)  # 0=REAL, 1=AI

    # ===================================================================
    # STEP 2: TRAIN / VAL / TEST SPLIT (70/15/15)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  STEP 2: DATA SPLIT (70/15/15)")
    logger.info("=" * 70)

    indices = np.arange(len(all_labels))
    labels_arr = np.array(all_labels)

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=42, stratify=labels_arr
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=42,
        stratify=labels_arr[temp_idx],
    )

    train_paths = [all_paths[i] for i in train_idx]
    train_labels = labels_arr[train_idx].tolist()
    val_paths = [all_paths[i] for i in val_idx]
    val_labels = labels_arr[val_idx].tolist()
    test_paths = [all_paths[i] for i in test_idx]
    test_labels = labels_arr[test_idx].tolist()

    logger.info(f"  Train: {len(train_idx)} (real: {sum(1 for l in train_labels if l==0)}, "
                f"ai: {sum(1 for l in train_labels if l==1)})")
    logger.info(f"  Val:   {len(val_idx)} (real: {sum(1 for l in val_labels if l==0)}, "
                f"ai: {sum(1 for l in val_labels if l==1)})")
    logger.info(f"  Test:  {len(test_idx)} (real: {sum(1 for l in test_labels if l==0)}, "
                f"ai: {sum(1 for l in test_labels if l==1)})")

    # ===================================================================
    # STEP 3: INITIALIZE FEATURE EXTRACTORS
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  STEP 3: INITIALIZE FEATURE EXTRACTORS")
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")

    pipeline = FeaturePipeline(device)

    if args.no_pca:
        pipeline.PCA_DIMS = {}
        logger.info("  PCA disabled")

    # ===================================================================
    # STEP 4: FEATURE EXTRACTION
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  STEP 4: FEATURE EXTRACTION")
    logger.info("=" * 70)

    augmenter = None if args.no_augment else RealImageAugmenter()
    if augmenter:
        logger.info("  Real image augmentation: ENABLED (flip, color jitter, noise)")
    else:
        logger.info("  Real image augmentation: DISABLED")

    cache_dir = os.path.join(output_dir, "feature_cache")

    logger.info("  Extracting TRAINING features...")
    train_cache = None if args.no_cache else os.path.join(cache_dir, "train_features.pkl")
    train_blocks, train_labels = pipeline.extract_all_raw(
        train_paths, train_labels, augmenter=augmenter, cache_path=train_cache
    )

    logger.info("  Extracting VALIDATION features...")
    val_cache = None if args.no_cache else os.path.join(cache_dir, "val_features.pkl")
    val_blocks, val_labels = pipeline.extract_all_raw(
        val_paths, val_labels, augmenter=None, cache_path=val_cache  # No augment for val
    )

    logger.info("  Extracting TEST features...")
    test_cache = None if args.no_cache else os.path.join(cache_dir, "test_features.pkl")
    test_blocks, test_labels = pipeline.extract_all_raw(
        test_paths, test_labels, augmenter=None, cache_path=test_cache  # No augment for test
    )

    # ===================================================================
    # STEP 5: FIT SCALERS + PCA (ON TRAINING DATA ONLY)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  STEP 5: FIT SCALERS + PCA")
    logger.info("=" * 70)

    pipeline.fit_scalers_and_pca(train_blocks)

    # Transform all splits
    X_train = pipeline.transform_batch(train_blocks)
    X_val = pipeline.transform_batch(val_blocks)
    X_test = pipeline.transform_batch(test_blocks)
    y_train = np.array(train_labels)
    y_val = np.array(val_labels)
    y_test = np.array(test_labels)

    logger.info(f"  Feature vector dimension: {X_train.shape[1]}")
    logger.info(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    # ===================================================================
    # STEP 6: TRAIN XGBOOST
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  STEP 6: TRAIN XGBOOST CLASSIFIER")
    logger.info("=" * 70)

    xgb_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
    }
    logger.info(f"  Hyperparameters: {xgb_params}")

    model = train_xgboost(X_train, y_train, X_val, y_val, xgb_params)

    # Evaluate on train and val (for overfitting check)
    logger.info("\n  Evaluating on TRAINING set...")
    train_metrics = evaluate_model(model, X_train, y_train, "Train (uncalibrated)")

    logger.info("  Evaluating on VALIDATION set...")
    val_metrics_uncal = evaluate_model(model, X_val, y_val, "Validation (uncalibrated)")

    check_overfitting(train_metrics, val_metrics_uncal)

    # ===================================================================
    # STEP 7: CONFIDENCE CALIBRATION
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  STEP 7: CONFIDENCE CALIBRATION")
    logger.info("=" * 70)

    calibrated_model = calibrate_model(model, X_val, y_val, args.calibration)

    # ===================================================================
    # STEP 8: FINAL EVALUATION ON HELD-OUT TEST SET
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  STEP 8: FINAL EVALUATION (TEST SET)")
    logger.info("=" * 70)

    test_metrics = evaluate_model(calibrated_model, X_test, y_test, "Test (calibrated)")

    # Also evaluate calibrated on val for report
    val_metrics_cal = evaluate_model(calibrated_model, X_val, y_val, "Validation (calibrated)")

    # ===================================================================
    # STEP 9: SAVE ALL ARTIFACTS
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("  STEP 9: SAVE ARTIFACTS")
    logger.info("=" * 70)

    # 1. Raw XGBoost model (.json)
    model_path = os.path.join(output_dir, "xgboost_model.json")
    model.save_model(model_path)
    logger.info(f"  Saved XGBoost model -> {model_path}")

    # 2. Calibrated model (.pkl)
    cal_path = os.path.join(output_dir, "calibrated_model.pkl")
    with open(cal_path, "wb") as f:
        pickle.dump(calibrated_model, f)
    logger.info(f"  Saved calibrated model -> {cal_path}")

    # 3. Feature processor (scalers + PCA)
    proc_path = os.path.join(output_dir, "feature_processor.pkl")
    pipeline.save(proc_path)

    # 4. Training metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2, default=str)
    logger.info(f"  Saved metrics -> {metrics_path}")

    # 5. Full report
    config = {
        "samples_per_class": min_count,
        "total_images": min_count * 2,
        "calibration": args.calibration,
        "xgb_params": xgb_params,
    }
    generate_report(output_dir, train_metrics, val_metrics_cal, test_metrics, config)

    # ===================================================================
    # DONE
    # ===================================================================
    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 70)
    logger.info("  TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test AUC-ROC:   {test_metrics['auc_roc']:.4f}")
    logger.info(f"  Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Test Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"  Test F1:        {test_metrics['f1']:.4f}")
    logger.info(f"  Total time:     {elapsed/60:.1f} minutes")
    logger.info(f"  Artifacts:      {output_dir}/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
