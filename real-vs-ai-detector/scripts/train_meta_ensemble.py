"""
scripts/train_meta_ensemble.py — Train meta-ensemble models.

Runs all sub-models on the dataset, collects predictions + FFT features,
and trains Logistic Regression + XGBoost meta-classifiers.

Usage:
    python scripts/train_meta_ensemble.py
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, BASE_DIR
from models.loader import model_manager
from services.predictor import predict_single
from services.fft_features import extract_fft_features
from services.patch_analyzer import analyze_patches
from utils.logger import setup_logger

logger = setup_logger("train_meta_ensemble")

META_DIR = os.path.join(BASE_DIR, "model", "meta")


def collect_features(image_paths: list, labels: list) -> tuple:
    """
    Run all sub-models on each image and build feature matrix.

    Returns:
        (X, y) where X is (N, 7) feature matrix and y is labels.
    """
    X, y = [], []

    for i, (path, label) in enumerate(zip(image_paths, labels)):
        try:
            # Sub-model predictions
            resnet_result = predict_single(path, "resnet")
            effnet_result = predict_single(path, "efficientnet")

            resnet_score = resnet_result.get("raw_score", 0.5)
            effnet_score = effnet_result.get("raw_score", 0.5)

            # FFT features
            fft = extract_fft_features(path)
            fft_hf = fft.get("high_freq_ratio", 0.0)
            fft_nv = fft.get("noise_variance", 0.0)
            fft_sc = fft.get("spectral_centroid", 0.0)

            # Patch score (simplified — just overall)
            patch_result = analyze_patches(path, patch_size=128, classifier_name="efficientnet")
            patch_score = patch_result.get("overall_score", 0.5)

            features = [resnet_score, effnet_score,
                        patch_score, fft_hf, fft_nv, fft_sc]

            X.append(features)
            y.append(label)

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(image_paths)} images...")

        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            continue

    return np.array(X, dtype=np.float32), np.array(y)


def main():
    logger.info("=" * 60)
    logger.info("META-ENSEMBLE TRAINING")
    logger.info("=" * 60)

    # Discover dataset
    real_dir = os.path.join(DATASET_DIR, "real")
    ai_dir = os.path.join(DATASET_DIR, "ai")

    if not os.path.isdir(real_dir) or not os.path.isdir(ai_dir):
        logger.error(f"Dataset not found at {DATASET_DIR}. Need 'real/' and 'ai/' subdirs.")
        sys.exit(1)

    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    ai_images = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    # Limit to avoid very long training times
    max_per_class = 100
    real_images = real_images[:max_per_class]
    ai_images = ai_images[:max_per_class]

    logger.info(f"Real images: {len(real_images)}")
    logger.info(f"AI images:   {len(ai_images)}")

    image_paths = real_images + ai_images
    labels = [0] * len(real_images) + [1] * len(ai_images)  # 0=Real, 1=AI

    # Collect features
    logger.info("Collecting sub-model predictions...")
    X, y = collect_features(image_paths, labels)
    logger.info(f"Feature matrix: {X.shape} ({len(y)} samples)")

    if len(X) < 10:
        logger.error("Not enough data to train. Need at least 10 samples.")
        sys.exit(1)

    # Train Logistic Regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr_scores = cross_val_score(lr, X, y, cv=min(5, len(y) // 2), scoring='accuracy')
    lr.fit(X, y)
    logger.info(f"LR accuracy: {lr_scores.mean():.4f} (+/- {lr_scores.std():.4f})")

    # Train XGBoost
    try:
        import xgboost as xgb
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric='logloss',
            use_label_encoder=False,
        )
        xgb_scores = cross_val_score(xgb_model, X, y, cv=min(5, len(y) // 2), scoring='accuracy')
        xgb_model.fit(X, y)
        logger.info(f"XGBoost accuracy: {xgb_scores.mean():.4f} (+/- {xgb_scores.std():.4f})")
    except ImportError:
        xgb_model = None
        logger.warning("XGBoost not installed. Skipping.")

    # Save models
    os.makedirs(META_DIR, exist_ok=True)

    lr_path = os.path.join(META_DIR, "meta_lr.pkl")
    with open(lr_path, "wb") as f:
        pickle.dump(lr, f)
    logger.info(f"Saved: {lr_path}")

    if xgb_model is not None:
        xgb_path = os.path.join(META_DIR, "meta_xgb.pkl")
        with open(xgb_path, "wb") as f:
            pickle.dump(xgb_model, f)
        logger.info(f"Saved: {xgb_path}")

    logger.info("=" * 60)
    logger.info("META-ENSEMBLE TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
