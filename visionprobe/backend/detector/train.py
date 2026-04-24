"""
Training pipeline for AI vs Real image detection.

Workflow:
  1. Load real images (COCO / local) + AI images (DiffusionDB / Kaggle / local)
  2. Extract multi-branch features (CLIP + EfficientNet + FFT + SRM)
  3. Fit per-block scalers + PCA
  4. Train XGBoost classifier
  5. Calibrate with Platt Scaling or Isotonic Regression
  6. Evaluate and save model artifacts

Usage:
  python train.py --real-dir ./data/real --ai-dir ./data/ai
  python train.py --samples-per-class 500
"""

import argparse
import logging
import os
import sys
import time
import pickle
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
import xgboost as xgb

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("visionprobe.train")

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
CACHE_DIR = os.path.join(WEIGHTS_DIR, "feature_cache")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def collect_images(directory: str, max_count: int = None) -> list:
    """Collect image paths from a directory."""
    paths = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if Path(f).suffix.lower() in VALID_EXTENSIONS:
                paths.append(os.path.join(root, f))
    if max_count and len(paths) > max_count:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(paths), max_count, replace=False)
        paths = [paths[i] for i in sorted(indices)]
    logger.info(f"  Collected {len(paths)} images from {directory}")
    return paths


def load_image_safe(path: str) -> Image.Image:
    """Load and convert image to RGB, handling errors."""
    try:
        img = Image.open(path).convert("RGB")
        # Resize large images to prevent memory issues
        w, h = img.size
        if max(w, h) > 2048:
            scale = 2048 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def download_sample_dataset(output_dir: str, samples_per_class: int = 500):
    """Download a small sample dataset from HuggingFace for training."""
    real_dir = os.path.join(output_dir, "real")
    ai_dir = os.path.join(output_dir, "ai")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)

    # Check if we already have enough images
    existing_real = len(collect_images(real_dir))
    existing_ai = len(collect_images(ai_dir))
    if existing_real >= samples_per_class and existing_ai >= samples_per_class:
        logger.info(f"Dataset already exists ({existing_real} real, {existing_ai} AI)")
        return real_dir, ai_dir

    try:
        from datasets import load_dataset
        logger.info("Downloading AI images from DiffusionDB (subset)...")
        ds = load_dataset("poloclub/diffusiondb", "2m_first_1k", split="train",
                         trust_remote_code=True)
        count = 0
        for i, item in enumerate(ds):
            if count >= samples_per_class:
                break
            try:
                img = item["image"]
                if img is not None:
                    img.save(os.path.join(ai_dir, f"diffusion_{i:05d}.jpg"), "JPEG", quality=95)
                    count += 1
            except Exception:
                continue
        logger.info(f"  Downloaded {count} AI images from DiffusionDB")
    except Exception as e:
        logger.warning(f"DiffusionDB download failed: {e}. Place AI images in {ai_dir} manually.")

    try:
        from datasets import load_dataset
        logger.info("Downloading real images from COCO (subset via HuggingFace)...")
        ds = load_dataset("detection-datasets/coco", split="val",
                         trust_remote_code=True)
        count = 0
        seen = set()
        for i, item in enumerate(ds):
            if count >= samples_per_class:
                break
            try:
                img = item["image"]
                if img is not None and i not in seen:
                    seen.add(i)
                    img.save(os.path.join(real_dir, f"coco_{i:05d}.jpg"), "JPEG", quality=95)
                    count += 1
            except Exception:
                continue
        logger.info(f"  Downloaded {count} real images from COCO")
    except Exception as e:
        logger.warning(f"COCO download failed: {e}. Place real images in {real_dir} manually.")

    return real_dir, ai_dir


def extract_features_with_cache(
    image_paths: list,
    labels: list,
    extractor,
    cache_path: str = None,
) -> tuple:
    """Extract features for all images with optional disk cache."""
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached features from {cache_path}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["blocks"], data["labels"]

    all_blocks = []
    valid_labels = []

    for i, (path, label) in enumerate(zip(image_paths, labels)):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info(f"  Extracting features: {i+1}/{len(image_paths)}")

        img = load_image_safe(path)
        if img is None:
            continue

        blocks = extractor.extract_raw(img)
        all_blocks.append(blocks)
        valid_labels.append(label)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"blocks": all_blocks, "labels": valid_labels}, f)
        logger.info(f"Cached {len(all_blocks)} feature sets to {cache_path}")

    return all_blocks, valid_labels


def train_xgboost(X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
    """Train XGBoost classifier."""
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1.0,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    return model


def calibrate_model(model, X_val, y_val, method="isotonic"):
    """Apply Platt Scaling or Isotonic Regression for calibration."""
    logger.info(f"Calibrating model with {method} regression...")
    try:
        # Try newer sklearn API first
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv="prefit",
        )
        calibrated.fit(X_val, y_val)
        return calibrated
    except Exception:
        # Fallback: use 3-fold CV calibration
        logger.info("  cv='prefit' not supported, using 3-fold CV calibration...")
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv=3,
        )
        calibrated.fit(X_val, y_val)
        return calibrated


def evaluate(model, X_test, y_test, label="Test"):
    """Evaluate and print metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"\n{'='*50}")
    logger.info(f"  {label} Results:")
    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  AUC-ROC:   {auc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  F1 Score:  {f1:.4f}")
    logger.info(f"  Confusion Matrix:\n{cm}")
    logger.info(f"{'='*50}\n")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['REAL', 'AI'])}")

    return {"accuracy": acc, "auc_roc": auc, "precision": prec, "recall": rec, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Train AI vs Real image detector")
    parser.add_argument("--real-dir", type=str, default=None, help="Directory of real images")
    parser.add_argument("--ai-dir", type=str, default=None, help="Directory of AI-generated images")
    parser.add_argument("--samples-per-class", type=int, default=500, help="Max samples per class")
    parser.add_argument("--data-dir", type=str, default=None, help="Auto-download dataset to this dir")
    parser.add_argument("--no-pca", action="store_true", help="Disable PCA")
    parser.add_argument("--calibration", type=str, default="isotonic", choices=["sigmoid", "isotonic"])
    parser.add_argument("--output-dir", type=str, default=None, help="Save models here")
    parser.add_argument("--no-cache", action="store_true", help="Disable feature cache")
    args = parser.parse_args()

    output_dir = args.output_dir or WEIGHTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    # 1. Collect data
    logger.info("=" * 60)
    logger.info("  STEP 1: Collecting training data")
    logger.info("=" * 60)

    if args.real_dir and args.ai_dir:
        real_dir, ai_dir = args.real_dir, args.ai_dir
    elif args.data_dir:
        real_dir, ai_dir = download_sample_dataset(args.data_dir, args.samples_per_class)
    else:
        data_dir = os.path.join(os.path.dirname(WEIGHTS_DIR), "data")
        real_dir, ai_dir = download_sample_dataset(data_dir, args.samples_per_class)

    real_paths = collect_images(real_dir, args.samples_per_class)
    ai_paths = collect_images(ai_dir, args.samples_per_class)

    if len(real_paths) == 0 or len(ai_paths) == 0:
        logger.error("No images found! Place images in data/real/ and data/ai/ directories.")
        logger.error(f"  Real dir: {real_dir}")
        logger.error(f"  AI dir:   {ai_dir}")
        sys.exit(1)

    # Balance 50:50
    min_count = min(len(real_paths), len(ai_paths))
    real_paths = real_paths[:min_count]
    ai_paths = ai_paths[:min_count]
    logger.info(f"Balanced dataset: {min_count} real + {min_count} AI = {min_count * 2} total")

    all_paths = real_paths + ai_paths
    all_labels = [0] * len(real_paths) + [1] * len(ai_paths)  # 0=REAL, 1=AI

    # 2. Initialize extractors
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 2: Initializing feature extractors")
    logger.info("=" * 60)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # OpenCLIP
    clip_ext = None
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        clip_ext = None  # Import from feature_extractors
        from detector.feature_extractors import CLIPFeatureExtractor
        clip_ext = CLIPFeatureExtractor(model, preprocess, tokenizer, device)
        logger.info("  OpenCLIP ViT-L/14 loaded")
    except Exception as e:
        logger.warning(f"  OpenCLIP failed: {e}. CLIP features will be zeros.")

    # EfficientNet
    effnet_ext = None
    try:
        import timm
        backbone = timm.create_model(
            "tf_efficientnetv2_l.in21k_ft_in1k",
            pretrained=True, num_classes=0
        ).to(device).eval()
        from detector.feature_extractors import EfficientNetFeatureExtractor
        effnet_ext = EfficientNetFeatureExtractor(backbone, device)
        logger.info("  EfficientNetV2-L loaded")
    except Exception as e:
        logger.warning(f"  EfficientNet failed: {e}. EfficientNet features will be zeros.")

    from detector.feature_extractors import MultiFeatureExtractor
    extractor = MultiFeatureExtractor(
        clip_extractor=clip_ext,
        effnet_extractor=effnet_ext,
        use_pca=not args.no_pca,
    )

    # 3. Extract features
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 3: Extracting features from all images")
    logger.info("=" * 60)

    cache_path = None if args.no_cache else os.path.join(CACHE_DIR, "features.pkl")
    all_blocks, valid_labels = extract_features_with_cache(
        all_paths, all_labels, extractor, cache_path
    )

    # 4. Split data (70/15/15)
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 4: Splitting data (70/15/15)")
    logger.info("=" * 60)

    indices = np.arange(len(valid_labels))
    labels_arr = np.array(valid_labels)

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=labels_arr
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42,
        stratify=labels_arr[temp_idx]
    )

    train_blocks = [all_blocks[i] for i in train_idx]
    val_blocks = [all_blocks[i] for i in val_idx]
    test_blocks = [all_blocks[i] for i in test_idx]
    y_train = labels_arr[train_idx]
    y_val = labels_arr[val_idx]
    y_test = labels_arr[test_idx]

    logger.info(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # 5. Fit scalers + PCA on training data
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 5: Fitting scalers and PCA")
    logger.info("=" * 60)

    extractor.fit_scalers(train_blocks)

    # Transform all splits
    X_train = extractor.process_batch(train_blocks)
    X_val = extractor.process_batch(val_blocks)
    X_test = extractor.process_batch(test_blocks)

    logger.info(f"  Feature vector dim: {X_train.shape[1]}")

    # 6. Train XGBoost
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 6: Training XGBoost classifier")
    logger.info("=" * 60)

    model = train_xgboost(X_train, y_train, X_val, y_val)
    evaluate(model, X_val, y_val, "Validation (uncalibrated)")

    # 7. Calibrate
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 7: Calibrating model")
    logger.info("=" * 60)

    calibrated_model = calibrate_model(model, X_val, y_val, args.calibration)

    # 8. Evaluate
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 8: Final evaluation on test set")
    logger.info("=" * 60)

    metrics = evaluate(calibrated_model, X_test, y_test, "Test (calibrated)")

    # 9. Save all artifacts
    logger.info("\n" + "=" * 60)
    logger.info("  STEP 9: Saving model artifacts")
    logger.info("=" * 60)

    # XGBoost raw model
    model_path = os.path.join(output_dir, "xgboost_model.json")
    model.save_model(model_path)
    logger.info(f"  Saved XGBoost model → {model_path}")

    # Calibrated model (pickle)
    cal_path = os.path.join(output_dir, "calibrated_model.pkl")
    with open(cal_path, "wb") as f:
        pickle.dump(calibrated_model, f)
    logger.info(f"  Saved calibrated model → {cal_path}")

    # Feature processor (scalers + PCA)
    proc_path = os.path.join(output_dir, "feature_processor.pkl")
    extractor.save(proc_path)
    logger.info(f"  Saved feature processor → {proc_path}")

    # Metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Saved metrics → {metrics_path}")

    logger.info("\n✅ Training complete!")
    logger.info(f"   Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"   Test AUC-ROC:  {metrics['auc_roc']:.4f}")
    logger.info(f"   Test F1:       {metrics['f1']:.4f}")
    logger.info(f"   Artifacts saved to: {output_dir}/")


if __name__ == "__main__":
    main()
