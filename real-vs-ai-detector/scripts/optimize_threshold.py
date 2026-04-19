"""
scripts/optimize_threshold.py — Find the optimal classification threshold.

Sweeps thresholds from 0.30 to 0.70 and selects the one that maximizes
F1-score for balanced performance across both Real and AI classes.

Saves result to model/meta/optimal_threshold.json.

Usage:
    python scripts/optimize_threshold.py
"""

import os
import sys
import json
import glob
import random
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, BASE_DIR
from models.loader import model_manager
from services.predictor import predict_single
from services.ensemble import weighted_voting
from utils.logger import setup_logger

logger = setup_logger("optimize_threshold")

META_DIR = os.path.join(BASE_DIR, "model", "meta")
THRESHOLD_PATH = os.path.join(META_DIR, "optimal_threshold.json")


def collect_predictions(num_samples=200):
    """
    Run ensemble prediction on dataset and collect raw scores + true labels.

    Returns:
        (raw_scores, true_labels) — numpy arrays
    """
    real_dir = os.path.join(DATASET_DIR, "real")
    ai_dir = os.path.join(DATASET_DIR, "ai")

    real_imgs = glob.glob(os.path.join(real_dir, "*.*"))
    ai_imgs = glob.glob(os.path.join(ai_dir, "*.*"))

    if not real_imgs or not ai_imgs:
        logger.error(f"Dataset not found at {DATASET_DIR}")
        sys.exit(1)

    random.seed(42)
    real_sample = random.sample(real_imgs, min(num_samples, len(real_imgs)))
    ai_sample = random.sample(ai_imgs, min(num_samples, len(ai_imgs)))

    paths = real_sample + ai_sample
    labels = [0] * len(real_sample) + [1] * len(ai_sample)

    logger.info(f"Collecting predictions for {len(paths)} images...")

    scores = []
    for i, path in enumerate(paths):
        try:
            # Get individual predictions
            predictions = {}
            for model_name in model_manager.available_models():
                result = predict_single(path, model_name)
                predictions[model_name] = result

            # Ensemble score
            ensemble = weighted_voting(predictions)
            raw_score = ensemble.get("raw_score", 0.5)
            scores.append(raw_score)
        except Exception as e:
            logger.warning(f"Error on {path}: {e}")
            scores.append(0.5)

        if (i + 1) % 25 == 0:
            logger.info(f"  Processed {i+1}/{len(paths)}")

    return np.array(scores), np.array(labels)


def sweep_thresholds(scores, labels, start=0.30, end=0.70, step=0.01):
    """
    Sweep thresholds and compute metrics at each.

    Returns:
        List of dicts with threshold, accuracy, precision, recall, f1.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    results = []
    for thresh in np.arange(start, end + step, step):
        preds = (scores >= thresh).astype(int)

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        # Per-class precision/recall
        prec_real = precision_score(labels, preds, pos_label=0, zero_division=0)
        rec_real = recall_score(labels, preds, pos_label=0, zero_division=0)

        results.append({
            "threshold": round(float(thresh), 3),
            "accuracy": round(float(acc), 4),
            "precision_ai": round(float(prec), 4),
            "recall_ai": round(float(rec), 4),
            "precision_real": round(float(prec_real), 4),
            "recall_real": round(float(rec_real), 4),
            "f1_score": round(float(f1), 4),
        })

    return results


def main():
    print("=" * 60)
    print("  Threshold Optimization — Real vs AI Detector")
    print("=" * 60)

    scores, labels = collect_predictions(num_samples=100)
    results = sweep_thresholds(scores, labels)

    # Find best F1
    best = max(results, key=lambda r: r["f1_score"])

    print("\n" + "=" * 60)
    print("  Threshold Sweep Results")
    print("=" * 60)
    print(f"  {'Thresh':>7} {'Acc':>7} {'F1':>7} {'P(AI)':>7} {'R(AI)':>7} {'P(Real)':>8} {'R(Real)':>8}")
    print("  " + "-" * 56)

    for r in results:
        marker = " <-- BEST" if r["threshold"] == best["threshold"] else ""
        print(f"  {r['threshold']:>7.3f} {r['accuracy']:>7.4f} {r['f1_score']:>7.4f} "
              f"{r['precision_ai']:>7.4f} {r['recall_ai']:>7.4f} "
              f"{r['precision_real']:>8.4f} {r['recall_real']:>8.4f}{marker}")

    print(f"\n  ✅ Optimal Threshold: {best['threshold']}")
    print(f"     F1 Score: {best['f1_score']}")
    print(f"     Accuracy: {best['accuracy']}")
    print(f"     Precision (AI): {best['precision_ai']} | Recall (AI): {best['recall_ai']}")
    print(f"     Precision (Real): {best['precision_real']} | Recall (Real): {best['recall_real']}")

    # Save
    os.makedirs(META_DIR, exist_ok=True)
    output = {
        "optimal_threshold": best["threshold"],
        "best_f1": best["f1_score"],
        "best_accuracy": best["accuracy"],
        "all_results": results,
    }
    with open(THRESHOLD_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  💾 Saved → {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()
