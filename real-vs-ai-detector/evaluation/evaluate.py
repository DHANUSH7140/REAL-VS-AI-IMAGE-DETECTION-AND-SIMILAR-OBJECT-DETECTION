"""
evaluation/evaluate.py — Comprehensive model evaluation with metrics and plots.

Generates:
    - Accuracy, Precision, Recall, F1 Score
    - Confusion Matrix plots
    - ROC Curves with AUC scores
    - Model comparison reports
"""

import os
import glob
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

from config import DATASET_DIR, EVAL_FOLDER, MODEL_CONFIGS
from models.loader import model_manager
from utils.preprocessing import preprocess_image
from services.predictor import predict_single
from services.ensemble import weighted_voting
from utils.logger import setup_logger

logger = setup_logger("evaluation")

# Plot styling
plt.rcParams.update({
    'figure.facecolor': '#0b0f19',
    'axes.facecolor': '#0b0f19',
    'text.color': '#e2e8f0',
    'axes.labelcolor': '#e2e8f0',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
})


def _collect_samples(num_samples: int = 200) -> tuple:
    """
    Collect image file paths and true labels from the dataset.

    Returns:
        Tuple of (file_paths, true_labels) where 0=real, 1=ai.
    """
    real_imgs = glob.glob(os.path.join(DATASET_DIR, "real", "*.*"))
    ai_imgs = glob.glob(os.path.join(DATASET_DIR, "ai", "*.*"))

    if not real_imgs or not ai_imgs:
        raise FileNotFoundError(
            f"Dataset not found or empty at {DATASET_DIR}. "
            "Need dataset_v2/real/ and dataset_v2/ai/ folders."
        )

    random.seed(42)
    real_sample = random.sample(real_imgs, min(num_samples, len(real_imgs)))
    ai_sample = random.sample(ai_imgs, min(num_samples, len(ai_imgs)))

    paths = real_sample + ai_sample
    labels = [0] * len(real_sample) + [1] * len(ai_sample)

    return paths, labels


def evaluate_model(model_name: str, num_samples: int = 200) -> dict:
    """
    Evaluate a single model on the dataset.

    Args:
        model_name:   Model identifier.
        num_samples:  Number of samples per class.

    Returns:
        Dict with metrics: accuracy, precision, recall, f1, auc_score,
        and paths to generated plots.
    """
    logger.info(f"Evaluating model: {model_name}")

    paths, y_true = _collect_samples(num_samples)
    y_true = np.array(y_true)
    y_scores = []
    y_pred = []

    model = model_manager.get(model_name)

    for i, img_path in enumerate(paths):
        try:
            img_array = preprocess_image(img_path, model_name)
            raw_score = float(model.predict(img_array, verbose=0)[0][0])
            y_scores.append(raw_score)
            y_pred.append(1 if raw_score >= 0.5 else 0)
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
            y_scores.append(0.5)
            y_pred.append(0)

        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i + 1}/{len(paths)} images")

    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # ── Metrics ─────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    # ── Confusion Matrix Plot ───────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "AI"],
                yticklabels=["Real", "AI"], ax=ax,
                cbar_kws={'label': 'Count'})
    display_name = MODEL_CONFIGS.get(model_name, {}).get("display_name", model_name)
    ax.set_title(f"{display_name} — Confusion Matrix", fontsize=13, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()

    cm_path = os.path.join(EVAL_FOLDER, f"confusion_matrix_{model_name}.png")
    plt.savefig(cm_path, dpi=120, facecolor='#0b0f19')
    plt.close()

    # ── ROC Curve Plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#818cf8', linewidth=2,
            label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.15, color='#6366f1')
    ax.set_title(f"{display_name} — ROC Curve", fontsize=13, pad=12)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.1)
    plt.tight_layout()

    roc_path = os.path.join(EVAL_FOLDER, f"roc_curve_{model_name}.png")
    plt.savefig(roc_path, dpi=120, facecolor='#0b0f19')
    plt.close()

    metrics = {
        "model": model_name,
        "display_name": display_name,
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "auc_score": round(auc_score, 4),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_plot": f"/static/evaluation/confusion_matrix_{model_name}.png",
        "roc_curve_plot": f"/static/evaluation/roc_curve_{model_name}.png",
        "total_samples": len(y_true),
    }

    logger.info(
        f"  {display_name}: Acc={acc:.2%}, Prec={prec:.2%}, "
        f"Rec={rec:.2%}, F1={f1:.2%}, AUC={auc_score:.4f}"
    )

    return metrics


def compare_models(num_samples: int = 200) -> dict:
    """
    Evaluate all available models and generate comparison.

    Returns:
        Dict with individual model metrics and a comparison plot path.
    """
    logger.info("Starting model comparison…")

    results = {}
    for name in model_manager.available_models():
        try:
            results[name] = evaluate_model(name, num_samples)
        except Exception as e:
            logger.error(f"Failed to evaluate {name}: {e}")

    if not results:
        return {"error": "No models could be evaluated."}

    # ── Comparison bar chart ────────────────────────────────────
    model_names = [r["display_name"] for r in results.values()]
    metrics_list = ["accuracy", "precision", "recall", "f1_score"]
    colors = ['#6366f1', '#a855f7', '#ec4899', '#22c55e']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.18

    for i, metric in enumerate(metrics_list):
        values = [results[m].get(metric, 0) for m in results]
        bars = ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title(),
                      color=colors[i], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8, color='#e2e8f0')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Comparison — All Metrics', fontsize=14, pad=15)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.1)
    plt.tight_layout()

    compare_path = os.path.join(EVAL_FOLDER, "model_comparison.png")
    plt.savefig(compare_path, dpi=120, facecolor='#0b0f19')
    plt.close()

    return {
        "models": results,
        "comparison_plot": "/static/evaluation/model_comparison.png",
    }


def main():
    """Run full evaluation from command line."""
    print("=" * 60)
    print("  Model Evaluation — Real vs AI Image Detector")
    print("=" * 60)

    result = compare_models(num_samples=100)

    if "error" in result:
        print(f"\n❌ {result['error']}")
        return

    print("\n" + "=" * 60)
    print("  Results Summary")
    print("=" * 60)

    for name, metrics in result["models"].items():
        print(f"\n  📊 {metrics['display_name']}:")
        print(f"     Accuracy:  {metrics['accuracy']}%")
        print(f"     Precision: {metrics['precision']}%")
        print(f"     Recall:    {metrics['recall']}%")
        print(f"     F1 Score:  {metrics['f1_score']}%")
        print(f"     AUC:       {metrics['auc_score']}")

    print(f"\n📈 Comparison plot: {result['comparison_plot']}")
    print("✅ All plots saved to static/evaluation/")


if __name__ == "__main__":
    main()
