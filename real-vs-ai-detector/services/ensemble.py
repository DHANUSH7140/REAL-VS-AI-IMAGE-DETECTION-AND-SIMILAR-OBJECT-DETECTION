"""
services/ensemble.py — Ensemble voting strategies.

Combines predictions from multiple models (ResNet50, EfficientNet)
and optionally FFT frequency analysis using weighted voting.
"""

from config import ENSEMBLE_WEIGHTS
from utils.logger import setup_logger

logger = setup_logger("services.ensemble")


def average_voting(predictions: dict) -> dict:
    """
    Combine model predictions using simple average voting.

    Args:
        predictions: Dict mapping model_name → result dict (must have 'raw_score').

    Returns:
        Ensemble result dict with label, confidence, method, individual_results.
    """
    valid = {k: v for k, v in predictions.items() if "raw_score" in v}

    if not valid:
        return {"error": "No valid predictions to ensemble."}

    scores = [v["raw_score"] for v in valid.values()]
    avg_score = sum(scores) / len(scores)

    if avg_score >= 0.5:
        label = "AI Generated"
        confidence = round(avg_score * 100, 2)
    else:
        label = "Real Image"
        confidence = round((1 - avg_score) * 100, 2)

    result = {
        "label": label,
        "confidence": confidence,
        "raw_score": avg_score,
        "model_name": "ensemble",
        "method": "average_voting",
        "individual_results": valid,
    }

    logger.info(f"Average ensemble: {label} ({confidence}%) from {len(valid)} models")
    return result


def fft_to_score(fft_result: dict) -> float:
    """
    Convert FFT analysis results into a 0-1 AI probability score.

    Uses noise variance and high-frequency ratio as signals:
    - Low noise + low high-freq → more likely AI (score closer to 1.0)
    - High noise + high high-freq → more likely Real (score closer to 0.0)

    Args:
        fft_result: Dict from extract_fft_features() with noise_variance,
                    high_freq_ratio, spectral_centroid.

    Returns:
        Float between 0 (Real) and 1 (AI).
    """
    if not fft_result or fft_result.get("error"):
        return 0.5  # neutral on error

    noise_var = fft_result.get("noise_variance", 200)
    high_freq = fft_result.get("high_freq_ratio", 0.25)

    # Noise variance signal: low noise → AI, high noise → Real
    # Map: 0-50 → strong AI (0.9), 50-200 → moderate, 200-800+ → strong Real (0.1)
    if noise_var < 50:
        noise_signal = 0.90
    elif noise_var < 100:
        noise_signal = 0.70
    elif noise_var < 200:
        noise_signal = 0.50
    elif noise_var < 500:
        noise_signal = 0.30
    else:
        noise_signal = 0.10

    # High-frequency ratio signal: low HF → AI, high HF → Real
    # Map: 0-0.10 → strong AI, 0.10-0.25 → moderate, 0.25-0.50+ → strong Real
    if high_freq < 0.10:
        freq_signal = 0.85
    elif high_freq < 0.20:
        freq_signal = 0.60
    elif high_freq < 0.35:
        freq_signal = 0.40
    else:
        freq_signal = 0.15

    # Combine: noise is more discriminative than frequency ratio
    score = 0.65 * noise_signal + 0.35 * freq_signal

    logger.debug(f"FFT→Score: noise_var={noise_var:.1f}→{noise_signal:.2f}, "
                 f"high_freq={high_freq:.4f}→{freq_signal:.2f}, "
                 f"combined={score:.3f}")
    return round(score, 4)


def weighted_voting(predictions: dict, weights: dict = None,
                    fft_result: dict = None) -> dict:
    """
    Combine model predictions using weighted voting, optionally including FFT.

    Args:
        predictions: Dict mapping model_name → result dict (must have 'raw_score').
        weights:     Dict mapping model_name → weight (default: config ENSEMBLE_WEIGHTS).
        fft_result:  Optional FFT analysis result dict to include as a voter.

    Returns:
        Ensemble result dict with label, confidence, method, individual_results.
    """
    if weights is None:
        weights = ENSEMBLE_WEIGHTS

    valid = {k: v for k, v in predictions.items() if "raw_score" in v}

    if not valid:
        return {"error": "No valid predictions to ensemble."}

    # Include FFT as a voter if provided
    if fft_result and not fft_result.get("error"):
        fft_score = fft_to_score(fft_result)
        valid["fft"] = {
            "raw_score": fft_score,
            "label": "AI Generated" if fft_score >= 0.5 else "Real Image",
            "confidence": round(max(fft_score, 1.0 - fft_score) * 100, 2),
            "model_name": "FFT Analysis",
        }

    # Compute weighted average
    total_weight = 0.0
    weighted_sum = 0.0

    for name, result in valid.items():
        w = weights.get(name, 1.0 / len(valid))  # fallback to equal weight
        weighted_sum += result["raw_score"] * w
        total_weight += w

    avg_score = weighted_sum / total_weight if total_weight > 0 else 0.5

    if avg_score >= 0.5:
        label = "AI Generated"
        confidence = round(avg_score * 100, 2)
    else:
        label = "Real Image"
        confidence = round((1 - avg_score) * 100, 2)

    result = {
        "label": label,
        "confidence": confidence,
        "raw_score": avg_score,
        "model_name": "ensemble",
        "method": "weighted_voting",
        "weights_used": {k: weights.get(k, 1.0 / len(valid)) for k in valid},
        "individual_results": valid,
    }

    # Log individual model contributions
    for name, pred in valid.items():
        w = weights.get(name, 1.0 / len(valid))
        logger.info(
            f"  → {name}: {pred.get('label', '?')} "
            f"({pred.get('confidence', 0)}%) weight={w:.2f}"
        )

    logger.info(
        f"Weighted ensemble: {label} ({confidence}%) "
        f"from {len(valid)} signals [score={avg_score:.4f}]"
    )
    return result
