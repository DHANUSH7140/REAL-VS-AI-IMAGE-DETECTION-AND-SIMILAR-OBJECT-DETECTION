"""
services/explanation_engine.py — Explainable AI reasoning engine.

Generates human-readable explanations for Real vs AI predictions
by combining model scores, FFT analysis, and image pattern analysis.
Provides dual explanations with different reasoning for AI vs Real.
"""

import cv2
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("services.explanation_engine")


# ──────────────────────────── PATTERN THRESHOLDS ────────────────
_TEXTURE_SMOOTH_THRESH = 80       # Laplacian var below → "too smooth"
_TEXTURE_NATURAL_THRESH = 300     # Laplacian var above → "natural micro-textures"
_NOISE_LOW_THRESH = 50            # noise_variance below → "AI-clean"
_NOISE_HIGH_THRESH = 500          # noise_variance above → "real sensor noise"
_FREQ_LOW_THRESH = 0.15           # high_freq_ratio below → "AI smoothing"
_FREQ_HIGH_THRESH = 0.35          # high_freq_ratio above → "rich detail"
_EDGE_LOW_THRESH = 0.03           # edge density below → "too smooth"
_EDGE_HIGH_THRESH = 0.12          # edge density above → "rich edge detail"


def _analyze_texture(image_path: str) -> dict:
    """Compute texture smoothness via Laplacian variance."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"score": 0.5, "variance": 0}
        img = cv2.resize(img, (256, 256))
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = float(laplacian.var())
        # Normalize to 0-1 (higher = more texture)
        score = min(variance / 800.0, 1.0)
        return {"score": round(score, 3), "variance": round(variance, 2)}
    except Exception as e:
        logger.warning(f"Texture analysis failed: {e}")
        return {"score": 0.5, "variance": 0}


def _analyze_edges(image_path: str) -> dict:
    """Compute edge density using Canny edge detection."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"score": 0.5, "density": 0}
        img = cv2.resize(img, (256, 256))
        edges = cv2.Canny(img, 50, 150)
        density = float(np.sum(edges > 0)) / edges.size
        score = min(density / 0.20, 1.0)
        return {"score": round(score, 3), "density": round(density, 4)}
    except Exception as e:
        logger.warning(f"Edge analysis failed: {e}")
        return {"score": 0.5, "density": 0}


def _analyze_color_consistency(image_path: str) -> dict:
    """Analyze color channel consistency and saturation distribution."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"score": 0.5, "std_dev": 0}
        img = cv2.resize(img, (256, 256))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat_std = float(hsv[:, :, 1].std())
        val_std = float(hsv[:, :, 2].std())
        combined = (sat_std + val_std) / 2.0
        score = min(combined / 80.0, 1.0)
        return {"score": round(score, 3), "std_dev": round(combined, 2)}
    except Exception as e:
        logger.warning(f"Color analysis failed: {e}")
        return {"score": 0.5, "std_dev": 0}


def _build_pattern_indicators(texture, edges, fft_result, color) -> list:
    """
    Build 4 pattern indicator cards with icon, score, and interpretation.

    Returns:
        List of dicts: { name, icon, score, interpretation, detail }
    """
    indicators = []

    # ── TEXTURE ──
    tex_score = texture["score"]
    tex_var = texture["variance"]
    if tex_var < _TEXTURE_SMOOTH_THRESH:
        tex_interp = "Unusually smooth — synthetic texture pattern"
    elif tex_var > _TEXTURE_NATURAL_THRESH:
        tex_interp = "Rich micro-textures — consistent with real photos"
    else:
        tex_interp = "Moderate texture detail — inconclusive"
    indicators.append({
        "name": "Texture",
        "icon": "🧶",
        "score": tex_score,
        "interpretation": tex_interp,
        "detail": f"Laplacian variance: {tex_var}",
    })

    # ── NOISE ──
    noise_var = fft_result.get("noise_variance", 0) if fft_result else 0
    noise_score = min(noise_var / 800.0, 1.0)
    if noise_var < _NOISE_LOW_THRESH:
        noise_interp = "Very low sensor noise — too clean for a real camera"
    elif noise_var > _NOISE_HIGH_THRESH:
        noise_interp = "High noise variance — real camera sensor pattern"
    else:
        noise_interp = "Normal noise levels"
    indicators.append({
        "name": "Noise",
        "icon": "📡",
        "score": round(noise_score, 3),
        "interpretation": noise_interp,
        "detail": f"Noise variance: {noise_var:.1f}",
    })

    # ── EDGES ──
    edge_score = edges["score"]
    edge_density = edges["density"]
    if edge_density < _EDGE_LOW_THRESH:
        edge_interp = "Low edge density — possible AI smoothing"
    elif edge_density > _EDGE_HIGH_THRESH:
        edge_interp = "Rich edge detail — matches natural photography"
    else:
        edge_interp = "Normal edge patterns"
    indicators.append({
        "name": "Edges",
        "icon": "📐",
        "score": edge_score,
        "interpretation": edge_interp,
        "detail": f"Edge density: {edge_density:.4f}",
    })

    # ── FREQUENCY ──
    hfr = fft_result.get("high_freq_ratio", 0) if fft_result else 0
    freq_score = min(hfr / 0.50, 1.0)
    if hfr < _FREQ_LOW_THRESH:
        freq_interp = "Low high-frequency content — may indicate AI smoothing"
    elif hfr > _FREQ_HIGH_THRESH:
        freq_interp = "Rich high-frequency detail — consistent with real photos"
    else:
        freq_interp = "Moderate frequency distribution"
    indicators.append({
        "name": "Frequency",
        "icon": "📊",
        "score": round(freq_score, 3),
        "interpretation": freq_interp,
        "detail": f"High-freq ratio: {hfr:.4f}",
    })

    return indicators


def _build_key_factors(prediction, indicators, fft_result) -> list:
    """Extract the top 3-4 factors that most influenced the decision."""
    factors = []
    is_ai = prediction.get("label") == "AI Generated"

    for ind in indicators:
        if is_ai and ind["score"] < 0.3:
            factors.append(f"Low {ind['name'].lower()}")
        elif not is_ai and ind["score"] > 0.6:
            factors.append(f"Strong {ind['name'].lower()}")

    # Add confidence-based factor
    conf = prediction.get("confidence", 0)
    if conf > 90:
        factors.append("Very high model confidence")
    elif conf > 75:
        factors.append("High model confidence")

    # Add FFT-specific factors
    if fft_result and not fft_result.get("error"):
        hfr = fft_result.get("high_freq_ratio", 0)
        if is_ai and hfr < _FREQ_LOW_THRESH:
            factors.append("Reduced high-frequency content")
        elif not is_ai and hfr > _FREQ_HIGH_THRESH:
            factors.append("Rich frequency spectrum")

    # Deduplicate and limit
    seen = set()
    unique = []
    for f in factors:
        if f.lower() not in seen:
            seen.add(f.lower())
            unique.append(f)
    return unique[:5]


def _build_explanation_text(prediction, indicators, fft_result) -> str:
    """Generate a 2-3 sentence human-readable explanation."""
    label = prediction.get("label", "Unknown")
    conf = prediction.get("confidence", 0)
    is_ai = label == "AI Generated"

    tex = next((i for i in indicators if i["name"] == "Texture"), None)
    noise = next((i for i in indicators if i["name"] == "Noise"), None)
    freq = next((i for i in indicators if i["name"] == "Frequency"), None)

    if is_ai:
        # AI explanation
        reasons = []
        if tex and tex["score"] < 0.3:
            reasons.append("synthetic smoothness in texture patterns")
        if noise and noise["score"] < 0.2:
            reasons.append("abnormally low sensor noise (too clean)")
        if freq and freq["score"] < 0.3:
            reasons.append("reduced high-frequency detail")

        if not reasons:
            reasons.append("patterns consistent with AI generation")

        reason_str = ", ".join(reasons[:2])
        text = (
            f"This image was classified as AI Generated with {conf}% confidence. "
            f"The model detected {reason_str}. "
            f"AI-generated images typically lack the natural micro-textures, "
            f"sensor noise, and fine-grained detail found in real photographs."
        )
    else:
        # Real explanation
        reasons = []
        if tex and tex["score"] > 0.5:
            reasons.append("natural micro-texture patterns")
        if noise and noise["score"] > 0.4:
            reasons.append("realistic camera sensor noise")
        if freq and freq["score"] > 0.5:
            reasons.append("rich high-frequency detail")

        if not reasons:
            reasons.append("patterns consistent with real photography")

        reason_str = ", ".join(reasons[:2])
        text = (
            f"This image was classified as a Real Image with {conf}% confidence. "
            f"The analysis found {reason_str}. "
            f"Real photographs contain natural imperfections, sensor noise, "
            f"and complex texture variations that are difficult to replicate synthetically."
        )

    return text


def _build_verdict_reasoning(prediction, indicators) -> str:
    """Build a concise verdict reasoning string."""
    is_ai = prediction.get("label") == "AI Generated"

    if is_ai:
        weak = [i for i in indicators if i["score"] < 0.35]
        if weak:
            names = ", ".join([i["name"].lower() for i in weak[:3]])
            return (
                f"The model's decision was primarily driven by low {names} scores, "
                f"which are characteristic of AI-generated content. "
                f"These patterns suggest the image was produced by a generative model."
            )
        return "The model detected patterns consistent with AI-generated imagery."
    else:
        strong = [i for i in indicators if i["score"] > 0.5]
        if strong:
            names = ", ".join([i["name"].lower() for i in strong[:3]])
            return (
                f"The model's decision was supported by strong {names} indicators, "
                f"which are characteristic of real photographs. "
                f"These natural patterns are difficult for generative models to replicate."
            )
        return "The model detected patterns consistent with authentic photography."


def generate_explanation(prediction: dict, fft_result: dict,
                         image_path: str) -> dict:
    """
    Generate a comprehensive explanation for a prediction.

    Args:
        prediction: Dict with 'label', 'confidence', 'raw_score'.
        fft_result:  Dict from extract_fft_features() (may be None or have 'error').
        image_path:  Absolute path to the image file.

    Returns:
        Dict with:
            - text: Human-readable explanation (2-3 sentences)
            - confidence_distribution: { ai, real } probabilities
            - pattern_indicators: List of 4 indicator dicts
            - key_factors: List of top factor strings
            - verdict_reasoning: Concise verdict explanation
    """
    logger.info(f"Generating explanation for: {prediction.get('label')}")

    # Clean FFT result (ignore errors)
    if fft_result and fft_result.get("error"):
        fft_result = None

    # ── Analyze image patterns ──
    texture = _analyze_texture(image_path)
    edges = _analyze_edges(image_path)
    color = _analyze_color_consistency(image_path)

    # ── Build explanation components ──
    raw_score = prediction.get("raw_score", 0.5)
    confidence_distribution = {
        "ai": round(raw_score, 4),
        "real": round(1.0 - raw_score, 4),
    }

    indicators = _build_pattern_indicators(texture, edges, fft_result, color)
    key_factors = _build_key_factors(prediction, indicators, fft_result)
    explanation_text = _build_explanation_text(prediction, indicators, fft_result)
    verdict = _build_verdict_reasoning(prediction, indicators)

    result = {
        "text": explanation_text,
        "confidence_distribution": confidence_distribution,
        "pattern_indicators": indicators,
        "key_factors": key_factors,
        "verdict_reasoning": verdict,
    }

    logger.info(f"Explanation generated: {len(indicators)} indicators, "
                f"{len(key_factors)} key factors")
    return result
