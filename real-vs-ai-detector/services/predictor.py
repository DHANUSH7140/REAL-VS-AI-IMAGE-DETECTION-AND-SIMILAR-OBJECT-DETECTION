"""
services/predictor.py — Hybrid prediction combining neural networks + signal analysis.

Uses trained Keras models when they provide discriminative outputs,
combined with robust FFT + image statistics analysis for reliable predictions.
The multi-signal approach prevents mode collapse from any single source.
"""

import numpy as np
import cv2
from models.loader import model_manager
from utils.preprocessing import preprocess_image
from utils.logger import setup_logger
from config import MODEL_CONFIGS
from services.fft_features import extract_fft_features

logger = setup_logger("services.predictor")


# ──────────────────────────── SIGNAL ANALYSIS ───────────────────

def _compute_image_signals(image_path: str) -> dict:
    """
    Extract multiple statistical signals from the image that differ
    between real photographs and AI-generated images.

    Returns dict with signal scores (each 0-1, higher = more likely AI).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (256, 256))
        img_resized = cv2.resize(img, (256, 256))

        signals = {}

        # 1. Laplacian variance (texture sharpness)
        #    Real photos have rich micro-textures → high variance
        #    AI images tend to be smoother → low variance
        lap_var = float(cv2.Laplacian(gray_resized, cv2.CV_64F).var())
        if lap_var < 30:
            signals["texture"] = 0.90  # very smooth → likely AI
        elif lap_var < 80:
            signals["texture"] = 0.70
        elif lap_var < 200:
            signals["texture"] = 0.45
        elif lap_var < 500:
            signals["texture"] = 0.25
        else:
            signals["texture"] = 0.10  # very textured → likely Real
        signals["texture_raw"] = lap_var

        # 2. Edge density via Canny
        #    Real photos have complex natural edges
        #    AI images often have cleaner/smoother edges
        edges = cv2.Canny(gray_resized, 50, 150)
        edge_density = float(np.sum(edges > 0)) / edges.size
        if edge_density < 0.02:
            signals["edges"] = 0.80
        elif edge_density < 0.05:
            signals["edges"] = 0.55
        elif edge_density < 0.10:
            signals["edges"] = 0.35
        else:
            signals["edges"] = 0.15
        signals["edge_density_raw"] = edge_density

        # 3. Color saturation uniformity
        #    AI images often have unnaturally uniform saturation
        #    Real photos have varied saturation across regions
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(float)

        # Divide into 4x4 grid and check saturation variation across blocks
        block_sats = []
        bh, bw = 64, 64
        for r in range(4):
            for c in range(4):
                block = sat[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
                block_sats.append(float(block.std()))
        sat_variation = float(np.std(block_sats))

        if sat_variation < 5:
            signals["color"] = 0.80  # uniform sat → AI-like
        elif sat_variation < 15:
            signals["color"] = 0.50
        elif sat_variation < 30:
            signals["color"] = 0.30
        else:
            signals["color"] = 0.15  # highly varied → Real
        signals["sat_variation_raw"] = sat_variation

        # 4. DCT block artifact analysis
        #    Real JPEG photos have specific block artifacts at 8x8 boundaries
        #    AI images may lack these or have different patterns
        dct_blocks = []
        for r in range(0, 256 - 8, 8):
            for c in range(0, 256 - 8, 8):
                block = gray_resized[r:r + 8, c:c + 8].astype(np.float64)
                dct = cv2.dct(block)
                dct_blocks.append(float(np.abs(dct[1:, 1:]).mean()))
        dct_energy = float(np.mean(dct_blocks))

        if dct_energy < 2.0:
            signals["dct"] = 0.85   # very low DCT energy → AI smooth
        elif dct_energy < 5.0:
            signals["dct"] = 0.60
        elif dct_energy < 15.0:
            signals["dct"] = 0.35
        else:
            signals["dct"] = 0.12   # high DCT energy → Real photo
        signals["dct_energy_raw"] = dct_energy

        # 5. Local Binary Pattern uniformity
        #    Real photos have diverse LBP patterns
        #    AI images have more uniform/repetitive micro-patterns
        def _simple_lbp(img_gray):
            h, w = img_gray.shape
            lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    center = img_gray[i, j]
                    code = 0
                    code |= (img_gray[i-1, j-1] >= center) << 7
                    code |= (img_gray[i-1, j] >= center) << 6
                    code |= (img_gray[i-1, j+1] >= center) << 5
                    code |= (img_gray[i, j+1] >= center) << 4
                    code |= (img_gray[i+1, j+1] >= center) << 3
                    code |= (img_gray[i+1, j] >= center) << 2
                    code |= (img_gray[i+1, j-1] >= center) << 1
                    code |= (img_gray[i, j-1] >= center) << 0
                    lbp[i-1, j-1] = code
            return lbp

        # Use downscaled image for speed
        small_gray = cv2.resize(gray_resized, (64, 64))
        lbp = _simple_lbp(small_gray)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()
        lbp_entropy = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))

        # Max entropy for 256 bins = 8.0
        if lbp_entropy > 7.0:
            signals["lbp"] = 0.20  # high diversity → Real
        elif lbp_entropy > 6.0:
            signals["lbp"] = 0.35
        elif lbp_entropy > 5.0:
            signals["lbp"] = 0.55
        else:
            signals["lbp"] = 0.80  # low diversity → AI
        signals["lbp_entropy_raw"] = lbp_entropy

        return signals

    except Exception as e:
        logger.error(f"Signal analysis failed: {e}", exc_info=True)
        return {}


def _fft_to_score(fft_result: dict) -> float:
    """Convert FFT features into a 0-1 AI probability score."""
    if not fft_result or fft_result.get("error"):
        return 0.5

    noise_var = fft_result.get("noise_variance", 200)
    high_freq = fft_result.get("high_freq_ratio", 0.25)

    # Noise variance: low → AI, high → Real
    if noise_var < 20:
        n_score = 0.92
    elif noise_var < 50:
        n_score = 0.78
    elif noise_var < 100:
        n_score = 0.60
    elif noise_var < 300:
        n_score = 0.40
    elif noise_var < 600:
        n_score = 0.22
    else:
        n_score = 0.10

    # High frequency ratio: low → AI, high → Real
    if high_freq < 0.05:
        f_score = 0.88
    elif high_freq < 0.15:
        f_score = 0.65
    elif high_freq < 0.30:
        f_score = 0.40
    else:
        f_score = 0.15

    return 0.60 * n_score + 0.40 * f_score


# ──────────────────────────── NEURAL NETWORK ────────────────────

def _nn_predict(image_path: str, model_name: str) -> dict:
    """
    Run a single Keras model. Returns raw_score and whether it's discriminative.
    """
    try:
        model = model_manager.get(model_name)
        img_array = preprocess_image(image_path, model_name)
        pred = model.predict(img_array, verbose=0)
        raw = float(pred[0][0])

        # Check if the model is actually discriminating (not collapsed)
        # A collapsed model outputs ~0.0 or ~1.0 for everything
        is_discriminative = 0.01 < raw < 0.99

        return {
            "raw_score": raw,
            "is_discriminative": is_discriminative,
            "model_name": MODEL_CONFIGS[model_name].get("display_name", model_name),
        }
    except Exception as e:
        logger.warning(f"Neural network '{model_name}' failed: {e}")
        return {"raw_score": 0.5, "is_discriminative": False, "model_name": model_name}


# ──────────────────────────── MAIN PREDICTOR ────────────────────

def predict_single(image_path: str, model_name: str) -> dict:
    """
    Run hybrid prediction combining neural network + multi-signal analysis.

    The neural network result is used if discriminative.
    Signal analysis (FFT + texture + edges + color + DCT + LBP) always
    contributes for robustness.

    Args:
        image_path: Absolute path to the image file.
        model_name: 'resnet', 'efficientnet', or 'ensemble'.

    Returns:
        Dict with label, confidence, raw_score, model_name, evidence.
    """
    if model_name not in MODEL_CONFIGS:
        return {"error": f"Unknown model '{model_name}'. Available: {list(MODEL_CONFIGS.keys())}"}

    logger.info(f"Hybrid prediction: model={model_name}, image={image_path}")

    # 1. Run neural network
    nn_result = _nn_predict(image_path, model_name)
    nn_score = nn_result["raw_score"]
    nn_useful = nn_result["is_discriminative"]

    # 2. Run FFT analysis
    fft_result = extract_fft_features(image_path)
    fft_score = _fft_to_score(fft_result)

    # 3. Run image signal analysis
    signals = _compute_image_signals(image_path)
    signal_scores = [
        v for k, v in signals.items()
        if not k.endswith("_raw") and isinstance(v, float)
    ]

    if signal_scores:
        # Weighted average of signal scores
        weights = {
            "texture": 0.25,
            "edges": 0.15,
            "color": 0.15,
            "dct": 0.25,
            "lbp": 0.20,
        }
        signal_avg = 0.0
        total_w = 0.0
        for key, w in weights.items():
            if key in signals:
                signal_avg += signals[key] * w
                total_w += w
        signal_avg = signal_avg / total_w if total_w > 0 else 0.5
    else:
        signal_avg = 0.5

    # 4. Combine all signals
    if nn_useful:
        # Neural network is working — trust it more
        # NN: 50%, FFT: 20%, Image signals: 30%
        final_score = (nn_score * 0.50 + fft_score * 0.20 + signal_avg * 0.30)
        method = f"{nn_result['model_name']} + FFT + Signal Analysis"
    else:
        # Neural network collapsed — rely on FFT + signals
        # FFT: 40%, Image signals: 60%
        final_score = (fft_score * 0.40 + signal_avg * 0.60)
        method = "FFT + Signal Analysis (NN bypassed)"
        logger.warning(
            f"Neural network '{model_name}' appears collapsed "
            f"(raw={nn_score:.6f}). Using signal analysis only."
        )

    # 5. Determine label and confidence
    if final_score >= 0.5:
        label = "AI Generated"
        confidence = round(final_score * 100, 2)
    else:
        label = "Real Image"
        confidence = round((1.0 - final_score) * 100, 2)

    # 6. Build evidence
    evidence = []
    if signals.get("texture_raw") is not None:
        tex = signals["texture_raw"]
        if tex < 80:
            evidence.append(f"Low texture variance ({tex:.1f}) — smoother than typical photos")
        elif tex > 300:
            evidence.append(f"Rich micro-textures ({tex:.1f}) — consistent with real cameras")

    if fft_result and not fft_result.get("error"):
        nv = fft_result.get("noise_variance", 0)
        if nv < 50:
            evidence.append(f"Very low sensor noise ({nv:.1f}) — too clean for real camera")
        elif nv > 300:
            evidence.append(f"High sensor noise ({nv:.1f}) — consistent with physical sensor")

    if signals.get("dct_energy_raw") is not None:
        dct = signals["dct_energy_raw"]
        if dct < 3:
            evidence.append(f"Low DCT energy ({dct:.1f}) — lacking natural compression artifacts")
        elif dct > 10:
            evidence.append(f"Normal DCT patterns ({dct:.1f}) — real JPEG characteristics")

    if signals.get("lbp_entropy_raw") is not None:
        ent = signals["lbp_entropy_raw"]
        if ent < 5.5:
            evidence.append(f"Low texture diversity (entropy={ent:.2f}) — repetitive patterns")
        elif ent > 6.5:
            evidence.append(f"High texture diversity (entropy={ent:.2f}) — natural complexity")

    if not evidence:
        evidence.append("Inconclusive — signals suggest borderline result")

    result = {
        "label": label,
        "confidence": confidence,
        "raw_score": round(final_score, 4),
        "model_name": method,
        "nn_used": nn_useful,
        "nn_raw": round(nn_score, 6),
        "fft_score": round(fft_score, 4),
        "signal_score": round(signal_avg, 4),
        "key_evidence": evidence,
        "signals": {k: v for k, v in signals.items() if k.endswith("_raw")},
    }

    logger.info(
        f"Hybrid result: {label} ({confidence}%) "
        f"[nn={'active' if nn_useful else 'collapsed'}={nn_score:.4f}, "
        f"fft={fft_score:.4f}, signals={signal_avg:.4f} → final={final_score:.4f}]"
    )
    return result


def predict_all(image_path: str) -> dict:
    """
    Run prediction with ALL available models.

    Returns:
        Dict mapping model_name → result dict.
    """
    results = {}

    for model_name in MODEL_CONFIGS:
        try:
            res = predict_single(image_path, model_name)
            results[model_name] = res
        except Exception as e:
            logger.error(f"Error predicting with '{model_name}': {e}")
            results[model_name] = {
                "error": str(e),
                "model_name": MODEL_CONFIGS[model_name].get("display_name", model_name),
            }

    return results
