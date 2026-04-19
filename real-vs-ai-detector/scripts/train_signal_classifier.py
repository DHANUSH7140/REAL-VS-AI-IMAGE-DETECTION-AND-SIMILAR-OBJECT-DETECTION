"""
scripts/train_signal_classifier.py — Train a lightweight classifier on image signal features.

Extracts texture, edge, color, DCT, LBP, and FFT features from the dataset,
then trains a Random Forest + Logistic Regression ensemble.
Saves the trained model to model/meta/signal_classifier.pkl
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import glob
import numpy as np
import pickle
import cv2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from services.fft_features import extract_fft_features


def extract_signals(image_path):
    """Extract all signal features from a single image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (256, 256))
        img_resized = cv2.resize(img, (256, 256))

        features = []

        # 1. Laplacian variance
        lap_var = float(cv2.Laplacian(gray_resized, cv2.CV_64F).var())
        features.append(lap_var)

        # 2. Edge density
        edges = cv2.Canny(gray_resized, 50, 150)
        edge_density = float(np.sum(edges > 0)) / edges.size
        features.append(edge_density)

        # 3. Saturation variation across blocks
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(float)
        block_sats = []
        for r in range(4):
            for c in range(4):
                block = sat[r*64:(r+1)*64, c*64:(c+1)*64]
                block_sats.append(float(block.std()))
        sat_variation = float(np.std(block_sats))
        features.append(sat_variation)

        # 4. DCT energy
        dct_blocks = []
        for r in range(0, 248, 8):
            for c in range(0, 248, 8):
                block = gray_resized[r:r+8, c:c+8].astype(np.float64)
                dct = cv2.dct(block)
                dct_blocks.append(float(np.abs(dct[1:, 1:]).mean()))
        dct_energy = float(np.mean(dct_blocks))
        features.append(dct_energy)

        # 5. LBP entropy (simplified on small image)
        small = cv2.resize(gray_resized, (64, 64))
        h, w = small.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = small[i, j]
                code = 0
                code |= (small[i-1, j-1] >= center) << 7
                code |= (small[i-1, j] >= center) << 6
                code |= (small[i-1, j+1] >= center) << 5
                code |= (small[i, j+1] >= center) << 4
                code |= (small[i+1, j+1] >= center) << 3
                code |= (small[i+1, j] >= center) << 2
                code |= (small[i+1, j-1] >= center) << 1
                code |= (small[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float) / (hist.sum() + 1e-10)
        lbp_entropy = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))
        features.append(lbp_entropy)

        # 6. FFT features
        fft = extract_fft_features(image_path)
        features.append(fft.get("noise_variance", 0))
        features.append(fft.get("high_freq_ratio", 0))
        features.append(fft.get("spectral_centroid", 0))

        # 7. Color statistics
        b, g, r_ch = cv2.split(img_resized)
        features.append(float(b.std()))
        features.append(float(g.std()))
        features.append(float(r_ch.std()))

        # 8. Mean saturation and value
        features.append(float(hsv[:, :, 1].mean()))
        features.append(float(hsv[:, :, 2].mean()))

        # 9. Gradient magnitude stats
        gx = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features.append(float(grad_mag.mean()))
        features.append(float(grad_mag.std()))

        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f"  Error extracting {image_path}: {e}")
        return None


FEATURE_NAMES = [
    "laplacian_var", "edge_density", "sat_variation",
    "dct_energy", "lbp_entropy",
    "fft_noise_var", "fft_high_freq", "fft_spectral_centroid",
    "blue_std", "green_std", "red_std",
    "mean_saturation", "mean_value",
    "gradient_mean", "gradient_std",
]


def main():
    print("=" * 60)
    print("  Signal Classifier Training")
    print("=" * 60)

    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset_v2")
    real_dir = os.path.join(dataset_dir, "real")
    ai_dir = os.path.join(dataset_dir, "ai")

    real_imgs = sorted(glob.glob(os.path.join(real_dir, "*.jpg")))
    ai_imgs = sorted(glob.glob(os.path.join(ai_dir, "*.jpg")))

    print(f"\nDataset: {len(real_imgs)} real, {len(ai_imgs)} AI images")

    # Limit to speed up training (use all if < 500 per class)
    max_per_class = 500
    real_imgs = real_imgs[:max_per_class]
    ai_imgs = ai_imgs[:max_per_class]

    print(f"Using: {len(real_imgs)} real, {len(ai_imgs)} AI images")

    # Extract features
    print("\nExtracting features...")
    X, y = [], []

    for i, path in enumerate(real_imgs):
        feat = extract_signals(path)
        if feat is not None:
            X.append(feat)
            y.append(0)  # 0 = Real
        if (i + 1) % 50 == 0:
            print(f"  Real: {i+1}/{len(real_imgs)}")

    for i, path in enumerate(ai_imgs):
        feat = extract_signals(path)
        if feat is not None:
            X.append(feat)
            y.append(1)  # 1 = AI
        if (i + 1) % 50 == 0:
            print(f"  AI: {i+1}/{len(ai_imgs)}")

    X = np.array(X)
    y = np.array(y)
    print(f"\nFeature matrix: {X.shape}")
    print(f"Class distribution: Real={np.sum(y==0)}, AI={np.sum(y==1)}")

    # Print feature statistics
    print(f"\n{'Feature':<25} {'Real mean':>10} {'AI mean':>10} {'Diff':>10}")
    print("-" * 60)
    for i, name in enumerate(FEATURE_NAMES):
        real_mean = X[y == 0, i].mean()
        ai_mean = X[y == 1, i].mean()
        diff = ai_mean - real_mean
        print(f"{name:<25} {real_mean:>10.3f} {ai_mean:>10.3f} {diff:>+10.3f}")

    # Train classifier
    print("\n\nTraining classifiers...")

    # Gradient Boosting (best for tabular data)
    gb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        ))
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(gb_pipe, X, y, cv=cv, scoring="accuracy")
    print(f"\n  Gradient Boosting CV accuracy: {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")

    # Also try Random Forest
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
        ))
    ])
    scores_rf = cross_val_score(rf_pipe, X, y, cv=cv, scoring="accuracy")
    print(f"  Random Forest CV accuracy:     {scores_rf.mean()*100:.1f}% (+/- {scores_rf.std()*100:.1f}%)")

    # Logistic Regression
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ])
    scores_lr = cross_val_score(lr_pipe, X, y, cv=cv, scoring="accuracy")
    print(f"  Logistic Regression CV accuracy: {scores_lr.mean()*100:.1f}% (+/- {scores_lr.std()*100:.1f}%)")

    # Pick best and train on full data
    best_score = max(scores.mean(), scores_rf.mean(), scores_lr.mean())
    if scores.mean() == best_score:
        best_pipe = gb_pipe
        best_name = "GradientBoosting"
    elif scores_rf.mean() == best_score:
        best_pipe = rf_pipe
        best_name = "RandomForest"
    else:
        best_pipe = lr_pipe
        best_name = "LogisticRegression"

    print(f"\n  Best model: {best_name} ({best_score*100:.1f}%)")

    # Train on full dataset
    best_pipe.fit(X, y)

    # Final evaluation
    y_pred = best_pipe.predict(X)
    print(f"\n  Training accuracy: {(y_pred == y).mean()*100:.1f}%")
    print("\n" + classification_report(y, y_pred, target_names=["Real", "AI"]))

    cm = confusion_matrix(y, y_pred)
    print(f"  Confusion Matrix:")
    print(f"    Real -> Real: {cm[0][0]}  Real -> AI: {cm[0][1]}")
    print(f"    AI -> Real:   {cm[1][0]}  AI -> AI:   {cm[1][1]}")

    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), "..", "model", "meta")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "signal_classifier.pkl")

    model_data = {
        "model": best_pipe,
        "feature_names": FEATURE_NAMES,
        "model_name": best_name,
        "cv_accuracy": best_score,
        "n_features": len(FEATURE_NAMES),
    }

    with open(save_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\n  Model saved → {save_path}")
    print(f"  Features: {len(FEATURE_NAMES)}")
    print(f"  CV Accuracy: {best_score*100:.1f}%")
    print("\nDone!")


if __name__ == "__main__":
    main()
