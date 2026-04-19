---
title: TRUESIGHT
emoji: 👁️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 5000
---
# 🔍 Real vs AI Image Detector — Production System

A **production-grade, explainable, multi-model deep learning system** that detects AI-generated images vs real photographs. Features ensemble voting, Grad-CAM heatmaps, FFT frequency analysis, and a premium web UI.

---

## 🏗️ Architecture

```
real-vs-ai-detector/
├── app.py                      # Flask entry point (application factory)
├── config.py                   # Centralized configuration
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
│
├── models/                     # Model management
│   └── loader.py               # Lazy loader with caching (ModelManager)
│
├── services/                   # Core business logic
│   ├── predictor.py            # Single & multi-model inference
│   ├── ensemble.py             # Average & weighted voting
│   ├── gradcam.py              # Grad-CAM heatmap generation
│   └── fft_features.py         # FFT frequency analysis
│
├── api/                        # Flask routes
│   └── routes.py               # /predict, /api/predict, /api/history
│
├── utils/                      # Shared utilities
│   ├── preprocessing.py        # Per-model image preprocessing
│   └── logger.py               # Structured logging (file + console)
│
├── evaluation/                 # Model evaluation & reporting
│   └── evaluate.py             # Metrics, confusion matrix, ROC curves
│
├── training/                   # Original training scripts
│   ├── train_cnn.py
│   ├── train_resnet.py
│   └── train_efficientnet.py
│
├── static/
│   ├── style.css               # Premium dark glassmorphism UI
│   ├── uploads/                # User-uploaded images
│   ├── gradcam/                # Generated heatmap overlays
│   ├── fft/                    # FFT spectrum visualizations
│   └── evaluation/             # Evaluation plots
│
├── templates/
│   └── index.html              # Advanced web UI
│
├── model/                      # Trained models
│   └── efficientnet_trained.h5
├── model_cnn.h5
├── model_resnet.h5
└── logs/
    └── app.log
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the server

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 🧠 Models

| Model | Architecture | Input Size | Weights |
|-------|-------------|------------|---------|
| Custom CNN | Conv2D 32→64→128→256, BN, Dropout | 128×128 | `model_cnn.h5` |
| ResNet50 | ImageNet pretrained + custom head | 224×224 | `model_resnet.h5` |
| EfficientNetB4 | ImageNet pretrained + custom head | 224×224 | `model/efficientnet_trained.h5` |

---

## 🎯 Ensemble Prediction

Combines all three models using **weighted voting**:

| Model | Weight |
|-------|--------|
| EfficientNet | 0.45 |
| ResNet50 | 0.35 |
| Custom CNN | 0.20 |

---

## 🔥 Grad-CAM Explainability

Generates heatmap overlays showing which image regions each model focused on:
- **ResNet50**: Uses `conv5_block3_out` layer
- **EfficientNet**: Uses `top_conv` layer
- **CNN**: Auto-detects last Conv2D layer

---

## 📡 FFT Frequency Analysis

Extracts frequency-domain features to detect subtle AI artifacts:
- **High-frequency ratio** — AI images often have less high-frequency detail
- **Spectral centroid** — Center of mass of the frequency spectrum
- **Noise variance** — AI images tend to be "too clean"

---

## 🔌 REST API

### `POST /api/predict`

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict?model=ensemble
```

**Response:**
```json
{
  "prediction": "AI",
  "confidence": 94.5,
  "model": "ensemble",
  "gradcam_image": "/static/gradcam/gradcam_abc123.png",
  "fft_analysis": {
    "high_freq_ratio": 0.12,
    "spectral_centroid": 45.3,
    "noise_variance": 32.1,
    "spectrum_url": "/static/fft/fft_def456.png",
    "analysis_summary": "Low high-frequency content..."
  }
}
```

### `GET /api/history`

Returns recent prediction history as JSON.

---

## 📊 Model Evaluation

Run comprehensive evaluation with metrics and plots:

```bash
python -m evaluation.evaluate
```

Generates:
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix plots
- ROC Curves with AUC
- Model comparison bar charts

---

## 🖥️ UI Features

- **Drag & drop** image upload
- **Model selector** — CNN, ResNet50, EfficientNet, or Ensemble
- **Confidence gauge** — animated circular indicator
- **Grad-CAM heatmap** — side-by-side with original
- **FFT spectrum** — frequency analysis visualization
- **Individual model scores** — when using ensemble
- **Upload history** — sidebar tracking recent predictions

---

## ⚠️ Error Handling

- Invalid file format → JSON error with allowed formats
- Model not found → clear message to train first
- Server errors → logged to `logs/app.log`
- All routes have try/catch with structured error responses

---

## 📝 Training

To retrain models, use the scripts in `training/`:

```bash
python training/train_cnn.py
python training/train_resnet.py
python training/train_efficientnet.py
```

Expects `dataset_v2/real/` and `dataset_v2/ai/` folders with images.
