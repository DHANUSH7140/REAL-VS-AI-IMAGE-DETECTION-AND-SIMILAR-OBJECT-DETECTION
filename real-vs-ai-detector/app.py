"""
app.py — Main entry point for the Real vs AI Image Detector.

Production-ready Flask application with modular architecture,
multi-model inference, Grad-CAM explainability, FFT analysis,
and ensemble predictions.

Usage:
    python app.py
    → Opens at http://127.0.0.1:5000
"""

import os
from flask import Flask

from config import SECRET_KEY, FLASK_DEBUG, MAX_CONTENT_LENGTH
from api.routes import main_bp
from utils.logger import setup_logger

logger = setup_logger("app")


def create_app() -> Flask:
    """
    Application factory — creates and configures the Flask app.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)

    # ── Flask config ────────────────────────────────────────────
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    # ── Register blueprints ─────────────────────────────────────
    app.register_blueprint(main_bp)

    logger.info("Flask application initialized.")
    logger.info(f"Debug mode: {FLASK_DEBUG}")

    return app


# ──────────────────────────── MAIN ──────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 56)
    print("  🔍 Real vs AI Image Detector — Production Server")
    print("=" * 56)
    print()
    print("  Features:")
    print("  ├── Multi-model inference (ResNet50, EfficientNet)")
    print("  ├── Ensemble predictions (weighted voting)")
    print("  ├── Grad-CAM explainability heatmaps")
    print("  ├── Dual AI/Real explanation engine")
    print("  ├── FFT frequency analysis")
    print("  ├── REST API (/api/predict)")
    print("  └── Upload history tracking")
    print()
    print("  🌐 Open http://127.0.0.1:5000 in your browser")
    print()

    app = create_app()
    app.run(debug=FLASK_DEBUG, host="0.0.0.0", port=5000)
