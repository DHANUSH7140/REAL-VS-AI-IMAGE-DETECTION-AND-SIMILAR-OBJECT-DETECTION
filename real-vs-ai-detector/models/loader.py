"""
models/loader.py — Lazy model loader with in-memory caching.

Loads Keras .h5 models on first request and caches them.
Thread-safe singleton pattern for production use.
"""

import os
import threading
from tensorflow.keras.models import load_model

from config import MODEL_CONFIGS
from utils.logger import setup_logger

logger = setup_logger("models.loader")


class ModelManager:
    """
    Singleton model manager that lazy-loads and caches Keras models.

    Usage:
        manager = ModelManager()
        model = manager.get("resnet")
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._cache = {}
                cls._instance._initialized = False
            return cls._instance

    def _validate_paths(self):
        """Check which model files exist on disk."""
        if self._initialized:
            return

        for name, cfg in MODEL_CONFIGS.items():
            path = cfg["path"]
            exists = os.path.isfile(path)
            status = "✅ found" if exists else "❌ missing"
            logger.info(f"Model '{name}': {status} → {path}")

        self._initialized = True

    def get(self, name: str):
        """
        Load and return a Keras model by name (cached after first load).

        Args:
            name: Model identifier ('cnn', 'resnet', 'efficientnet').

        Returns:
            Loaded Keras model.

        Raises:
            ValueError: If model name is unknown.
            FileNotFoundError: If model file doesn't exist.
        """
        self._validate_paths()

        if name in self._cache:
            logger.debug(f"Model '{name}' loaded from cache.")
            return self._cache[name]

        cfg = MODEL_CONFIGS.get(name)
        if cfg is None:
            raise ValueError(
                f"Unknown model '{name}'. Available: {list(MODEL_CONFIGS.keys())}"
            )

        path = cfg["path"]
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Please train the model first (see README)."
            )

        logger.info(f"Loading model '{name}' from {path}…")
        with self._lock:
            # Double-check after acquiring lock
            if name not in self._cache:
                self._cache[name] = load_model(path)
        logger.info(f"Model '{name}' loaded successfully.")

        return self._cache[name]

    def get_all(self) -> dict:
        """
        Load and return all available models.

        Returns:
            Dict mapping model name → loaded Keras model.
        """
        available = {}
        for name in MODEL_CONFIGS:
            try:
                available[name] = self.get(name)
            except FileNotFoundError:
                logger.warning(f"Skipping model '{name}' — file not found.")
        return available

    def available_models(self) -> list:
        """Return list of model names whose .h5 files exist."""
        return [
            name for name, cfg in MODEL_CONFIGS.items()
            if os.path.isfile(cfg["path"])
        ]

    def is_loaded(self, name: str) -> bool:
        """Check if a model is currently loaded in cache."""
        return name in self._cache


# Module-level convenience instance
model_manager = ModelManager()
