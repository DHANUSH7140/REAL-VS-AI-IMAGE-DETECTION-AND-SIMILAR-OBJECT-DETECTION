"""
utils/logger.py — Centralized logging configuration.

Sets up file + console logging with structured format.
"""

import os
import logging
from config import LOG_DIR


def setup_logger(name: str = "app", level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name:  Logger name (typically module name).
        level: Logging level (DEBUG, INFO, WARNING, etc.).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ─────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler ────────────────────────────────────────────
    log_file = os.path.join(LOG_DIR, "app.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
