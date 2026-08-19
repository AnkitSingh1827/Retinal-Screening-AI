"""
src/utils.py
============
Shared helper utilities used across the project.
Keeps the other modules clean and avoids code duplication.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────

def get_logger(name: str = "retinal_screening") -> logging.Logger:
    """
    Returns a configured logger.
    Call once per module: logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# FILE / DIRECTORY HELPERS
# ─────────────────────────────────────────────────────────

def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    """Serialize data to a JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_json_serializer)
    logger.info(f"Saved JSON → {path}")


def load_json(path: Path) -> Any:
    """Load and return data from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_serializer(obj: Any) -> Any:
    """Custom serializer so numpy types don't break json.dump."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ─────────────────────────────────────────────────────────
# CLASS NAMES
# ─────────────────────────────────────────────────────────

def save_class_names(class_names: List[str], path: Path) -> None:
    """Save the ordered list of class names to JSON."""
    save_json(class_names, path)


def load_class_names(path: Path) -> List[str]:
    """Load class names from JSON. Returns a list of strings."""
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    return data


# ─────────────────────────────────────────────────────────
# TIMING
# ─────────────────────────────────────────────────────────

class Timer:
    """Simple wall-clock timer. Use as a context manager."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start

    def __str__(self):
        return f"{self.elapsed:.2f}s"


# ─────────────────────────────────────────────────────────
# NUMPY / ARRAY HELPERS
# ─────────────────────────────────────────────────────────

def set_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across numpy and Python random.
    TensorFlow/Keras seeds are set separately in train.py because TF
    must be imported before seeding.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> Dict[int, float]:
    """
    Compute balanced class weights for imbalanced datasets.
    Formula: weight[c] = n_samples / (n_classes * n_samples_class_c)

    Args:
        labels:      1-D array of integer class indices.
        num_classes: total number of classes.

    Returns:
        dict mapping class_index → weight_float
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ─────────────────────────────────────────────────────────
# IMAGE HELPERS
# ─────────────────────────────────────────────────────────

def pil_to_numpy(pil_image) -> np.ndarray:
    """Convert a PIL Image to a numpy uint8 array (H, W, 3)."""
    import numpy as np
    img = pil_image.convert("RGB")
    return np.array(img, dtype=np.uint8)


def numpy_to_pil(array: np.ndarray):
    """Convert a numpy uint8 array (H, W, 3) to a PIL Image."""
    from PIL import Image
    if array.dtype != np.uint8:
        array = (array * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize a numpy image (H, W, C) to (size[0], size[1], C).
    Uses INTER_AREA for downscaling, INTER_LINEAR for upscaling.
    """
    import cv2
    h, w = size
    curr_h, curr_w = image.shape[:2]
    interp = cv2.INTER_AREA if (h < curr_h or w < curr_w) else cv2.INTER_LINEAR
    return cv2.resize(image, (w, h), interpolation=interp)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1].
    Input should be uint8 or float in [0, 255].
    """
    return image.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────

def print_section(title: str, width: int = 60) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_class_distribution(labels: List[str], class_names: List[str]) -> None:
    """Pretty-print class distribution."""
    from collections import Counter
    counts = Counter(labels)
    total = sum(counts.values())
    print_section("Class Distribution")
    for name in class_names:
        n = counts.get(name, 0)
        bar = "█" * int(30 * n / total) if total > 0 else ""
        print(f"  {name:<12} {n:>5}  ({100*n/total:5.1f}%)  {bar}")
    print(f"\n  Total: {total}")
