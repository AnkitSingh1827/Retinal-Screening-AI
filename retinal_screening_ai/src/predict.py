"""
src/predict.py
==============
Inference pipeline for a single retinal fundus image.

Usage (Python)
──────────────
    from src.predict import RetinalPredictor
    predictor = RetinalPredictor()
    result = predictor.predict(pil_image)
    print(result["prediction"], result["confidence"])

Usage (command line)
──────────────────────
    python -m src.predict path/to/fundus_image.jpg
"""

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    IMAGE_SIZE,
    INPUT_SHAPE,
    LOW_CONFIDENCE_THRESHOLD,
    TARGET_CLASSES,
)
from src.preprocessing import check_image_quality, preprocess_image
from src.utils import get_logger, load_class_names

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# PREDICTOR CLASS
# ─────────────────────────────────────────────────────────

class RetinalPredictor:
    """
    Wraps the trained Keras model for single-image inference.

    Loads the model once and caches it — safe to reuse across Streamlit reruns
    via st.session_state.

    Parameters
    ----------
    model_path       : path to the saved .keras model
    class_names_path : path to class_names.json
    """

    def __init__(
        self,
        model_path:        Path = BEST_MODEL_PATH,
        class_names_path:  Path = CLASS_NAMES_PATH,
    ):
        self.model_path       = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self._model     = None
        self._class_names = None

    # ── Lazy loading ─────────────────────────────────────

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def class_names(self):
        if self._class_names is None:
            self._class_names = self._load_class_names()
        return self._class_names

    def _load_model(self):
        """Load and return the Keras model."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Please run: python -m src.train"
            )
        from tensorflow import keras
        logger.info(f"Loading model from {self.model_path} …")
        try:
            model = keras.models.load_model(str(self.model_path))
            logger.info("Model loaded successfully.")
            return model
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            raise

    def _load_class_names(self):
        """Load class names from JSON."""
        try:
            return load_class_names(self.class_names_path)
        except FileNotFoundError:
            logger.warning(f"class_names.json not found. Using TARGET_CLASSES.")
            return TARGET_CLASSES

    def is_ready(self) -> bool:
        """Return True if the model file exists."""
        return self.model_path.exists()

    # ── Core prediction ──────────────────────────────────

    def predict(
        self,
        pil_image,
        run_gradcam: bool = True,
    ) -> dict:
        """
        Full prediction pipeline for a single PIL Image.

        Parameters
        ----------
        pil_image   : PIL.Image.Image
        run_gradcam : whether to generate Grad-CAM explanation

        Returns
        -------
        dict with keys:
          prediction     : str   — predicted class name
          class_index    : int   — predicted class index
          confidence     : float — softmax probability of predicted class [0, 1]
          probabilities  : dict  — {class_name: probability} for all classes
          low_confidence : bool  — True if confidence < LOW_CONFIDENCE_THRESHOLD
          quality_ok     : bool  — True if image passed quality check
          quality_msg    : str   — quality check message
          gradcam        : dict  — Grad-CAM result dict (or None)
          error          : str | None — error message if prediction failed
        """
        result = {
            "prediction":     None,
            "class_index":    None,
            "confidence":     None,
            "probabilities":  {},
            "low_confidence": False,
            "quality_ok":     True,
            "quality_msg":    "",
            "gradcam":        None,
            "error":          None,
        }

        # ── Step 1: Image quality check ───────────────────
        quality_ok, quality_msg = check_image_quality(pil_image)
        result["quality_ok"]  = quality_ok
        result["quality_msg"] = quality_msg

        if not quality_ok:
            result["error"] = quality_msg
            return result

        # ── Step 2: Preprocess ────────────────────────────
        img_array = preprocess_image(pil_image, target_size=IMAGE_SIZE, normalize=True)
        if img_array is None:
            result["error"] = "Image preprocessing failed. Please try a different image."
            return result

        img_batch = np.expand_dims(img_array, axis=0)   # (1, 224, 224, 3)

        # ── Step 3: Predict ───────────────────────────────
        try:
            probs = self.model.predict(img_batch, verbose=0)[0]   # (num_classes,)
        except Exception as exc:
            logger.error(f"Model prediction failed: {exc}")
            result["error"] = f"Prediction error: {exc}"
            return result

        class_index = int(np.argmax(probs))
        confidence  = float(probs[class_index])
        class_name  = self.class_names[class_index]

        result["prediction"]    = class_name
        result["class_index"]   = class_index
        result["confidence"]    = confidence
        result["probabilities"] = {
            name: float(p) for name, p in zip(self.class_names, probs)
        }
        result["low_confidence"] = confidence < LOW_CONFIDENCE_THRESHOLD

        # ── Step 4: Grad-CAM ──────────────────────────────
        if run_gradcam:
            try:
                from src.explainability import generate_gradcam_overlay
                gradcam_result = generate_gradcam_overlay(
                    self.model, pil_image, class_index=class_index
                )
                result["gradcam"] = gradcam_result
            except Exception as exc:
                logger.warning(f"Grad-CAM failed (non-critical): {exc}")
                result["gradcam"] = {"success": False, "message": str(exc)}

        return result


# ─────────────────────────────────────────────────────────
# COMMAND-LINE USAGE
# ─────────────────────────────────────────────────────────

def _cli():
    """Allow running: python -m src.predict path/to/image.jpg"""
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(description="Retinal screening prediction")
    parser.add_argument("image", type=str, help="Path to retinal fundus image")
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: File not found: {img_path}")
        sys.exit(1)

    try:
        pil_img = Image.open(img_path)
    except Exception as exc:
        print(f"Error loading image: {exc}")
        sys.exit(1)

    predictor = RetinalPredictor()
    if not predictor.is_ready():
        print("Model not found. Please run: python -m src.train")
        sys.exit(1)

    print(f"\nRunning prediction on: {img_path.name}")
    result = predictor.predict(pil_img, run_gradcam=not args.no_gradcam)

    if result.get("error"):
        print(f"\n❌ Error: {result['error']}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  Predicted condition : {result['prediction']}")
    print(f"  Model confidence    : {result['confidence']*100:.1f}%")
    if result["low_confidence"]:
        print(f"  ⚠  Low confidence — professional evaluation recommended")
    print(f"\n  Probabilities:")
    for cls, prob in sorted(result["probabilities"].items(),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 30)
        print(f"    {cls:<12} {prob*100:5.1f}%  {bar}")
    print(f"{'='*50}")

    if result.get("gradcam") and result["gradcam"].get("success"):
        print("\n  Grad-CAM explanation generated successfully.")


if __name__ == "__main__":
    _cli()
