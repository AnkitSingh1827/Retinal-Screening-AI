"""
src/explainability.py
=====================
Grad-CAM (Gradient-weighted Class Activation Mapping) for EfficientNetB0.

Usage example
─────────────
    from src.explainability import generate_gradcam_overlay
    from PIL import Image
    import numpy as np

    pil_img = Image.open("fundus.jpg")
    result  = generate_gradcam_overlay(model, pil_img, class_index=2)
    # result["overlay"]  → PIL Image with heatmap overlay
    # result["heatmap"]  → raw heatmap as uint8 array

IMPORTANT DISCLAIMER
────────────────────
Grad-CAM highlights regions that influenced the model's prediction.
It is NOT a medical diagnosis tool and should NOT be interpreted as
confirmation of disease presence or absence.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.config import (
    GRADCAM_ALPHA,
    GRADCAM_COLORMAP,
    GRADCAM_LAYER_NAME,
    IMAGE_SIZE,
    TARGET_CLASSES,
)
from src.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# GRAD-CAM IMPLEMENTATION
# ─────────────────────────────────────────────────────────

def _find_gradcam_layer(model, preferred_name: str) -> Optional[str]:
    """
    Find the best convolutional layer for Grad-CAM.
    Searches for the preferred name first, then falls back to the last Conv layer.
    """
    # Search in top-level layers
    for layer in reversed(model.layers):
        if layer.name == preferred_name:
            return layer.name

    # If the backbone is a sub-model, search inside it
    for layer in model.layers:
        try:
            sub_layers = layer.layers
            for sl in reversed(sub_layers):
                if sl.name == preferred_name:
                    # Return as backbone.layer_name
                    return sl.name
                if "conv" in sl.name.lower() and len(sl.output_shape) == 4:
                    last_conv = sl.name
            # Return last conv found in sub-model
            return last_conv
        except AttributeError:
            continue

    # Fallback: last Conv2D in main model
    last_conv = None
    for layer in model.layers:
        if "conv" in layer.name.lower():
            try:
                if len(layer.output_shape) == 4:
                    last_conv = layer.name
            except AttributeError:
                pass
    return last_conv


def compute_gradcam(
    model,
    img_array: np.ndarray,
    class_index: int,
    layer_name: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Compute the Grad-CAM heatmap for a single image.

    Parameters
    ----------
    model       : Keras model
    img_array   : float32 array, shape (1, H, W, 3), values in [0, 1]
    class_index : index of the target class
    layer_name  : name of the convolutional layer to use
                  (auto-detected if None)

    Returns
    -------
    heatmap : float32 array of shape (H, W), values in [0, 1]
              or None if computation fails
    """
    try:
        import tensorflow as tf
        from tensorflow import keras

        if layer_name is None:
            layer_name = _find_gradcam_layer(model, GRADCAM_LAYER_NAME)

        if layer_name is None:
            logger.warning("Could not find a suitable convolutional layer for Grad-CAM.")
            return None

        # Build a sub-model that outputs both the target layer and the final predictions
        # We need to handle the case where the conv layer is inside a sub-model (EfficientNet)
        try:
            # Try direct layer access
            target_layer = model.get_layer(layer_name)
            grad_model = keras.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output],
            )
        except ValueError:
            # Layer is inside a backbone sub-model
            backbone = None
            for l in model.layers:
                try:
                    l.get_layer(layer_name)
                    backbone = l
                    break
                except (ValueError, AttributeError):
                    continue

            if backbone is None:
                logger.warning(f"Layer '{layer_name}' not found. Skipping Grad-CAM.")
                return None

            target_layer = backbone.get_layer(layer_name)
            grad_model = keras.Model(
                inputs=model.inputs,
                outputs=[target_layer.output, model.output],
            )

        # Compute gradients
        with tf.GradientTape() as tape:
            img_tensor = tf.cast(img_array, tf.float32)
            tape.watch(img_tensor)
            conv_outputs, predictions = grad_model(img_tensor)
            target_score = predictions[:, class_index]

        grads = tape.gradient(target_score, conv_outputs)

        if grads is None:
            logger.warning("Gradient computation returned None. Grad-CAM unavailable.")
            return None

        # Pool gradients over spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Compute weighted feature map
        conv_outputs_np = conv_outputs[0].numpy()
        pooled_grads_np = pooled_grads.numpy()

        heatmap = np.zeros(conv_outputs_np.shape[:2], dtype=np.float32)
        for i, w in enumerate(pooled_grads_np):
            heatmap += w * conv_outputs_np[:, :, i]

        # Apply ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap

    except Exception as exc:
        logger.error(f"Grad-CAM computation failed: {exc}")
        return None


def heatmap_to_colormap(heatmap: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize heatmap to target_size and apply a colormap.

    Returns
    -------
    colored_heatmap : uint8 array of shape (H, W, 3), RGB format
    """
    import cv2

    # Resize to target image size
    heatmap_resized = cv2.resize(heatmap, (target_size[1], target_size[0]))

    # Apply colormap (jet by default → red=high, blue=low)
    colormap_map = {
        "jet":   cv2.COLORMAP_JET,
        "hot":   cv2.COLORMAP_HOT,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma":  cv2.COLORMAP_PLASMA,
    }
    cv2_cmap = colormap_map.get(GRADCAM_COLORMAP, cv2.COLORMAP_JET)

    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2_cmap)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored_rgb


def overlay_heatmap(
    original_img: np.ndarray,
    colored_heatmap: np.ndarray,
    alpha: float = GRADCAM_ALPHA,
) -> np.ndarray:
    """
    Blend the original image with the heatmap overlay.

    Parameters
    ----------
    original_img     : uint8 or float array (H, W, 3)
    colored_heatmap  : uint8 array (H, W, 3)
    alpha            : heatmap transparency (0=invisible, 1=opaque)

    Returns
    -------
    overlay : uint8 array (H, W, 3)
    """
    if original_img.dtype != np.uint8:
        orig_uint8 = (original_img * 255).clip(0, 255).astype(np.uint8)
    else:
        orig_uint8 = original_img.copy()

    overlay = (
        (1 - alpha) * orig_uint8.astype(np.float32)
        + alpha * colored_heatmap.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    return overlay


def generate_gradcam_overlay(
    model,
    pil_image,
    class_index: int,
    layer_name: Optional[str] = None,
) -> dict:
    """
    Full Grad-CAM pipeline: PIL image in → overlay + heatmap dict out.

    Parameters
    ----------
    model       : Keras model
    pil_image   : PIL.Image.Image  (any size / mode)
    class_index : predicted class index
    layer_name  : convolutional layer to target (auto-detected if None)

    Returns
    -------
    {
        "original"  : PIL.Image   — resized original (224×224)
        "heatmap"   : np.uint8    — coloured heatmap (224×224×3)
        "overlay"   : PIL.Image   — heatmap overlaid on original
        "success"   : bool
        "message"   : str
    }
    """
    from PIL import Image
    from src.preprocessing import preprocess_image

    result = {
        "original": None,
        "heatmap":  None,
        "overlay":  None,
        "success":  False,
        "message":  "",
    }

    try:
        # Preprocess: resize + normalize → (H, W, 3) float32
        img_processed = preprocess_image(pil_image, target_size=IMAGE_SIZE, normalize=True)
        if img_processed is None:
            result["message"] = "Image preprocessing failed."
            return result

        # Keep uint8 version for overlay
        img_uint8 = (img_processed * 255).astype(np.uint8)
        result["original"] = Image.fromarray(img_uint8)

        # Batch dimension: (1, H, W, 3)
        img_batch = np.expand_dims(img_processed, axis=0)

        # Compute Grad-CAM heatmap
        heatmap = compute_gradcam(model, img_batch, class_index, layer_name)

        if heatmap is None:
            result["message"] = "Grad-CAM computation was not successful for this image."
            return result

        # Convert to coloured heatmap
        colored = heatmap_to_colormap(heatmap, IMAGE_SIZE)
        result["heatmap"] = colored

        # Overlay
        overlay_arr = overlay_heatmap(img_uint8, colored, alpha=GRADCAM_ALPHA)
        result["overlay"] = Image.fromarray(overlay_arr)

        result["success"] = True
        result["message"] = (
            "Highlighted regions show areas that influenced the model prediction. "
            "This is an AI explanation and is NOT a medical diagnosis."
        )
        return result

    except Exception as exc:
        logger.error(f"generate_gradcam_overlay failed: {exc}")
        result["message"] = f"Grad-CAM error: {exc}"
        return result
