"""
src/preprocessing.py
====================
Data preprocessing, augmentation, and train/val/test splitting.

Key responsibilities
────────────────────
• Patient-level splitting (no data leakage between splits)
• Image validation and corruption handling
• Keras ImageDataGenerator for augmentation (train only)
• Retinal-safe augmentation settings
• Produces split CSV files for reproducible downstream use
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.config import (
    AUGMENTATION_CONFIG,
    IMAGE_SIZE,
    RANDOM_SEED,
    SPLITS_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    TARGET_CLASSES,
)
from src.utils import ensure_dir, get_logger, set_seeds

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# 1.  PATIENT-LEVEL SPLITTING
# ─────────────────────────────────────────────────────────

def split_by_patient(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    test_ratio:  float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset at the PATIENT level to prevent data leakage.

    If 'patient_id' is available and has multiple unique values, we use
    GroupShuffleSplit so that both eyes of the same patient land in the
    same split.

    If patient_id is not available (or all ids are unique == row-level),
    we fall back to a stratified image-level split.

    Returns (train_df, val_df, test_df)
    """
    set_seeds(seed)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    has_patient_id = (
        "patient_id" in df.columns
        and df["patient_id"].notna().all()
        and df["patient_id"].nunique() < len(df)   # some patients have 2 eyes
    )

    if has_patient_id:
        logger.info("Using patient-level splitting (GroupShuffleSplit) …")
        patient_ids = df["patient_id"].values
        labels      = df["class_index"].values

        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_val_idx, test_idx = next(gss_test.split(df, labels, groups=patient_ids))

        df_train_val = df.iloc[train_val_idx]
        df_test      = df.iloc[test_idx]

        # Now split train_val into train / val
        val_frac_of_trainval = val_ratio / (train_ratio + val_ratio)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_trainval, random_state=seed)
        tv_patient_ids = df_train_val["patient_id"].values
        tv_labels      = df_train_val["class_index"].values
        train_idx_local, val_idx_local = next(
            gss_val.split(df_train_val, tv_labels, groups=tv_patient_ids)
        )
        df_train = df_train_val.iloc[train_idx_local]
        df_val   = df_train_val.iloc[val_idx_local]

    else:
        logger.info("Patient-level split not possible. Using stratified image-level split …")
        df_train_val, df_test = train_test_split(
            df, test_size=test_ratio, stratify=df["class_index"], random_state=seed
        )
        val_frac_of_trainval = val_ratio / (train_ratio + val_ratio)
        df_train, df_val = train_test_split(
            df_train_val,
            test_size=val_frac_of_trainval,
            stratify=df_train_val["class_index"],
            random_state=seed,
        )

    logger.info(f"Split sizes — train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
    _check_class_coverage(df_train, "train")
    _check_class_coverage(df_val,   "val")
    _check_class_coverage(df_test,  "test")
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def _check_class_coverage(df: pd.DataFrame, split_name: str) -> None:
    """Warn if any class is missing from a split."""
    present = set(df["class_name"].unique())
    missing = set(TARGET_CLASSES) - present
    if missing:
        logger.warning(f"Split '{split_name}' is missing classes: {missing}")


def save_splits(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
) -> None:
    """Persist split DataFrames as CSV files."""
    ensure_dir(SPLITS_DIR)
    df_train.to_csv(SPLITS_DIR / "train.csv", index=False)
    df_val.to_csv(  SPLITS_DIR / "val.csv",   index=False)
    df_test.to_csv( SPLITS_DIR / "test.csv",  index=False)
    logger.info(f"Split CSVs saved to {SPLITS_DIR}")


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-saved split CSVs."""
    def _load(name):
        p = SPLITS_DIR / f"{name}.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"Split file not found: {p}\n"
                "Run: python -m src.dataset  then  python -m src.train"
            )
        return pd.read_csv(p)
    return _load("train"), _load("val"), _load("test")


# ─────────────────────────────────────────────────────────
# 2.  IMAGE PREPROCESSING (single image)
# ─────────────────────────────────────────────────────────

def preprocess_image(
    pil_image,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    normalize: bool = True,
) -> Optional[np.ndarray]:
    """
    Validate, convert, resize, and optionally normalize a single PIL image.

    Returns
    -------
    np.ndarray of shape (H, W, 3) float32  if successful
    None                                    if the image is invalid/corrupted
    """
    try:
        import cv2
        from PIL import Image, UnidentifiedImageError

        if pil_image is None:
            logger.warning("Received None image.")
            return None

        # Ensure PIL Image
        if not isinstance(pil_image, Image.Image):
            logger.warning(f"Expected PIL.Image, got {type(pil_image)}")
            return None

        # Convert to RGB (handles grayscale, RGBA, palette images)
        img_rgb = pil_image.convert("RGB")
        img_np  = np.array(img_rgb, dtype=np.uint8)

        # Minimum size check
        h, w = img_np.shape[:2]
        if h < 50 or w < 50:
            logger.warning(f"Image too small: {w}×{h}")
            return None

        # Resize
        img_resized = cv2.resize(
            img_np,
            (target_size[1], target_size[0]),   # cv2 takes (width, height)
            interpolation=cv2.INTER_AREA if (h > target_size[0]) else cv2.INTER_LINEAR,
        )

        if normalize:
            return img_resized.astype(np.float32) / 255.0
        return img_resized

    except Exception as exc:
        logger.warning(f"Failed to preprocess image: {exc}")
        return None


# ─────────────────────────────────────────────────────────
# 3.  KERAS DATA GENERATORS
# ─────────────────────────────────────────────────────────

def _get_train_datagen():
    """Return an ImageDataGenerator with retinal-safe augmentations."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

    cfg = AUGMENTATION_CONFIG
    return ImageDataGenerator(
        rotation_range     = cfg["rotation_range"],
        width_shift_range  = cfg["width_shift_range"],
        height_shift_range = cfg["height_shift_range"],
        zoom_range         = cfg["zoom_range"],
        horizontal_flip    = cfg["horizontal_flip"],
        brightness_range   = cfg.get("brightness_range"),
        fill_mode          = cfg["fill_mode"],
        # EfficientNet expects pixel values in [0, 1] after our normalization
        # (we do NOT use rescale here because we handle normalization manually)
        rescale            = None,
    )


def _get_val_test_datagen():
    """Return an ImageDataGenerator with NO augmentation."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
    return ImageDataGenerator()   # identity — no augmentation


# ─────────────────────────────────────────────────────────
# 4.  HUGGING FACE DATASET → NUMPY ARRAYS
# ─────────────────────────────────────────────────────────

def build_numpy_arrays(
    hf_dataset,
    df_split: pd.DataFrame,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build numpy (X, y) arrays from the HF dataset filtered to the rows
    in df_split (by positional index in the HF dataset, using patient_id).

    Because the HF dataset has one image per row and our DataFrame rows
    correspond to HF dataset rows (same order), we can use integer indices.

    Parameters
    ----------
    hf_dataset : datasets.Dataset
    df_split   : DataFrame with columns [class_index, ...]
                 The index of df_split should correspond to HF row indices.
    target_size: (H, W)
    verbose    : print progress every 200 images

    Returns
    -------
    X : float32 array shape (N, H, W, 3)
    y : int32   array shape (N,)
    """
    indices = df_split.index.tolist()   # row positions in the HF dataset
    N = len(indices)

    X = np.zeros((N,) + target_size + (3,), dtype=np.float32)
    y = df_split["class_index"].values.astype(np.int32)

    bad_count = 0
    for i, hf_idx in enumerate(indices):
        if verbose and i % 200 == 0:
            logger.info(f"  Processing image {i+1}/{N} …")
        try:
            row = hf_dataset[int(hf_idx)]
            pil_img = row["image"]
            processed = preprocess_image(pil_img, target_size=target_size, normalize=True)
            if processed is not None:
                X[i] = processed
            else:
                bad_count += 1
        except Exception as exc:
            logger.warning(f"  Row {hf_idx}: error — {exc}")
            bad_count += 1

    if bad_count > 0:
        logger.warning(f"{bad_count} images were invalid or corrupted and replaced with zeros.")

    logger.info(f"Built array: X={X.shape}, y={y.shape}")
    return X, y


# ─────────────────────────────────────────────────────────
# 5.  FULL PREPROCESSING PIPELINE  (called from train.py)
# ─────────────────────────────────────────────────────────

def run_full_pipeline(hf_dataset, df_labelled: pd.DataFrame):
    """
    Orchestrates:
      1. Patient-level split
      2. Save split CSVs
      3. Build numpy arrays for train / val / test
      4. Return (X_train, y_train, X_val, y_val, X_test, y_test, datagen_train)
    """
    logger.info("Running full preprocessing pipeline …")

    df_train, df_val, df_test = split_by_patient(df_labelled)
    save_splits(df_train, df_val, df_test)

    logger.info("Building training numpy arrays …")
    X_train, y_train = build_numpy_arrays(hf_dataset, df_train)

    logger.info("Building validation numpy arrays …")
    X_val, y_val = build_numpy_arrays(hf_dataset, df_val)

    logger.info("Building test numpy arrays …")
    X_test, y_test = build_numpy_arrays(hf_dataset, df_test)

    datagen_train = _get_train_datagen()

    return X_train, y_train, X_val, y_val, X_test, y_test, datagen_train


# ─────────────────────────────────────────────────────────
# 6.  IMAGE QUALITY CHECK (used in Streamlit UI + predict.py)
# ─────────────────────────────────────────────────────────

def check_image_quality(pil_image) -> Tuple[bool, str]:
    """
    Basic image quality assessment before running inference.

    Returns
    -------
    (is_ok: bool, message: str)
    """
    from PIL import Image
    import cv2

    try:
        if pil_image is None:
            return False, "No image provided."

        img = pil_image.convert("RGB")
        arr = np.array(img)

        h, w = arr.shape[:2]
        from config.config import MIN_IMAGE_DIMENSION
        if h < MIN_IMAGE_DIMENSION or w < MIN_IMAGE_DIMENSION:
            return False, (
                f"Image is too small ({w}×{h} px). "
                f"Minimum required: {MIN_IMAGE_DIMENSION}×{MIN_IMAGE_DIMENSION} px."
            )

        # Brightness check (mean pixel value in grayscale)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        mean_bright = gray.mean()
        if mean_bright < 10:
            return False, "Image appears extremely dark. Please upload a properly lit fundus image."
        if mean_bright > 245:
            return False, "Image appears extremely bright/overexposed. Please check the image."

        # Blur check using Laplacian variance
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 25:
            return False, (
                "Image appears blurry (sharpness score: "
                f"{lap_var:.1f}). Please upload a clearer retinal fundus image."
            )

        return True, "Image quality is acceptable."

    except Exception as exc:
        return False, f"Could not assess image quality: {exc}"
