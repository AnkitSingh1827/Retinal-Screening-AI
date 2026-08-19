"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           AI-Based Retinal Imaging and Ophthalmic Screening System          ║
║                         B.Tech Final Year Project                           ║
║                                                                              ║
║  Single-file version — contains ALL project code in one file.               ║
║                                                                              ║
║  HOW TO RUN:                                                                 ║
║  ─────────────────────────────────────────────────────────────────           ║
║  1. Install dependencies:                                                    ║
║       pip install tensorflow datasets huggingface_hub opencv-python          ║
║               Pillow numpy pandas scikit-learn matplotlib seaborn            ║
║               streamlit plotly tqdm                                          ║
║                                                                              ║
║  2. Dataset inspection:    python retinal_ai_complete.py --dataset           ║
║  3. Train model:           python retinal_ai_complete.py --train             ║
║  4. Evaluate model:        python retinal_ai_complete.py --evaluate          ║
║  5. Predict single image:  python retinal_ai_complete.py --predict img.jpg   ║
║  6. Launch Streamlit UI:   streamlit run retinal_ai_complete.py              ║
║                                                                              ║
║  DISCLAIMER: For educational/research use only. NOT a medical device.       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — STANDARD LIBRARY & BASIC IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION  (equivalent to config/config.py)
# ══════════════════════════════════════════════════════════════════════════════

# ── 1.1  Root paths (auto-detected from this file's location) ────────────────
ROOT_DIR      = Path(__file__).resolve().parent
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR    = DATA_DIR / "splits"
MODELS_DIR    = ROOT_DIR / "models"
REPORTS_DIR   = ROOT_DIR / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"
METRICS_DIR   = REPORTS_DIR / "metrics"
ASSETS_DIR    = ROOT_DIR / "assets"

# ── 1.2  Dataset ─────────────────────────────────────────────────────────────
DATASET_NAME  = "bumbledeep/odir"
DATASET_SPLIT = "train"

# ── 1.3  Four-class labels ────────────────────────────────────────────────────
TARGET_CLASSES = ["Normal", "Diabetes", "Glaucoma", "AMD"]

LABEL_KEYWORD_MAP = {
    "Normal":   ["normal"],
    "Diabetes": ["diabetes", "diabetic retinopathy", "dr"],
    "Glaucoma": ["glaucoma"],
    "AMD":      ["age-related macular degeneration", "amd", "macular degeneration"],
}

# "exclude" → drop multi-label rows | "first" → keep first matched class
MULTI_LABEL_POLICY = "exclude"

# ── 1.4  Image settings ───────────────────────────────────────────────────────
IMAGE_SIZE     = (224, 224)
IMAGE_CHANNELS = 3
INPUT_SHAPE    = IMAGE_SIZE + (IMAGE_CHANNELS,)   # (224, 224, 3)

# ── 1.5  Data split ratios ────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── 1.6  Reproducibility ─────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── 1.7  Model / training hyperparameters ────────────────────────────────────
BASE_MODEL_NAME      = "EfficientNetB0"
PRETRAINED_WEIGHTS   = "imagenet"
BATCH_SIZE           = 32
EPOCHS_HEAD          = 20
EPOCHS_FINETUNE      = 10
LEARNING_RATE_HEAD   = 1e-3
LEARNING_RATE_FINETUNE = 1e-5
DROPOUT_RATE         = 0.40
DENSE_UNITS          = 256
FINETUNE_LAYERS      = 20

# ── 1.8  Callbacks ────────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE      = 3
REDUCE_LR_FACTOR        = 0.5
REDUCE_LR_MIN           = 1e-7

# ── 1.9  Model save paths ─────────────────────────────────────────────────────
BEST_MODEL_PATH    = MODELS_DIR / "best_model.keras"
CLASS_NAMES_PATH   = MODELS_DIR / "class_names.json"
TRAINING_META_PATH = MODELS_DIR / "training_metadata.json"

# ── 1.10  Prediction / UI settings ────────────────────────────────────────────
LOW_CONFIDENCE_THRESHOLD = 0.60
MIN_IMAGE_DIMENSION      = 100

# ── 1.11  Grad-CAM ────────────────────────────────────────────────────────────
GRADCAM_ALPHA      = 0.4
GRADCAM_COLORMAP   = "jet"
GRADCAM_LAYER_NAME = "top_conv"

# ── 1.12  Report paths ────────────────────────────────────────────────────────
CONFUSION_MATRIX_PATH   = FIGURES_DIR / "confusion_matrix.png"
TRAINING_ACC_PATH       = FIGURES_DIR / "training_accuracy.png"
TRAINING_LOSS_PATH      = FIGURES_DIR / "training_loss.png"
CLASSIFICATION_RPT_PATH = METRICS_DIR / "classification_report.json"
EVAL_METRICS_PATH       = METRICS_DIR / "evaluation_metrics.json"

# ── 1.13  Augmentation (training only) ────────────────────────────────────────
AUGMENTATION_CONFIG = {
    "rotation_range":    10,
    "width_shift_range": 0.05,
    "height_shift_range":0.05,
    "zoom_range":        0.10,
    "horizontal_flip":   True,
    "brightness_range":  [0.85, 1.15],
    "fill_mode":         "reflect",
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — UTILITIES  (equivalent to src/utils.py)
# ══════════════════════════════════════════════════════════════════════════════

def get_logger(name: str = "retinal_screening") -> logging.Logger:
    """Configured logger — call once per module."""
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

logger = get_logger("retinal_ai")


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_serializer(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(data: Any, path: Path, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_json_serializer)
    logger.info(f"Saved JSON → {path}")


def load_json(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_class_names(class_names: List[str], path: Path) -> None:
    save_json(class_names, path)


def load_class_names(path: Path) -> List[str]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")
    return data


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> Dict[int, float]:
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=labels
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


def pil_to_numpy(pil_image) -> np.ndarray:
    return np.array(pil_image.convert("RGB"), dtype=np.uint8)


def numpy_to_pil(array: np.ndarray):
    from PIL import Image
    if array.dtype != np.uint8:
        array = (array * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def print_section(title: str, width: int = 60) -> None:
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_class_distribution(labels: List[str], class_names: List[str]) -> None:
    from collections import Counter
    counts = Counter(labels)
    total  = sum(counts.values())
    print_section("Class Distribution")
    for name in class_names:
        n = counts.get(name, 0)
        bar = "█" * int(30 * n / total) if total > 0 else ""
        print(f"  {name:<12} {n:>5}  ({100*n/total:5.1f}%)  {bar}")
    print(f"\n  Total: {total}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATASET  (equivalent to src/dataset.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_hf_dataset(streaming: bool = False):
    """Load bumbledeep/odir from Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("'datasets' package not found. Install: pip install datasets")
        sys.exit(1)

    logger.info(f"Loading '{DATASET_NAME}' (streaming={streaming}) …")
    try:
        ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=streaming)
        logger.info(f"Dataset loaded. Records: {len(ds):,}")
        return ds
    except Exception as exc:
        logger.error(f"Failed to load dataset: {exc}")
        sys.exit(1)


def inspect_dataset(ds) -> None:
    print_section("DATASET INSPECTION REPORT")
    print(f"  Identifier : {DATASET_NAME}")
    print(f"  Split      : {DATASET_SPLIT}")
    print_section("Column Schema")
    try:
        for col, feat in ds.features.items():
            print(f"  {col:<20} {feat}")
    except AttributeError:
        print("  (streaming mode)")
    print_section("Sample Records (first 3)")
    try:
        for i, row in enumerate(ds.select(range(min(3, len(ds))))):
            print(f"\n  Row {i}:")
            for k, v in row.items():
                if k == "image":
                    print(f"    {k:<15} PIL.Image size={v.size} mode={v.mode}")
                else:
                    print(f"    {k:<15} {repr(v)}")
    except Exception as e:
        logger.warning(f"Cannot print sample rows: {e}")


def build_dataframe(ds):
    import pandas as pd
    logger.info("Building metadata DataFrame …")
    try:
        df = ds.to_pandas()
        if "image" in df.columns:
            df = df.drop(columns=["image"])
        logger.info(f"DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        logger.warning(f"to_pandas() failed ({e}). Falling back to manual iteration …")
        rows = []
        for row in ds:
            rows.append({
                "patient_id": row.get("patient_id"),
                "age":        row.get("age"),
                "sex":        row.get("sex"),
                "label":      row.get("label"),
                "label_code": row.get("label_code"),
            })
        return pd.DataFrame(rows)


def _match_label(label_text: str) -> List[str]:
    if not isinstance(label_text, str):
        return []
    lower = label_text.lower()
    return [cls for cls, kws in LABEL_KEYWORD_MAP.items()
            if any(kw in lower for kw in kws)]


def assign_four_class_labels(df):
    import pandas as pd
    logger.info("Assigning four-class labels …")
    df = df.copy()
    df["matched_classes"] = df["label"].apply(_match_label)

    no_match   = df["matched_classes"].apply(lambda x: len(x) == 0)
    single_match = df["matched_classes"].apply(lambda x: len(x) == 1)
    multi_match  = df["matched_classes"].apply(lambda x: len(x) > 1)

    n_no_match = no_match.sum()
    n_single   = single_match.sum()
    n_multi    = multi_match.sum()

    logger.info(f"  Total: {len(df):,} | No-match: {n_no_match:,} | Single: {n_single:,} | Multi: {n_multi:,}")

    def _pick_class(matches):
        if len(matches) == 0: return None
        if len(matches) == 1: return matches[0]
        if MULTI_LABEL_POLICY == "exclude": return "__multi__"
        for cls in TARGET_CLASSES:
            if cls in matches: return cls
        return None

    df["class_name"] = df["matched_classes"].apply(_pick_class)
    df_valid = df[df["class_name"].isin(TARGET_CLASSES)].copy()
    df_valid["class_index"] = df_valid["class_name"].apply(TARGET_CLASSES.index)

    stats = {
        "total_records":      int(len(df)),
        "kept_records":       int(len(df_valid)),
        "multi_label_excl":   int(n_multi if MULTI_LABEL_POLICY == "exclude" else 0),
        "class_distribution": df_valid["class_name"].value_counts().to_dict(),
    }

    logger.info(f"Records kept: {len(df_valid):,}")
    print_class_distribution(df_valid["class_name"].tolist(), TARGET_CLASSES)
    return df_valid, stats


def save_metadata_csv(df) -> Path:
    ensure_dir(SPLITS_DIR)
    out = SPLITS_DIR / "full_dataset.csv"
    cols = [c for c in ["patient_id", "age", "sex", "label", "class_name", "class_index"]
            if c in df.columns]
    df[cols].to_csv(out, index=False)
    logger.info(f"Metadata CSV → {out}")
    return out


def plot_class_distribution(df, save: bool = True) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed. Skipping plot.")
        return

    counts  = df["class_name"].value_counts().reindex(TARGET_CLASSES, fill_value=0)
    palette = ["#4CAF50", "#FF5722", "#2196F3", "#FF9800"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bars = axes[0].bar(counts.index, counts.values, color=palette, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     str(val), ha="center", fontsize=11, fontweight="bold")
    axes[0].set_title("ODIR-5K — Four-Class Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Images")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].pie(counts.values, labels=counts.index, colors=palette,
                autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Class Proportions", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save:
        ensure_dir(FIGURES_DIR)
        fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
        logger.info("Class distribution plot saved.")
    plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PREPROCESSING  (equivalent to src/preprocessing.py)
# ══════════════════════════════════════════════════════════════════════════════

def split_by_patient(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                     test_ratio=TEST_RATIO, seed=RANDOM_SEED):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    import pandas as pd

    set_seeds(seed)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    has_pid = (
        "patient_id" in df.columns
        and df["patient_id"].notna().all()
        and df["patient_id"].nunique() < len(df)
    )

    if has_pid:
        logger.info("Patient-level splitting (GroupShuffleSplit) …")
        gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        tv_idx, test_idx = next(gss.split(df, df["class_index"], groups=df["patient_id"]))
        df_tv   = df.iloc[tv_idx]
        df_test = df.iloc[test_idx]
        val_frac = val_ratio / (train_ratio + val_ratio)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tr_idx, vl_idx = next(gss2.split(df_tv, df_tv["class_index"], groups=df_tv["patient_id"]))
        df_train = df_tv.iloc[tr_idx]
        df_val   = df_tv.iloc[vl_idx]
    else:
        logger.info("Image-level stratified split …")
        df_tv, df_test = train_test_split(
            df, test_size=test_ratio, stratify=df["class_index"], random_state=seed)
        val_frac = val_ratio / (train_ratio + val_ratio)
        df_train, df_val = train_test_split(
            df_tv, test_size=val_frac, stratify=df_tv["class_index"], random_state=seed)

    logger.info(f"Split — train:{len(df_train)}, val:{len(df_val)}, test:{len(df_test)}")
    return (df_train.reset_index(drop=True),
            df_val.reset_index(drop=True),
            df_test.reset_index(drop=True))


def save_splits(df_train, df_val, df_test) -> None:
    ensure_dir(SPLITS_DIR)
    df_train.to_csv(SPLITS_DIR / "train.csv", index=False)
    df_val.to_csv(  SPLITS_DIR / "val.csv",   index=False)
    df_test.to_csv( SPLITS_DIR / "test.csv",  index=False)
    logger.info(f"Split CSVs saved → {SPLITS_DIR}")


def preprocess_image(pil_image, target_size=IMAGE_SIZE,
                     normalize: bool = True) -> Optional[np.ndarray]:
    """Validate, convert RGB, resize, normalize a PIL image."""
    try:
        import cv2
        from PIL import Image

        if pil_image is None:
            return None
        if not isinstance(pil_image, Image.Image):
            return None

        img_rgb = pil_image.convert("RGB")
        img_np  = np.array(img_rgb, dtype=np.uint8)

        h, w = img_np.shape[:2]
        if h < 50 or w < 50:
            return None

        img_res = cv2.resize(
            img_np,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_AREA if h > target_size[0] else cv2.INTER_LINEAR,
        )

        if normalize:
            return img_res.astype(np.float32) / 255.0
        return img_res

    except Exception as exc:
        logger.warning(f"preprocess_image failed: {exc}")
        return None


def check_image_quality(pil_image) -> Tuple[bool, str]:
    """Brightness, blur, and size check for a PIL image."""
    try:
        import cv2
        from PIL import Image

        if pil_image is None:
            return False, "No image provided."

        arr  = np.array(pil_image.convert("RGB"))
        h, w = arr.shape[:2]

        if h < MIN_IMAGE_DIMENSION or w < MIN_IMAGE_DIMENSION:
            return False, (f"Image too small ({w}×{h} px). "
                           f"Minimum: {MIN_IMAGE_DIMENSION}×{MIN_IMAGE_DIMENSION} px.")

        gray       = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        brightness = gray.mean()
        if brightness < 10:
            return False, "Image is extremely dark."
        if brightness > 245:
            return False, "Image is extremely bright/overexposed."

        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 25:
            return False, (f"Image appears blurry (sharpness={lap_var:.1f}). "
                           "Please upload a clearer fundus image.")

        return True, "Image quality is acceptable."
    except Exception as exc:
        return False, f"Quality check error: {exc}"


def _get_train_datagen():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    cfg = AUGMENTATION_CONFIG
    return ImageDataGenerator(
        rotation_range     = cfg["rotation_range"],
        width_shift_range  = cfg["width_shift_range"],
        height_shift_range = cfg["height_shift_range"],
        zoom_range         = cfg["zoom_range"],
        horizontal_flip    = cfg["horizontal_flip"],
        brightness_range   = cfg.get("brightness_range"),
        fill_mode          = cfg["fill_mode"],
        rescale            = None,
    )


def build_numpy_arrays(hf_dataset, df_split,
                       target_size=IMAGE_SIZE, verbose=True):
    indices   = df_split.index.tolist()
    N         = len(indices)
    X         = np.zeros((N,) + target_size + (3,), dtype=np.float32)
    y         = df_split["class_index"].values.astype(np.int32)
    bad_count = 0

    for i, hf_idx in enumerate(indices):
        if verbose and i % 200 == 0:
            logger.info(f"  Processing {i+1}/{N} …")
        try:
            row = hf_dataset[int(hf_idx)]
            arr = preprocess_image(row["image"], target_size=target_size, normalize=True)
            if arr is not None:
                X[i] = arr
            else:
                bad_count += 1
        except Exception as exc:
            logger.warning(f"  Row {hf_idx}: {exc}")
            bad_count += 1

    if bad_count > 0:
        logger.warning(f"{bad_count} bad/corrupted images (replaced with zeros).")
    logger.info(f"Built arrays: X={X.shape}, y={y.shape}")
    return X, y


def run_full_pipeline(hf_dataset, df_labelled):
    logger.info("Running full preprocessing pipeline …")
    df_train, df_val, df_test = split_by_patient(df_labelled)
    save_splits(df_train, df_val, df_test)

    logger.info("Building train arrays …")
    X_train, y_train = build_numpy_arrays(hf_dataset, df_train)
    logger.info("Building val arrays …")
    X_val,   y_val   = build_numpy_arrays(hf_dataset, df_val)
    logger.info("Building test arrays …")
    X_test,  y_test  = build_numpy_arrays(hf_dataset, df_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, _get_train_datagen()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MODEL TRAINING  (equivalent to src/train.py)
# ══════════════════════════════════════════════════════════════════════════════

def _setup_tensorflow():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError: pass
    tf.random.set_seed(RANDOM_SEED)
    logger.info(f"TF {tf.__version__} | GPUs: {len(gpus)}")
    return tf


def build_model(num_classes: int = 4, trainable_backbone: bool = False):
    import tensorflow as tf
    from tensorflow import keras

    backbone_map = {
        "EfficientNetB0": keras.applications.EfficientNetB0,
        "MobileNetV2":    keras.applications.MobileNetV2,
        "ResNet50":       keras.applications.ResNet50,
    }
    BackboneClass = backbone_map.get(BASE_MODEL_NAME, keras.applications.EfficientNetB0)
    logger.info(f"Building {BASE_MODEL_NAME} (trainable_backbone={trainable_backbone})")

    try:
        backbone = BackboneClass(include_top=False, weights=PRETRAINED_WEIGHTS,
                                 input_shape=INPUT_SHAPE)
    except Exception:
        backbone = keras.applications.EfficientNetB0(include_top=False, weights=None,
                                                      input_shape=INPUT_SHAPE)
    backbone.trainable = trainable_backbone

    inputs  = keras.Input(shape=INPUT_SHAPE, name="retinal_input")
    x       = backbone(inputs, training=trainable_backbone)
    x       = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x       = keras.layers.Dense(DENSE_UNITS, activation="relu", name="dense_1")(x)
    x       = keras.layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="RetinalScreeningModel")
    logger.info(f"Total params: {model.count_params():,}")
    return model


def compile_model(model, learning_rate: float):
    from tensorflow import keras
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(phase: str = "head"):
    from tensorflow import keras
    ensure_dir(MODELS_DIR)
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH), monitor="val_accuracy",
            save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE, min_lr=REDUCE_LR_MIN, verbose=1),
    ]


def plot_history(history_head, history_ft=None) -> None:
    import matplotlib.pyplot as plt
    ensure_dir(FIGURES_DIR)

    def merge(h1, h2, key):
        vals = h1.history.get(key, [])
        if h2 is not None:
            vals = vals + h2.history.get(key, [])
        return vals

    acc     = merge(history_head, history_ft, "accuracy")
    val_acc = merge(history_head, history_ft, "val_accuracy")
    loss    = merge(history_head, history_ft, "loss")
    val_los = merge(history_head, history_ft, "val_loss")
    epochs  = range(1, len(acc) + 1)
    p1_end  = len(history_head.history.get("accuracy", []))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (tr, vl, ylabel, title) in zip(axes, [
        (acc, val_acc, "Accuracy", "Model Accuracy"),
        (loss, val_los, "Loss",    "Model Loss"),
    ]):
        ax.plot(epochs, tr, label="Train",      color="#2196F3", linewidth=2)
        ax.plot(epochs, vl, label="Validation", color="#FF5722", linewidth=2, linestyle="--")
        if history_ft and p1_end < len(epochs):
            ax.axvline(x=p1_end+0.5, color="gray", linestyle=":", linewidth=1.5, label="Fine-tune start")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel,  fontsize=12)
        ax.set_title(title,    fontsize=13, fontweight="bold")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(TRAINING_ACC_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Training plots saved.")


def save_training_metadata(history_head, history_ft, class_weights,
                            n_train, n_val, n_test, duration_s):
    best_va  = max(history_head.history.get("val_accuracy", [0]))
    best_vl  = min(history_head.history.get("val_loss",     [9999]))
    if history_ft:
        ft_a = history_ft.history.get("val_accuracy", [])
        ft_l = history_ft.history.get("val_loss", [])
        if ft_a: best_va = max(best_va, max(ft_a))
        if ft_l: best_vl = min(best_vl, min(ft_l))

    meta = {
        "model_name":          BASE_MODEL_NAME,
        "pretrained_weights":  PRETRAINED_WEIGHTS,
        "input_shape":         list(INPUT_SHAPE),
        "num_classes":         len(TARGET_CLASSES),
        "class_names":         TARGET_CLASSES,
        "batch_size":          BATCH_SIZE,
        "epochs_head":         EPOCHS_HEAD,
        "epochs_finetune":     EPOCHS_FINETUNE,
        "learning_rate_head":  LEARNING_RATE_HEAD,
        "learning_rate_ft":    LEARNING_RATE_FINETUNE,
        "dropout_rate":        DROPOUT_RATE,
        "dense_units":         DENSE_UNITS,
        "train_samples":       n_train,
        "val_samples":         n_val,
        "test_samples":        n_test,
        "class_weights":       class_weights,
        "best_val_accuracy":   round(float(best_va), 4),
        "best_val_loss":       round(float(best_vl), 4),
        "training_duration_s": round(duration_s, 1),
        "random_seed":         RANDOM_SEED,
    }
    save_json(meta, TRAINING_META_PATH)


def run_training():
    """Full training pipeline — call from CLI: python file.py --train"""
    print_section("AI RETINAL SCREENING — MODEL TRAINING")
    set_seeds(RANDOM_SEED)
    tf = _setup_tensorflow()

    ds = load_hf_dataset()
    df_meta = build_dataframe(ds)
    df_labelled, _ = assign_four_class_labels(df_meta)

    (X_train, y_train, X_val, y_val,
     X_test,  y_test,  datagen) = run_full_pipeline(ds, df_labelled)

    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
    cw = compute_class_weights(y_train, len(TARGET_CLASSES))
    logger.info(f"Class weights: {cw}")

    # Phase 1 — frozen backbone
    print_section("Phase 1: Classification Head Training")
    model = build_model(num_classes=len(TARGET_CLASSES), trainable_backbone=False)
    model = compile_model(model, LEARNING_RATE_HEAD)
    model.summary()

    gen   = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=RANDOM_SEED)
    t0    = time.perf_counter()
    hist1 = model.fit(
        gen,
        steps_per_epoch=max(1, n_train // BATCH_SIZE),
        epochs=EPOCHS_HEAD,
        validation_data=(X_val, y_val),
        callbacks=get_callbacks("head"),
        class_weight=cw, verbose=1,
    )

    # Phase 2 — fine-tune top layers
    hist2 = None
    if EPOCHS_FINETUNE > 0:
        print_section("Phase 2: Fine-tuning Backbone (top layers)")
        if BEST_MODEL_PATH.exists():
            model.load_weights(str(BEST_MODEL_PATH))
        backbone_layer = model.get_layer(index=1)
        for layer in backbone_layer.layers[-FINETUNE_LAYERS:]:
            layer.trainable = True
        model = compile_model(model, LEARNING_RATE_FINETUNE)
        hist2 = model.fit(
            gen,
            steps_per_epoch=max(1, n_train // BATCH_SIZE),
            epochs=EPOCHS_FINETUNE,
            validation_data=(X_val, y_val),
            callbacks=get_callbacks("finetune"),
            class_weight=cw, verbose=1,
        )

    duration = time.perf_counter() - t0
    logger.info(f"Training time: {duration/60:.1f} min")

    save_class_names(TARGET_CLASSES, CLASS_NAMES_PATH)
    save_training_metadata(hist1, hist2, cw, n_train, n_val, n_test, duration)
    plot_history(hist1, hist2)

    print_section("Training Complete")
    print(f"  Model saved → {BEST_MODEL_PATH}")
    print("  Next: python retinal_ai_complete.py --evaluate")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EVALUATION  (equivalent to src/evaluate.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation():
    """Full evaluation pipeline — call from CLI: python file.py --evaluate"""
    import pandas as pd
    print_section("AI RETINAL SCREENING — MODEL EVALUATION")

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {BEST_MODEL_PATH}\nRun: --train first")

    from tensorflow import keras
    model = keras.models.load_model(str(BEST_MODEL_PATH))
    logger.info("Model loaded.")

    try:
        class_names = load_class_names(CLASS_NAMES_PATH)
    except FileNotFoundError:
        class_names = TARGET_CLASSES

    test_csv = SPLITS_DIR / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"test.csv not found: {test_csv}\nRun: --train first")

    df_test = pd.read_csv(test_csv)
    ds      = load_hf_dataset()
    logger.info(f"Building test arrays ({len(df_test)} samples) …")
    X_test, y_test = build_numpy_arrays(ds, df_test)

    logger.info("Running predictions …")
    probs  = model.predict(X_test, batch_size=32, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize

    acc       = accuracy_score(y_test, y_pred)
    macro_f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
    macro_pre = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

    n_cls  = len(class_names)
    y_bin  = label_binarize(y_test, classes=list(range(n_cls)))
    try:
        roc_auc = roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = None

    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print_section("Evaluation Results")
    print(f"  Accuracy      : {acc*100:.2f}%")
    print(f"  Macro F1      : {macro_f1*100:.2f}%")
    print(f"  Macro Recall  : {macro_rec*100:.2f}%")
    if roc_auc: print(f"  Macro ROC-AUC : {roc_auc:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    metrics = {
        "accuracy":         round(float(acc), 4),
        "macro_f1":         round(float(macro_f1), 4),
        "macro_precision":  round(float(macro_pre), 4),
        "macro_recall":     round(float(macro_rec), 4),
        "roc_auc_macro":    round(float(roc_auc), 4) if roc_auc else None,
        "confusion_matrix": cm.tolist(),
        "per_class_report": report_dict,
        "class_names":      class_names,
        "n_test_samples":   int(len(y_test)),
    }
    ensure_dir(METRICS_DIR)
    save_json(metrics, EVAL_METRICS_PATH)
    save_json(report_dict, CLASSIFICATION_RPT_PATH)

    # Confusion matrix plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        ensure_dir(FIGURES_DIR)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True",      fontsize=12)
        ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix → {CONFUSION_MATRIX_PATH}")
    except Exception as e:
        logger.warning(f"Could not save confusion matrix plot: {e}")

    print_section("Evaluation Complete")
    print(f"  Metrics → {METRICS_DIR}")
    print("  Next: streamlit run retinal_ai_complete.py")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — GRAD-CAM EXPLAINABILITY  (equivalent to src/explainability.py)
# ══════════════════════════════════════════════════════════════════════════════

def _find_gradcam_layer(model, preferred_name: str) -> Optional[str]:
    for layer in reversed(model.layers):
        if layer.name == preferred_name:
            return layer.name
    last_conv = None
    for layer in model.layers:
        try:
            for sl in reversed(layer.layers):
                if sl.name == preferred_name:
                    return sl.name
                if "conv" in sl.name.lower() and len(sl.output_shape) == 4:
                    last_conv = sl.name
            if last_conv:
                return last_conv
        except AttributeError:
            if "conv" in layer.name.lower():
                try:
                    if len(layer.output_shape) == 4:
                        last_conv = layer.name
                except AttributeError:
                    pass
    return last_conv


def compute_gradcam(model, img_array: np.ndarray, class_index: int,
                    layer_name: Optional[str] = None) -> Optional[np.ndarray]:
    try:
        import tensorflow as tf
        from tensorflow import keras

        if layer_name is None:
            layer_name = _find_gradcam_layer(model, GRADCAM_LAYER_NAME)
        if layer_name is None:
            return None

        try:
            target_layer = model.get_layer(layer_name)
            grad_model   = keras.Model(inputs=model.inputs,
                                       outputs=[target_layer.output, model.output])
        except ValueError:
            backbone = None
            for l in model.layers:
                try:
                    l.get_layer(layer_name)
                    backbone = l
                    break
                except (ValueError, AttributeError):
                    continue
            if backbone is None:
                return None
            target_layer = backbone.get_layer(layer_name)
            grad_model   = keras.Model(inputs=model.inputs,
                                       outputs=[target_layer.output, model.output])

        with tf.GradientTape() as tape:
            img_tensor = tf.cast(img_array, tf.float32)
            tape.watch(img_tensor)
            conv_out, preds = grad_model(img_tensor)
            score = preds[:, class_index]

        grads       = tape.gradient(score, conv_out)
        if grads is None:
            return None

        pooled_g    = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_np     = conv_out[0].numpy()
        pooled_np   = pooled_g.numpy()
        heatmap     = np.zeros(conv_np.shape[:2], dtype=np.float32)
        for i, w in enumerate(pooled_np):
            heatmap += w * conv_np[:, :, i]

        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap

    except Exception as exc:
        logger.error(f"Grad-CAM failed: {exc}")
        return None


def generate_gradcam_overlay(model, pil_image, class_index: int,
                              layer_name=None) -> dict:
    """Full Grad-CAM pipeline → returns dict with original/heatmap/overlay."""
    import cv2
    from PIL import Image

    result = {"original": None, "heatmap": None, "overlay": None,
              "success": False, "message": ""}

    img_proc = preprocess_image(pil_image, target_size=IMAGE_SIZE, normalize=True)
    if img_proc is None:
        result["message"] = "Image preprocessing failed."
        return result

    img_u8           = (img_proc * 255).astype(np.uint8)
    result["original"] = Image.fromarray(img_u8)
    img_batch          = np.expand_dims(img_proc, axis=0)

    heatmap = compute_gradcam(model, img_batch, class_index, layer_name)
    if heatmap is None:
        result["message"] = "Grad-CAM computation was not successful."
        return result

    # Colormap
    colormap_map = {
        "jet":    cv2.COLORMAP_JET,
        "hot":    cv2.COLORMAP_HOT,
        "viridis":cv2.COLORMAP_VIRIDIS,
    }
    hm_resized = cv2.resize(heatmap, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    hm_u8      = (hm_resized * 255).astype(np.uint8)
    colored    = cv2.cvtColor(
        cv2.applyColorMap(hm_u8, colormap_map.get(GRADCAM_COLORMAP, cv2.COLORMAP_JET)),
        cv2.COLOR_BGR2RGB)
    result["heatmap"] = colored

    overlay = ((1 - GRADCAM_ALPHA) * img_u8.astype(np.float32) +
                GRADCAM_ALPHA * colored.astype(np.float32)).clip(0, 255).astype(np.uint8)
    result["overlay"]  = Image.fromarray(overlay)
    result["success"]  = True
    result["message"]  = ("Highlighted regions show areas that influenced the model prediction. "
                          "This is an AI explanation and is NOT a medical diagnosis.")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PREDICTION  (equivalent to src/predict.py)
# ══════════════════════════════════════════════════════════════════════════════

class RetinalPredictor:
    """
    Single-image inference wrapper.
    Lazy-loads the model on first call — safe to cache in Streamlit.
    """

    def __init__(self, model_path=BEST_MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
        self.model_path       = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self._model           = None
        self._class_names     = None

    @property
    def model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {self.model_path}\nRun: python retinal_ai_complete.py --train")
            from tensorflow import keras
            logger.info(f"Loading model from {self.model_path} …")
            self._model = keras.models.load_model(str(self.model_path))
            logger.info("Model loaded.")
        return self._model

    @property
    def class_names(self):
        if self._class_names is None:
            try:
                self._class_names = load_class_names(self.class_names_path)
            except FileNotFoundError:
                self._class_names = TARGET_CLASSES
        return self._class_names

    def is_ready(self) -> bool:
        return self.model_path.exists()

    def predict(self, pil_image, run_gradcam: bool = True) -> dict:
        """
        Full pipeline: quality check → preprocess → predict → Grad-CAM.
        Returns a result dict with: prediction, confidence, probabilities,
        low_confidence, quality_ok, quality_msg, gradcam, error.
        """
        result = {
            "prediction":    None,
            "class_index":   None,
            "confidence":    None,
            "probabilities": {},
            "low_confidence":False,
            "quality_ok":    True,
            "quality_msg":   "",
            "gradcam":       None,
            "error":         None,
        }

        # Step 1: Quality check
        ok, msg = check_image_quality(pil_image)
        result["quality_ok"]  = ok
        result["quality_msg"] = msg
        if not ok:
            result["error"] = msg
            return result

        # Step 2: Preprocess
        arr = preprocess_image(pil_image, target_size=IMAGE_SIZE, normalize=True)
        if arr is None:
            result["error"] = "Image preprocessing failed."
            return result
        batch = np.expand_dims(arr, axis=0)

        # Step 3: Predict
        try:
            probs = self.model.predict(batch, verbose=0)[0]
        except Exception as exc:
            result["error"] = f"Prediction error: {exc}"
            return result

        idx  = int(np.argmax(probs))
        conf = float(probs[idx])
        name = self.class_names[idx]

        result["prediction"]    = name
        result["class_index"]   = idx
        result["confidence"]    = conf
        result["probabilities"] = {n: float(p) for n, p in zip(self.class_names, probs)}
        result["low_confidence"] = conf < LOW_CONFIDENCE_THRESHOLD

        # Step 4: Grad-CAM
        if run_gradcam:
            try:
                gc = generate_gradcam_overlay(self.model, pil_image, class_index=idx)
                result["gradcam"] = gc
            except Exception as exc:
                result["gradcam"] = {"success": False, "message": str(exc)}

        return result


def run_predict_cli(image_path: str, no_gradcam: bool = False):
    """CLI single-image prediction."""
    from PIL import Image as PILImage

    path = Path(image_path)
    if not path.exists():
        print(f"Error: File not found: {path}")
        sys.exit(1)

    try:
        pil_img = PILImage.open(path)
    except Exception as exc:
        print(f"Error loading image: {exc}")
        sys.exit(1)

    predictor = RetinalPredictor()
    if not predictor.is_ready():
        print("Model not found. Run: python retinal_ai_complete.py --train")
        sys.exit(1)

    print(f"\nRunning prediction on: {path.name}")
    result = predictor.predict(pil_img, run_gradcam=not no_gradcam)

    if result.get("error"):
        print(f"\n❌ Error: {result['error']}")
        sys.exit(1)

    print(f"\n{'='*52}")
    print(f"  Predicted condition : {result['prediction']}")
    print(f"  Model confidence    : {result['confidence']*100:.1f}%")
    if result["low_confidence"]:
        print("  ⚠  Low confidence — seek professional evaluation")
    print(f"\n  Probabilities:")
    for cls, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 30)
        print(f"    {cls:<12} {prob*100:5.1f}%  {bar}")
    print(f"{'='*52}")
    if result.get("gradcam") and result["gradcam"].get("success"):
        print("\n  ✅ Grad-CAM explanation generated.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — STREAMLIT APPLICATION  (equivalent to app.py)
# ══════════════════════════════════════════════════════════════════════════════

def _run_streamlit_app():
    """
    Called when running: streamlit run retinal_ai_complete.py
    Contains the full 6-tab Streamlit dashboard.
    """
    import streamlit as st
    from PIL import Image as PILImage
    import plotly.graph_objects as go

    st.set_page_config(
        page_title="AI Retinal Screening",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ───────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f1117; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%); }
    footer { visibility: hidden; }

    .card {
        background: linear-gradient(135deg, #1e2a3a 0%, #162032 100%);
        border: 1px solid #2a3f55; border-radius: 16px; padding: 24px;
        margin-bottom: 18px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .metric-card {
        background: linear-gradient(135deg, #1a2a4a 0%, #0f1f35 100%);
        border: 1px solid #2e4a6e; border-radius: 12px; padding: 20px; text-align: center;
    }
    .result-card { border-radius: 16px; padding: 28px; margin: 12px 0; text-align: center; }
    .result-normal   { background: linear-gradient(135deg, #1a4a2e, #0f2d1a); border: 1px solid #2e7d45; }
    .result-diabetes { background: linear-gradient(135deg, #4a1a1a, #2d0f0f); border: 1px solid #c0392b; }
    .result-glaucoma { background: linear-gradient(135deg, #1a2a4a, #0f1a2d); border: 1px solid #2980b9; }
    .result-amd      { background: linear-gradient(135deg, #3a2a0a, #2d1f05); border: 1px solid #f39c12; }

    .disclaimer {
        background: linear-gradient(135deg, #2d1a00, #1a1000); border: 1px solid #f39c12;
        border-radius: 12px; padding: 16px 20px; margin: 16px 0;
        color: #f0c060; font-size: 0.88rem; line-height: 1.6;
    }
    .section-title { font-size: 1.5rem; font-weight: 700; color: #e0eaff; margin-bottom: 8px; }
    .badge { display: inline-block; border-radius: 20px; padding: 5px 16px;
             font-size: 0.82rem; font-weight: 600; margin: 4px; }
    .badge-normal   { background: #1b5e20; color: #a5d6a7; }
    .badge-diabetes { background: #7f0000; color: #ef9a9a; }
    .badge-glaucoma { background: #0d47a1; color: #90caf9; }
    .badge-amd      { background: #e65100; color: #ffcc80; }

    .footer { text-align: center; padding: 20px; color: #5a6a7a;
              font-size: 0.80rem; border-top: 1px solid #1e2d3e; margin-top: 40px; }
    .info-box { background: #0d2137; border-left: 4px solid #1976D2;
                border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 10px 0;
                color: #b0c8e0; font-size: 0.90rem; }
    .big-metric { font-size: 2.5rem; font-weight: 700; line-height: 1.1; }
    .metric-label { font-size: 0.85rem; color: #7a8fa0; margin-top: 4px; }

    .stButton > button {
        background: linear-gradient(135deg, #1565C0, #0D47A1); color: white;
        border: none; border-radius: 10px; padding: 12px 30px;
        font-size: 1rem; font-weight: 600; width: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0d1b2a; border-radius: 12px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #7a8fa0; font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: #1565C0 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── Helper functions ─────────────────────────────────────────────────────
    def _safe_json(path):
        try:
            return load_json(path)
        except Exception:
            return None

    def _conf_color(c):
        return "#4CAF50" if c >= 0.80 else "#FF9800" if c >= 0.60 else "#F44336"

    def _emoji(name):
        return {"Normal":"✅","Diabetes":"🩸","Glaucoma":"👁","AMD":"🌅"}.get(name,"❓")

    def _css_class(name):
        return {"Normal":"result-normal","Diabetes":"result-diabetes",
                "Glaucoma":"result-glaucoma","AMD":"result-amd"}.get(name,"result-normal")

    def _footer():
        st.markdown("""
        <div class='footer'>
            AI-assisted screening prototype &nbsp;|&nbsp;
            For educational/research use only &nbsp;|&nbsp;
            Final assessment must be by a qualified ophthalmologist<br>
            <span style='opacity:0.5;font-size:0.75rem;'>
                EfficientNetB0 · ODIR-5K · Grad-CAM · Streamlit
            </span>
        </div>""", unsafe_allow_html=True)

    # ── Cached predictor ─────────────────────────────────────────────────────
    @st.cache_resource(show_spinner=False)
    def _get_predictor():
        try:
            return RetinalPredictor()
        except Exception:
            return None

    # ── Session state ────────────────────────────────────────────────────────
    for k, v in [("uploaded_pil", None), ("result", None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:20px 0 10px;'>
            <div style='font-size:3rem;'>👁️</div>
            <div style='font-size:1.1rem;font-weight:700;color:#e0eaff;margin-top:6px;'>
                AI Retinal Screening</div>
            <div style='font-size:0.78rem;color:#5a8fa0;margin-top:2px;'>B.Tech Final Project</div>
        </div>""", unsafe_allow_html=True)
        st.divider()
        st.markdown("#### 📊 Dataset")
        st.markdown("<div class='info-box'><b>ODIR-5K</b><br>Ocular Disease Intelligent Recognition<br>5,000 patients · Fundus images</div>", unsafe_allow_html=True)
        st.markdown("#### 🤖 Model")
        st.markdown("<div class='info-box'><b>EfficientNetB0</b><br>Transfer learning · ImageNet weights<br>Input: 224×224×3</div>", unsafe_allow_html=True)
        st.markdown("#### 🏷 Classes")
        st.markdown("""<div>
            <span class='badge badge-normal'>✅ Normal</span>
            <span class='badge badge-diabetes'>🩸 Diabetes</span><br>
            <span class='badge badge-glaucoma'>👁 Glaucoma</span>
            <span class='badge badge-amd'>🌅 AMD</span></div>""", unsafe_allow_html=True)
        st.divider()
        st.markdown("<div class='disclaimer'>⚠️ <b>Medical Disclaimer</b><br>Prototype for educational use only. NOT a medical diagnosis tool. Always consult a qualified ophthalmologist.</div>", unsafe_allow_html=True)
        st.markdown("#### 🟢 Model Status")
        if BEST_MODEL_PATH.exists():
            st.success("Model loaded ✓")
        else:
            st.error("Model not found")
            st.info("Run: `python retinal_ai_complete.py --train`")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs(["🏠 Home","🔬 Screening","📊 Result","🔍 AI Explanation","ℹ️ Model Info","📖 About"])

    # ── TAB 1: HOME ───────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("""
        <div style='text-align:center;padding:40px 0 30px;'>
            <div style='font-size:4rem;margin-bottom:10px;'>👁️</div>
            <h1 style='font-size:2.6rem;font-weight:800;
                background:linear-gradient(90deg,#64b5f6,#42a5f5,#1e88e5);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
                AI Retinal Screening</h1>
            <p style='font-size:1.15rem;color:#7a9ab0;max-width:600px;margin:0 auto;'>
                AI-Assisted Retinal Image Screening System — B.Tech Final Year Project</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='disclaimer'>⚠️ <b>Important Medical Disclaimer</b> — This application is an AI-assisted screening prototype and is <b>not</b> a substitute for examination or diagnosis by a qualified ophthalmologist. All results must be interpreted by a licensed medical professional.</div>", unsafe_allow_html=True)
        st.markdown("---")

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("""<div class='card'><div class='section-title'>🔬 How It Works</div><br>
            <ol style='color:#b0c8e0;line-height:2.0;padding-left:20px;'>
            <li>Upload a retinal fundus image (JPG/PNG)</li>
            <li>The system checks image quality</li>
            <li>EfficientNetB0 analyses the image</li>
            <li>A prediction with confidence is shown</li>
            <li>Grad-CAM highlights influential regions</li>
            <li>A screening recommendation is provided</li>
            </ol></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class='card'><div class='section-title'>🏥 Disease Categories</div><br>
            <table style='width:100%;color:#b0c8e0;border-collapse:collapse;'>
            <tr><td style='padding:8px;'>✅ <b>Normal</b></td><td style='color:#7a9ab0;'>No signs of ocular disease</td></tr>
            <tr><td style='padding:8px;'>🩸 <b>Diabetes</b></td><td style='color:#7a9ab0;'>Diabetic retinopathy indicators</td></tr>
            <tr><td style='padding:8px;'>👁 <b>Glaucoma</b></td><td style='color:#7a9ab0;'>Optic nerve / pressure signs</td></tr>
            <tr><td style='padding:8px;'>🌅 <b>AMD</b></td><td style='color:#7a9ab0;'>Age-related macular degeneration</td></tr>
            </table></div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🌍 Problem Being Solved")
        pc = st.columns(3, gap="medium")
        for col, (icon, title, desc) in zip(pc, [
            ("👨‍⚕️","Specialist Shortage","Ophthalmologists are scarce in rural and developing regions"),
            ("⏱️","Delayed Diagnosis","Manual screening is slow; diseases are often detected late"),
            ("🌐","Accessibility","High cost of eye exams limits access for lower-income populations"),
        ]):
            col.markdown(f"""<div class='metric-card' style='height:160px;'>
            <div style='font-size:2.2rem;'>{icon}</div>
            <div style='font-weight:700;color:#e0eaff;margin:10px 0 6px;'>{title}</div>
            <div style='font-size:0.85rem;color:#7a9ab0;line-height:1.5;'>{desc}</div>
            </div>""", unsafe_allow_html=True)
        _footer()

    # ── TAB 2: SCREENING ──────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("## 🔬 Retinal Image Screening")
        st.markdown("<div class='info-box'>Upload a <b>retinal fundus photograph</b> (JPG/JPEG/PNG). The image should be a clear, colour fundus image.</div>", unsafe_allow_html=True)

        cu, cp = st.columns(2, gap="large")
        with cu:
            st.markdown("### 📁 Upload Image")
            uploaded = st.file_uploader("Choose a retinal fundus image",
                                        type=["jpg","jpeg","png"], key="uploaded_file")
            if uploaded:
                try:
                    pil_img = PILImage.open(uploaded).convert("RGB")
                    st.session_state["uploaded_pil"] = pil_img
                    st.session_state["result"]       = None
                    st.success(f"✅ {uploaded.name}  ({pil_img.size[0]}×{pil_img.size[1]} px)")
                    ok, msg = check_image_quality(pil_img)
                    if ok: st.success(f"🟢 Quality OK: {msg}")
                    else:  st.warning(f"🔴 Quality: {msg}")
                except Exception as exc:
                    st.error(f"❌ {exc}")
                    st.session_state["uploaded_pil"] = None

        with cp:
            st.markdown("### 🖼 Image Preview")
            if st.session_state.get("uploaded_pil"):
                st.image(st.session_state["uploaded_pil"],
                         caption="Uploaded fundus image", use_container_width=True)
            else:
                st.markdown("""<div style='border:2px dashed #2a3f55;border-radius:12px;
                height:300px;display:flex;align-items:center;justify-content:center;
                color:#3a5a6e;flex-direction:column;'>
                <div style='font-size:3rem;'>👁️</div>
                <div style='margin-top:10px;'>Image preview will appear here</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        _, cb, _ = st.columns([1,2,1])
        with cb:
            analyze = st.button("🔍 Analyze Retina", key="analyze_btn", type="primary")

        if analyze:
            if not st.session_state.get("uploaded_pil"):
                st.error("Please upload an image first.")
            else:
                predictor = _get_predictor()
                if not predictor or not predictor.is_ready():
                    st.error("⚠️ Model not found. Run: `python retinal_ai_complete.py --train`")
                else:
                    with st.spinner("🤖 Analysing retinal image …"):
                        res = predictor.predict(st.session_state["uploaded_pil"], run_gradcam=True)
                    st.session_state["result"] = res
                    if res.get("error"):
                        st.warning(f"⚠️ {res['error']}") if not res.get("quality_ok") else st.error(f"❌ {res['error']}")
                    else:
                        st.success(f"✅ Predicted: **{res['prediction']}** ({res['confidence']*100:.1f}% confidence)")
                        st.info("See the **Result** and **AI Explanation** tabs for details.")
        _footer()

    # ── TAB 3: RESULT ─────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("## 📊 Screening Result")
        res = st.session_state.get("result")
        if res is None:
            st.info("No result yet. Go to the **Screening** tab and analyse an image.")
        elif res.get("error") and not res.get("prediction"):
            st.error(f"Analysis failed: {res['error']}")
        else:
            pred = res["prediction"]; conf = res["confidence"]
            probs = res["probabilities"]; low_c = res["low_confidence"]

            st.markdown(f"""
            <div class='result-card {_css_class(pred)}'>
                <div style='font-size:3.5rem;margin-bottom:10px;'>{_emoji(pred)}</div>
                <div style='font-size:1.2rem;color:#a0b8c8;margin-bottom:4px;'>Predicted Condition</div>
                <div style='font-size:2.4rem;font-weight:800;color:#e0eaff;margin-bottom:12px;'>
                    Possible {pred}</div>
                <div style='font-size:1rem;color:#a0b8c8;margin-bottom:8px;'>Model Confidence</div>
                <div style='font-size:3rem;font-weight:800;color:{_conf_color(conf)};'>
                    {conf*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

            if low_c:
                st.warning("⚠️ **Low-confidence prediction.** The model is uncertain. Please seek professional evaluation.")

            recs = {
                "Normal":   ("🟢","No immediate pathology detected. Routine annual eye exam recommended."),
                "Diabetes": ("🔴","Possible diabetic retinopathy. Urgent ophthalmologist evaluation recommended."),
                "Glaucoma": ("🔴","Possible glaucoma. Immediate IOP measurement and full evaluation recommended."),
                "AMD":      ("🟡","Possible AMD. Specialist OCT imaging assessment recommended."),
            }
            icon, rec = recs.get(pred, ("ℹ️","Consult a specialist."))
            st.markdown(f"""<div class='card'>
            <div style='font-size:1.05rem;font-weight:600;color:#b0c8e0;margin-bottom:8px;'>
                {icon} Screening Recommendation</div>
            <div style='color:#d0e0f0;line-height:1.7;'>{rec}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📈 Class Probability Distribution")
            cn   = list(probs.keys())
            pv   = [probs[c]*100 for c in cn]
            cols = {"Normal":"#4CAF50","Diabetes":"#F44336","Glaucoma":"#2196F3","AMD":"#FF9800"}
            fig  = go.Figure(go.Bar(
                x=cn, y=pv,
                marker_color=[cols.get(c,"#78909C") for c in cn],
                text=[f"{v:.1f}%" for v in pv], textposition="outside", textfont_size=13))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,25,40,0.7)",
                font=dict(color="#b0c8e0",family="Inter"),
                title=dict(text="Model Probability per Class",font_size=15,x=0.5),
                yaxis=dict(title="Probability (%)",range=[0,110],
                           gridcolor="rgba(42,63,85,0.5)"),
                xaxis=dict(title="Condition"), margin=dict(t=60,b=40,l=40,r=40), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<div class='disclaimer'>⚠️ <b>Important</b> — The prediction above is generated by an AI model trained on the ODIR-5K research dataset. It is a <b>screening indicator only</b> and must <b>not</b> be treated as a clinical diagnosis. All findings must be confirmed by a qualified ophthalmologist.</div>", unsafe_allow_html=True)
        _footer()

    # ── TAB 4: AI EXPLANATION ─────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("## 🔍 AI Explanation (Grad-CAM)")
        st.markdown("<div class='info-box'><b>Gradient-weighted Class Activation Mapping (Grad-CAM)</b> highlights the regions of the retinal image that most influenced the model's prediction. Warmer colours (red/yellow) = higher influence.</div>", unsafe_allow_html=True)

        res = st.session_state.get("result")
        if res is None:
            st.info("No result yet. Go to the **Screening** tab first.")
        else:
            gc   = res.get("gradcam")
            pred = res.get("prediction","Unknown")
            if not gc or not gc.get("success"):
                st.warning(f"⚠️ Grad-CAM unavailable: {gc.get('message','') if gc else 'No data.'}")
            else:
                c1, c2, c3 = st.columns(3, gap="medium")
                with c1:
                    st.markdown("##### 🖼 Original Image")
                    if gc.get("original"): st.image(gc["original"], caption="Preprocessed 224×224", use_container_width=True)
                with c2:
                    st.markdown("##### 🌡 Grad-CAM Heatmap")
                    if gc.get("heatmap") is not None: st.image(gc["heatmap"], caption="Activation heatmap", use_container_width=True)
                with c3:
                    st.markdown("##### 🔀 Overlay")
                    if gc.get("overlay"): st.image(gc["overlay"], caption=f"Overlaid — {pred}", use_container_width=True)

                st.markdown("---")
                st.markdown(f"""<div class='card'>
                <div style='font-size:1.05rem;font-weight:600;color:#64b5f6;margin-bottom:10px;'>🧠 What This Shows</div>
                <p style='color:#b0c8e0;line-height:1.8;'>The heatmap shows regions influencing the <b>{pred}</b> prediction. Red/yellow = high influence. Blue = low influence.</p>
                <div style='margin-top:14px;padding:12px;background:#1a2a1a;border-radius:8px;
                            border-left:4px solid #f39c12;color:#f0c060;'>
                    ⚠️ <b>Disclaimer:</b> {gc['message']}</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("##### 🎨 Colour Legend")
                st.markdown("""<div style='display:flex;gap:20px;margin-top:6px;'>
                <div style='display:flex;align-items:center;gap:8px;'>
                    <div style='width:20px;height:20px;background:#FF0000;border-radius:4px;'></div>
                    <span style='color:#b0c8e0;font-size:0.85rem;'>Highest influence</span></div>
                <div style='display:flex;align-items:center;gap:8px;'>
                    <div style='width:20px;height:20px;background:#FFFF00;border-radius:4px;'></div>
                    <span style='color:#b0c8e0;font-size:0.85rem;'>High influence</span></div>
                <div style='display:flex;align-items:center;gap:8px;'>
                    <div style='width:20px;height:20px;background:#0000FF;border-radius:4px;'></div>
                    <span style='color:#b0c8e0;font-size:0.85rem;'>Low influence</span></div>
                </div>""", unsafe_allow_html=True)
        _footer()

    # ── TAB 5: MODEL INFO ─────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("## ℹ️ Model Information")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("### 🏗 Architecture")
            st.markdown(f"""<div class='card'><table style='width:100%;color:#b0c8e0;border-collapse:collapse;'>
            <tr><td style='padding:8px;color:#64b5f6;'><b>Base Model</b></td><td style='padding:8px;'>{BASE_MODEL_NAME}</td></tr>
            <tr><td style='padding:8px;color:#64b5f6;'><b>Pretrained Weights</b></td><td style='padding:8px;'>ImageNet</td></tr>
            <tr><td style='padding:8px;color:#64b5f6;'><b>Input Shape</b></td><td style='padding:8px;'>{INPUT_SHAPE}</td></tr>
            <tr><td style='padding:8px;color:#64b5f6;'><b>Dense Units</b></td><td style='padding:8px;'>{DENSE_UNITS}</td></tr>
            <tr><td style='padding:8px;color:#64b5f6;'><b>Dropout Rate</b></td><td style='padding:8px;'>{DROPOUT_RATE}</td></tr>
            <tr><td style='padding:8px;color:#64b5f6;'><b>Output Classes</b></td><td style='padding:8px;'>{len(TARGET_CLASSES)}</td></tr>
            <tr><td style='padding:8px;color:#64b5f6;'><b>Loss Function</b></td><td style='padding:8px;'>Sparse Categorical Crossentropy</td></tr>
            </table></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("### 📋 Training Configuration")
            st.markdown(f"""<div class='card'><table style='width:100%;color:#b0c8e0;border-collapse:collapse;'>
            <tr><td style='padding:8px;color:#64b5f6;'><b>Dataset</b></td><td style='padding:8px;'>ODIR-5K (Hugging Face)</td></tr>
            <tr><td style='padding:8px;color:#64b5f6;'><b>Classes</b></td><td style='padding:8px;'>Normal, Diabetes, Glaucoma, AMD</td></tr>
            <tr><td style='padding:8px;color:#64b5f6;'><b>Batch Size</b></td><td style='padding:8px;'>{BATCH_SIZE}</td></tr>
            <tr><td style='padding:8px;color:#64b5f6;'><b>Head Epochs</b></td><td style='padding:8px;'>{EPOCHS_HEAD}</td></tr>
            <tr><td style='padding:8px;color:#64b5f6;'><b>Fine-tune Epochs</b></td><td style='padding:8px;'>{EPOCHS_FINETUNE}</td></tr>
            <tr><td style='padding:8px;color:#64b5f6;'><b>Split Ratio</b></td><td style='padding:8px;'>70% / 15% / 15%</td></tr>
            <tr><td style='padding:8px;color:#64b5f6;'><b>Splitting</b></td><td style='padding:8px;'>Patient-level (no leakage)</td></tr>
            <tr><td style='padding:8px;color:#64b5f6;'><b>Explainability</b></td><td style='padding:8px;'>Grad-CAM</td></tr>
            </table></div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📊 Training Results")
        meta = _safe_json(TRAINING_META_PATH)
        if meta:
            mc = st.columns(4, gap="medium")
            for col, (label, val) in zip(mc, [
                ("🎯 Best Val Accuracy", f"{meta.get('best_val_accuracy',0)*100:.1f}%"),
                ("📉 Best Val Loss",     f"{meta.get('best_val_loss',0):.4f}"),
                ("🏋 Train Samples",    f"{meta.get('train_samples','?'):,}" if isinstance(meta.get('train_samples'), int) else "?"),
                ("⏱ Training Time",    f"{meta.get('training_duration_s',0)/60:.0f} min"),
            ]):
                col.markdown(f"""<div class='metric-card'>
                <div class='big-metric' style='color:#64b5f6;'>{val}</div>
                <div class='metric-label'>{label}</div></div>""", unsafe_allow_html=True)
        else:
            st.info("Training metadata not available. Run: `python retinal_ai_complete.py --train`")

        st.markdown("### 🏆 Evaluation Metrics (Test Set)")
        ev = _safe_json(EVAL_METRICS_PATH)
        if ev:
            ec = st.columns(4, gap="medium")
            for col, (label, val, color) in zip(ec, [
                ("✅ Accuracy",    f"{ev.get('accuracy',0)*100:.1f}%",   "#4CAF50"),
                ("📏 Macro F1",    f"{ev.get('macro_f1',0)*100:.1f}%",   "#2196F3"),
                ("🔍 Macro Recall",f"{ev.get('macro_recall',0)*100:.1f}%","#FF9800"),
                ("🔑 ROC-AUC",     f"{ev.get('roc_auc_macro') or '—'}",  "#9C27B0"),
            ]):
                col.markdown(f"""<div class='metric-card'>
                <div class='big-metric' style='color:{color};'>{val}</div>
                <div class='metric-label'>{label}</div></div>""", unsafe_allow_html=True)

            report = ev.get("per_class_report", {})
            if report:
                import pandas as pd
                rows = []
                for cls in TARGET_CLASSES:
                    r = report.get(cls, {})
                    rows.append({"Class":cls,
                                 "Precision":f"{r.get('precision',0)*100:.1f}%",
                                 "Recall":   f"{r.get('recall',0)*100:.1f}%",
                                 "F1-Score": f"{r.get('f1-score',0)*100:.1f}%",
                                 "Support":  int(r.get("support",0))})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Evaluation metrics not available. Run: `python retinal_ai_complete.py --evaluate`")

        st.markdown("---")
        st.markdown("### 📈 Training History Plots")
        pc = st.columns(3, gap="medium")
        for col, (path, title) in zip(pc, [
            (TRAINING_ACC_PATH,    "Training Accuracy"),
            (TRAINING_LOSS_PATH,   "Training Loss"),
            (CONFUSION_MATRIX_PATH,"Confusion Matrix"),
        ]):
            if Path(path).exists():
                col.image(str(path), caption=title, use_container_width=True)
            else:
                col.info(f"{title} not yet available.")
        _footer()

    # ── TAB 6: ABOUT ──────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("## 📖 About This Project")
        st.markdown("""<div class='card'><div class='section-title'>🎓 B.Tech Final Year Project</div><br>
        <p style='color:#b0c8e0;line-height:1.8;font-size:1.02rem;'>
        This system was developed as a final-year B.Tech project demonstrating how AI can assist in
        ophthalmic screening. It uses the <b>ODIR-5K dataset</b>, transfer learning on
        <b>EfficientNetB0</b>, and <b>Grad-CAM explainability</b> to create an end-to-end retinal
        disease screening prototype.</p></div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("### ✅ Key Features")
            for f in ["Transfer learning (EfficientNetB0)","Four-class ophthalmic screening",
                      "Patient-level splitting (no leakage)","Class imbalance handling",
                      "Grad-CAM visual explainability","Image quality assessment",
                      "Confidence-based uncertainty warning","Interactive Streamlit dashboard",
                      "Medical disclaimer integration"]:
                st.markdown(f"✅ {f}")
        with col2:
            st.markdown("### ⚠️ Known Limitations")
            for lim in ["Trained on ODIR-5K only — may not generalise to all cameras",
                        "Four-class prototype — not all eye diseases covered",
                        "AI confidence ≠ diagnostic certainty",
                        "Image quality affects prediction reliability",
                        "Not validated in a clinical trial",
                        "Should not replace specialist examination"]:
                st.markdown(f"⚠️ {lim}")

        st.markdown("---")
        st.markdown("### 🚀 Future Scope")
        fc = st.columns(3, gap="medium")
        for i, (icon, title, desc) in enumerate([
            ("🔬","Multi-label classification","Screen for all 8 ODIR categories"),
            ("📱","Mobile deployment","Android/iOS app for field use"),
            ("🌍","Multi-language support","Local languages for rural healthcare"),
            ("🧪","Clinical validation","Partnership with hospitals"),
            ("📊","Longitudinal tracking","Track disease progression over time"),
            ("🔗","EHR integration","Connect with health record systems"),
        ]):
            fc[i % 3].markdown(f"""<div class='metric-card' style='margin-bottom:16px;height:150px;'>
            <div style='font-size:2rem;'>{icon}</div>
            <div style='font-weight:700;color:#e0eaff;margin:8px 0 4px;'>{title}</div>
            <div style='font-size:0.8rem;color:#7a9ab0;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div class='disclaimer'>⚠️ <b>Full Medical Disclaimer</b> — This application is an AI-assisted screening prototype for educational and research purposes. It does not provide medical diagnoses, treatment recommendations, or prescriptions. Retinal images and any AI-generated predictions must be reviewed and interpreted by a qualified, licensed ophthalmologist before any clinical decision is made. The developers and institution assume no medical or legal liability for outcomes based on this tool's predictions.</div>", unsafe_allow_html=True)
        _footer()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _is_streamlit() -> bool:
    """Detect if we're running inside Streamlit."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        try:
            import streamlit as st
            return hasattr(st, '_is_running_with_streamlit')
        except Exception:
            return False


if __name__ == "__main__":
    # ── Running via: streamlit run retinal_ai_complete.py ────────────────────
    # Streamlit calls __main__ with no args; we detect this via the ctx check.
    if _is_streamlit() or "streamlit" in sys.modules:
        _run_streamlit_app()

    else:
        # ── Running via: python retinal_ai_complete.py <args> ────────────────
        import argparse

        parser = argparse.ArgumentParser(
            description="AI Retinal Screening System — B.Tech Final Year Project",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser.add_argument("--dataset",  action="store_true", help="Inspect & label the ODIR-5K dataset")
        parser.add_argument("--train",    action="store_true", help="Train the EfficientNetB0 model")
        parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained model on test set")
        parser.add_argument("--predict",  type=str,            help="Predict on a single image (path)")
        parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM during prediction")
        args = parser.parse_args()

        if args.dataset:
            print_section("AI RETINAL SCREENING — DATASET INSPECTION")
            ds = load_hf_dataset()
            inspect_dataset(ds)
            df_meta = build_dataframe(ds)
            df_labelled, stats = assign_four_class_labels(df_meta)
            ensure_dir(METRICS_DIR)
            save_json(stats, METRICS_DIR / "labeling_stats.json")
            save_metadata_csv(df_labelled)
            plot_class_distribution(df_labelled, save=True)
            print_section("Done")
            print("  Next: python retinal_ai_complete.py --train")

        elif args.train:
            run_training()

        elif args.evaluate:
            run_evaluation()

        elif args.predict:
            run_predict_cli(args.predict, no_gradcam=args.no_gradcam)

        else:
            # No args → show help
            parser.print_help()
            print("\n  ─────────────────────────────────────────────────────")
            print("  Quick Start:")
            print("    1. python retinal_ai_complete.py --dataset")
            print("    2. python retinal_ai_complete.py --train")
            print("    3. python retinal_ai_complete.py --evaluate")
            print("    4. streamlit run retinal_ai_complete.py")
            print("  ─────────────────────────────────────────────────────")

# ── If launched by Streamlit (not __main__) ───────────────────────────────────
else:
    # When Streamlit imports this file, __name__ == "retinal_ai_complete"
    # We detect this and render the UI.
    try:
        import streamlit as st
        _run_streamlit_app()
    except Exception:
        pass
