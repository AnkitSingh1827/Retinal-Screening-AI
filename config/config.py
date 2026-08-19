"""
config/config.py
================
Central configuration for the AI Retinal Screening System.
All paths, hyperparameters, and toggles are defined here.
Change values here to experiment — you don't need to touch other source files.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────
# 1. ROOT PATHS  (all paths are relative to project root)
# ─────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent   # project root
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR    = DATA_DIR / "splits"
MODELS_DIR    = ROOT_DIR / "models"
REPORTS_DIR   = ROOT_DIR / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"
METRICS_DIR   = REPORTS_DIR / "metrics"
ASSETS_DIR    = ROOT_DIR / "assets"

# ─────────────────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────────────────
DATASET_NAME  = "bumbledeep/odir"           # Hugging Face dataset identifier
DATASET_SPLIT = "train"                     # only split available in this HF dataset

# ─────────────────────────────────────────────────────────
# 3. FOUR-CLASS PROTOTYPE LABELS
#    Mapping from the free-text 'label' column → one of our four classes.
#    Keywords are matched case-insensitively (substring match).
# ─────────────────────────────────────────────────────────
TARGET_CLASSES = ["Normal", "Diabetes", "Glaucoma", "AMD"]

# Keywords that map to each target class.
# The pipeline reads the 'label' column and checks for these substrings.
LABEL_KEYWORD_MAP = {
    "Normal":   ["normal"],
    "Diabetes": ["diabetes", "diabetic retinopathy", "dr"],
    "Glaucoma": ["glaucoma"],
    "AMD":      ["age-related macular degeneration", "amd", "macular degeneration"],
}

# How to handle images that match MULTIPLE target classes:
#   "exclude" → remove them from the dataset and report the count
#   "first"   → assign the first matched class (in TARGET_CLASSES order)
MULTI_LABEL_POLICY = "exclude"

# ─────────────────────────────────────────────────────────
# 4. IMAGE SETTINGS
# ─────────────────────────────────────────────────────────
IMAGE_SIZE    = (224, 224)      # (height, width) fed to the model
IMAGE_CHANNELS = 3              # RGB
INPUT_SHAPE   = IMAGE_SIZE + (IMAGE_CHANNELS,)  # (224, 224, 3)

# ─────────────────────────────────────────────────────────
# 5. DATA SPLIT RATIOS  (must sum to 1.0)
# ─────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ─────────────────────────────────────────────────────────
# 6. REPRODUCIBILITY
# ─────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────
# 7. MODEL / TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────
BASE_MODEL_NAME   = "EfficientNetB0"   # or "MobileNetV2" / "ResNet50"
PRETRAINED_WEIGHTS = "imagenet"

BATCH_SIZE        = 32
EPOCHS_HEAD       = 20        # phase 1: frozen backbone, train head only
EPOCHS_FINETUNE   = 10        # phase 2: unfreeze top layers and fine-tune
LEARNING_RATE_HEAD      = 1e-3
LEARNING_RATE_FINETUNE  = 1e-5

DROPOUT_RATE      = 0.40
DENSE_UNITS       = 256       # units in the fully-connected layer before softmax

# Number of top layers to unfreeze during fine-tuning
FINETUNE_LAYERS   = 20        # last N layers of the backbone

# ─────────────────────────────────────────────────────────
# 8. CALLBACKS
# ─────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE  = 7
REDUCE_LR_PATIENCE       = 3
REDUCE_LR_FACTOR         = 0.5
REDUCE_LR_MIN            = 1e-7

# ─────────────────────────────────────────────────────────
# 9. MODEL SAVE PATHS
# ─────────────────────────────────────────────────────────
BEST_MODEL_PATH       = MODELS_DIR / "best_model.keras"
CLASS_NAMES_PATH      = MODELS_DIR / "class_names.json"
TRAINING_META_PATH    = MODELS_DIR / "training_metadata.json"

# ─────────────────────────────────────────────────────────
# 10. PREDICTION / UI SETTINGS
# ─────────────────────────────────────────────────────────
# If the model confidence is below this threshold, warn the user.
LOW_CONFIDENCE_THRESHOLD = 0.60

# Minimum image dimension (pixels) accepted for prediction
MIN_IMAGE_DIMENSION = 100

# ─────────────────────────────────────────────────────────
# 11. GRAD-CAM SETTINGS
# ─────────────────────────────────────────────────────────
GRADCAM_ALPHA       = 0.4      # transparency of the heatmap overlay
GRADCAM_COLORMAP    = "jet"    # matplotlib colormap for heatmap
# Layer to target for Grad-CAM in EfficientNetB0
GRADCAM_LAYER_NAME  = "top_conv"

# ─────────────────────────────────────────────────────────
# 12. REPORT PATHS
# ─────────────────────────────────────────────────────────
CONFUSION_MATRIX_PATH   = FIGURES_DIR / "confusion_matrix.png"
TRAINING_ACC_PATH       = FIGURES_DIR / "training_accuracy.png"
TRAINING_LOSS_PATH      = FIGURES_DIR / "training_loss.png"
CLASSIFICATION_RPT_PATH = METRICS_DIR / "classification_report.json"
EVAL_METRICS_PATH       = METRICS_DIR / "evaluation_metrics.json"

# ─────────────────────────────────────────────────────────
# 13. AUGMENTATION SETTINGS (training only)
# ─────────────────────────────────────────────────────────
AUGMENTATION_CONFIG = {
    "rotation_range":       10,     # degrees  (small — preserves optic disc geometry)
    "width_shift_range":    0.05,   # fraction of width
    "height_shift_range":   0.05,   # fraction of height
    "zoom_range":           0.10,   # fraction
    "horizontal_flip":      True,   # acceptable for fundus images
    "brightness_range":     [0.85, 1.15],
    "fill_mode":            "reflect",
}
