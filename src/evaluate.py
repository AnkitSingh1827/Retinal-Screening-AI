"""
src/evaluate.py
===============
Comprehensive model evaluation on the held-out test set.

Run with:
    python -m src.evaluate

Computes
────────
  • Accuracy
  • Precision, Recall, F1-score (per class and macro average)
  • Confusion matrix
  • Full classification report
  • ROC-AUC (one-vs-rest, if sklearn is available)

Saves
─────
  reports/figures/confusion_matrix.png
  reports/metrics/classification_report.json
  reports/metrics/evaluation_metrics.json
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    CONFUSION_MATRIX_PATH,
    FIGURES_DIR,
    IMAGE_SIZE,
    METRICS_DIR,
    TARGET_CLASSES,
    CLASSIFICATION_RPT_PATH,
    EVAL_METRICS_PATH,
)
from src.utils import (
    ensure_dir,
    get_logger,
    load_class_names,
    load_json,
    print_section,
    save_json,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# 1.  LOAD TRAINED MODEL
# ─────────────────────────────────────────────────────────

def load_model():
    """Load the best saved Keras model."""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {BEST_MODEL_PATH}.\n"
            "Run: python -m src.train"
        )
    try:
        from tensorflow import keras
        logger.info(f"Loading model from {BEST_MODEL_PATH} …")
        model = keras.models.load_model(str(BEST_MODEL_PATH))
        logger.info("Model loaded successfully.")
        return model
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        raise


# ─────────────────────────────────────────────────────────
# 2.  LOAD TEST DATA
# ─────────────────────────────────────────────────────────

def load_test_data():
    """
    Re-load the HF dataset and build the test set numpy arrays.
    Uses the pre-saved test.csv from data/splits/.
    """
    from src.dataset import load_hf_dataset, assign_four_class_labels, build_dataframe
    from src.preprocessing import load_splits
    from config.config import SPLITS_DIR

    test_csv = SPLITS_DIR / "test.csv"
    if not test_csv.exists():
        logger.info("Test CSV not found. Re-running dataset pipeline …")
        ds = load_hf_dataset()
        df_meta = build_dataframe(ds)
        df_labelled, _ = assign_four_class_labels(df_meta)
        from src.preprocessing import split_by_patient, save_splits
        _, _, df_test = split_by_patient(df_labelled)
        save_splits(_, _, df_test)  # partial save
    else:
        df_test = pd.read_csv(test_csv)
        ds = load_hf_dataset()

    from src.preprocessing import build_numpy_arrays
    logger.info(f"Building test arrays for {len(df_test)} samples …")
    X_test, y_test = build_numpy_arrays(ds, df_test)
    return X_test, y_test


# ─────────────────────────────────────────────────────────
# 3.  RUN PREDICTIONS
# ─────────────────────────────────────────────────────────

def predict_test_set(model, X_test: np.ndarray):
    """Return predicted class indices and probability arrays."""
    logger.info("Running model predictions on test set …")
    probs = model.predict(X_test, batch_size=32, verbose=1)  # (N, num_classes)
    preds = np.argmax(probs, axis=1)
    return preds, probs


# ─────────────────────────────────────────────────────────
# 4.  COMPUTE METRICS
# ─────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray,
                    class_names: list) -> dict:
    """
    Compute all evaluation metrics.

    Returns a dict with accuracy, per-class precision/recall/F1,
    macro averages, and ROC-AUC.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.preprocessing import label_binarize

    acc = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro",     zero_division=0)
    macro_pre = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro",  zero_division=0)

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    # ROC-AUC (one-vs-rest)
    try:
        n_classes = len(class_names)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        roc_auc = roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")
    except Exception as e:
        logger.warning(f"ROC-AUC computation failed: {e}")
        roc_auc = None

    metrics = {
        "accuracy":      round(float(acc),       4),
        "macro_f1":      round(float(macro_f1),  4),
        "macro_precision": round(float(macro_pre), 4),
        "macro_recall":  round(float(macro_rec), 4),
        "roc_auc_macro": round(float(roc_auc), 4) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "per_class_report": report_dict,
        "class_names":   class_names,
        "n_test_samples": int(len(y_true)),
    }

    return metrics


# ─────────────────────────────────────────────────────────
# 5.  CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, class_names: list) -> None:
    """Save a formatted confusion matrix heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    ensure_dir(FIGURES_DIR)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label",    fontsize=12, labelpad=10)
    ax.set_ylabel("True Label",         fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
    logger.info(f"Confusion matrix saved → {CONFUSION_MATRIX_PATH}")
    plt.close()


# ─────────────────────────────────────────────────────────
# 6.  PRINT EVALUATION REPORT
# ─────────────────────────────────────────────────────────

def print_evaluation_report(metrics: dict) -> None:
    """Pretty-print the evaluation results."""
    print_section("MODEL EVALUATION RESULTS")
    print(f"\n  Test samples    : {metrics['n_test_samples']}")
    print(f"  Accuracy        : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1        : {metrics['macro_f1']*100:.2f}%")
    print(f"  Macro Precision : {metrics['macro_precision']*100:.2f}%")
    print(f"  Macro Recall    : {metrics['macro_recall']*100:.2f}%")
    if metrics.get("roc_auc_macro") is not None:
        print(f"  Macro ROC-AUC   : {metrics['roc_auc_macro']:.4f}")

    print_section("Per-class Metrics")
    report = metrics["per_class_report"]
    header = f"  {'Class':<12}  {'Precision':>10}  {'Recall':>10}  {'F1-score':>10}  {'Support':>10}"
    print(header)
    print("  " + "-" * 58)
    for cls in metrics["class_names"]:
        r = report.get(cls, {})
        print(
            f"  {cls:<12}  {r.get('precision', 0)*100:>9.1f}%  "
            f"{r.get('recall', 0)*100:>9.1f}%  "
            f"{r.get('f1-score', 0)*100:>9.1f}%  "
            f"{int(r.get('support', 0)):>10}"
        )
    print()

    # Emphasize recall (sensitivity) — critical for medical screening
    print("  ⚠  NOTE: For medical screening, RECALL (sensitivity) is the primary metric.")
    print("     High recall means fewer missed disease cases.")


# ─────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────

def evaluate():
    """Full evaluation pipeline."""
    print_section("AI RETINAL SCREENING — MODEL EVALUATION")

    # Load model
    model = load_model()

    # Load class names
    try:
        class_names = load_class_names(CLASS_NAMES_PATH)
    except FileNotFoundError:
        logger.warning("class_names.json not found. Using default TARGET_CLASSES.")
        class_names = TARGET_CLASSES

    # Load test data
    X_test, y_test = load_test_data()

    # Predict
    y_pred, probs = predict_test_set(model, X_test)

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, probs, class_names)

    # Print report
    print_evaluation_report(metrics)

    # Save
    ensure_dir(METRICS_DIR)
    save_json(metrics["per_class_report"], CLASSIFICATION_RPT_PATH)
    save_json(metrics, EVAL_METRICS_PATH)
    logger.info(f"Metrics saved → {METRICS_DIR}")

    # Plot confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(cm, class_names)

    print_section("Evaluation Complete")
    print(f"  Metrics   → {METRICS_DIR}")
    print(f"  Figures   → {FIGURES_DIR}")
    print(f"  Next step → streamlit run app.py")


if __name__ == "__main__":
    evaluate()
