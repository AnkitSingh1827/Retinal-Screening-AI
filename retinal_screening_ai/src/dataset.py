"""
src/dataset.py
==============
Dataset loading and inspection for the ODIR-5K retinal fundus dataset
from Hugging Face (bumbledeep/odir).

Run directly for a full dataset report:
    python -m src.dataset

What this module does
─────────────────────
1. Loads the HF dataset (with optional caching).
2. Inspects the real schema (columns, dtypes, number of records).
3. Reads sample rows and prints label distributions.
4. Builds a pandas DataFrame with patient_id, label, label_code.
5. Assigns each record to one of four target classes using the
   keyword-matching policy defined in config.py.
6. Reports how many multi-label records were excluded (if policy = "exclude").
7. Saves the processed DataFrame to data/splits/full_dataset.csv.
8. Saves class distribution plots to reports/figures/.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path so "config" and "src" imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.config import (
    DATASET_NAME,
    DATASET_SPLIT,
    FIGURES_DIR,
    LABEL_KEYWORD_MAP,
    MULTI_LABEL_POLICY,
    SPLITS_DIR,
    TARGET_CLASSES,
)
from src.utils import ensure_dir, get_logger, print_class_distribution, print_section, save_json

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# 1.  LOAD FROM HUGGING FACE
# ─────────────────────────────────────────────────────────

def load_hf_dataset(streaming: bool = False):
    """
    Load the bumbledeep/odir dataset from Hugging Face.

    Parameters
    ----------
    streaming : bool
        If True, use streaming mode (no full download).
        Useful for quick inspection on limited bandwidth.

    Returns
    -------
    dataset : datasets.Dataset  (or IterableDataset if streaming=True)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("'datasets' package not found. Install: pip install datasets")
        sys.exit(1)

    logger.info(f"Loading dataset '{DATASET_NAME}' (split='{DATASET_SPLIT}', streaming={streaming}) …")
    try:
        ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=streaming)
        logger.info("Dataset loaded successfully.")
        return ds
    except Exception as exc:
        logger.error(f"Failed to load dataset: {exc}")
        logger.info("Troubleshooting tips:")
        logger.info("  • Check internet connection")
        logger.info("  • Run: pip install --upgrade datasets huggingface_hub")
        logger.info(f"  • Dataset URL: https://huggingface.co/datasets/{DATASET_NAME}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────
# 2.  INSPECT DATASET STRUCTURE
# ─────────────────────────────────────────────────────────

def inspect_dataset(ds) -> None:
    """Print a detailed report of the dataset schema and sample records."""
    print_section("DATASET INSPECTION REPORT")
    print(f"  Dataset identifier : {DATASET_NAME}")
    print(f"  Split used         : {DATASET_SPLIT}")

    # ── Schema ──────────────────────────────────────────
    print_section("Column Schema")
    try:
        features = ds.features
        for col, feat in features.items():
            print(f"  {col:<20} {feat}")
    except AttributeError:
        print("  (streaming mode — features not directly available)")

    # ── Size ────────────────────────────────────────────
    print_section("Dataset Size")
    try:
        n = len(ds)
        print(f"  Total records: {n:,}")
    except TypeError:
        print("  (streaming mode — length unknown)")

    # ── Sample rows ─────────────────────────────────────
    print_section("Sample Records (first 3)")
    try:
        for i, row in enumerate(ds.select(range(min(3, len(ds))))):
            print(f"\n  Row {i}:")
            for k, v in row.items():
                if k == "image":
                    # PIL Image object
                    print(f"    {k:<15} PIL.Image size={v.size} mode={v.mode}")
                else:
                    print(f"    {k:<15} {repr(v)}")
    except Exception as e:
        logger.warning(f"Could not print sample rows: {e}")


# ─────────────────────────────────────────────────────────
# 3.  BUILD PANDAS DATAFRAME
# ─────────────────────────────────────────────────────────

def build_dataframe(ds) -> pd.DataFrame:
    """
    Convert the HF dataset to a pandas DataFrame containing only metadata
    (patient_id, age, sex, label, label_code).
    Images are NOT stored in the DataFrame — they are fetched on-the-fly
    during training to keep memory usage low.

    Returns
    -------
    df : pd.DataFrame
        One row per image record.
    """
    logger.info("Building metadata DataFrame …")
    try:
        # Fast path: use the built-in to_pandas() if available
        df = ds.to_pandas()
        # Drop the image column from the DataFrame (keep only metadata)
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
        df = pd.DataFrame(rows)
        logger.info(f"DataFrame shape: {df.shape}")
        return df


# ─────────────────────────────────────────────────────────
# 4.  FOUR-CLASS LABELING PIPELINE
# ─────────────────────────────────────────────────────────

def _match_label(label_text: str) -> List[str]:
    """
    Given a raw label string, return all target classes whose keywords
    appear in the label (case-insensitive).
    """
    if not isinstance(label_text, str):
        return []
    lower = label_text.lower()
    matched = []
    for cls, keywords in LABEL_KEYWORD_MAP.items():
        if any(kw in lower for kw in keywords):
            matched.append(cls)
    return matched


def assign_four_class_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Map each record's free-text 'label' column to one of the four target classes.

    Multi-label records (matching more than one target class) are handled
    according to MULTI_LABEL_POLICY in config.py:
      - "exclude" → drop those rows and report the count
      - "first"   → keep the first matching class (TARGET_CLASSES order)

    Returns
    -------
    df_filtered : pd.DataFrame  — rows with a unique class assignment
    stats       : dict          — labeling statistics
    """
    logger.info("Assigning four-class labels …")

    df = df.copy()
    df["matched_classes"] = df["label"].apply(_match_label)

    # Count categories
    no_match   = df["matched_classes"].apply(lambda x: len(x) == 0)
    single_match = df["matched_classes"].apply(lambda x: len(x) == 1)
    multi_match  = df["matched_classes"].apply(lambda x: len(x) > 1)

    n_no_match   = no_match.sum()
    n_single     = single_match.sum()
    n_multi      = multi_match.sum()
    n_total      = len(df)

    logger.info(f"  Total records       : {n_total:,}")
    logger.info(f"  No target match     : {n_no_match:,}  (will be excluded)")
    logger.info(f"  Single-class match  : {n_single:,}")
    logger.info(f"  Multi-class match   : {n_multi:,}  (policy='{MULTI_LABEL_POLICY}')")

    # Assign class column
    def _pick_class(matches):
        if len(matches) == 0:
            return None                       # no match → exclude
        if len(matches) == 1:
            return matches[0]                 # unambiguous
        # Multi-label
        if MULTI_LABEL_POLICY == "exclude":
            return "__multi__"                # sentinel → exclude below
        elif MULTI_LABEL_POLICY == "first":
            for cls in TARGET_CLASSES:        # respect ordering
                if cls in matches:
                    return cls
        return None

    df["class_name"] = df["matched_classes"].apply(_pick_class)

    # Filter: keep only rows with a valid single class
    df_valid = df[df["class_name"].isin(TARGET_CLASSES)].copy()
    n_kept   = len(df_valid)
    n_excluded_no_match = n_no_match
    n_excluded_multi    = n_multi if MULTI_LABEL_POLICY == "exclude" else 0

    df_valid["class_index"] = df_valid["class_name"].apply(
        lambda c: TARGET_CLASSES.index(c)
    )

    stats = {
        "total_records":        int(n_total),
        "no_target_match":      int(n_no_match),
        "single_class_match":   int(n_single),
        "multi_class_match":    int(n_multi),
        "multi_label_policy":   MULTI_LABEL_POLICY,
        "excluded_no_match":    int(n_excluded_no_match),
        "excluded_multi_label": int(n_excluded_multi),
        "kept_records":         int(n_kept),
        "class_distribution":   df_valid["class_name"].value_counts().to_dict(),
    }

    logger.info(f"Records kept after filtering: {n_kept:,}")
    print_section("Four-Class Label Assignment Result")
    print(f"  Excluded (no match)         : {n_excluded_no_match:,}")
    print(f"  Excluded (multi-label)      : {n_excluded_multi:,}")
    print(f"  Final usable records        : {n_kept:,}")

    print_class_distribution(df_valid["class_name"].tolist(), TARGET_CLASSES)

    return df_valid, stats


# ─────────────────────────────────────────────────────────
# 5.  MISSING VALUE & DUPLICATE CHECK
# ─────────────────────────────────────────────────────────

def data_quality_report(df: pd.DataFrame) -> None:
    """Print missing values and duplicate records."""
    print_section("Data Quality Report")

    print("\n  Missing values per column:")
    nulls = df.isnull().sum()
    for col, n in nulls.items():
        print(f"    {col:<20} {n}")

    dupes = df.duplicated(subset=["patient_id"]).sum() if "patient_id" in df.columns else 0
    print(f"\n  Duplicate patient_ids: {dupes}")


# ─────────────────────────────────────────────────────────
# 6.  VISUALISE CLASS DISTRIBUTION
# ─────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, save: bool = True) -> None:
    """
    Bar chart of class distribution.
    Saves to reports/figures/class_distribution.png if save=True.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed. Skipping plot.")
        return

    counts = df["class_name"].value_counts().reindex(TARGET_CLASSES, fill_value=0)

    palette = ["#4CAF50", "#FF5722", "#2196F3", "#FF9800"]  # green/red/blue/orange
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=palette, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("ODIR-5K — Four-Class Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    sns.despine()
    plt.tight_layout()

    if save:
        ensure_dir(FIGURES_DIR)
        out = FIGURES_DIR / "class_distribution.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"Class distribution plot saved → {out}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────
# 7.  SAVE PROCESSED METADATA CSV
# ─────────────────────────────────────────────────────────

def save_metadata_csv(df: pd.DataFrame) -> Path:
    """Save the labelled metadata DataFrame to data/splits/full_dataset.csv."""
    ensure_dir(SPLITS_DIR)
    out = SPLITS_DIR / "full_dataset.csv"
    cols = [c for c in ["patient_id", "age", "sex", "label", "class_name", "class_index"] if c in df.columns]
    df[cols].to_csv(out, index=False)
    logger.info(f"Metadata CSV saved → {out}")
    return out


# ─────────────────────────────────────────────────────────
# 8.  MAIN — run as   python -m src.dataset
# ─────────────────────────────────────────────────────────

def main():
    print_section("AI RETINAL SCREENING — DATASET INSPECTION")

    # Step 1: Load
    ds = load_hf_dataset(streaming=False)

    # Step 2: Inspect
    inspect_dataset(ds)

    # Step 3: Build DataFrame
    df_meta = build_dataframe(ds)

    # Step 4: Quality report
    data_quality_report(df_meta)

    # Step 5: Assign four-class labels
    df_labelled, stats = assign_four_class_labels(df_meta)

    # Step 6: Save stats
    ensure_dir(FIGURES_DIR.parent)
    save_json(stats, FIGURES_DIR.parent / "labeling_stats.json")

    # Step 7: Save CSV
    save_metadata_csv(df_labelled)

    # Step 8: Plot
    plot_class_distribution(df_labelled, save=True)

    print_section("Dataset inspection complete")
    print("  Next step: python -m src.train")


if __name__ == "__main__":
    main()
