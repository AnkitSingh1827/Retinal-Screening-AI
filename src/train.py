"""
src/train.py
============
Model building, training, and saving.

Run with:
    python -m src.train

Training phases
───────────────
Phase 1 (head training):
    Backbone frozen → train only the classification head.

Phase 2 (fine-tuning, optional):
    Unfreeze the top N layers of the backbone and train at a very small LR.

Outputs saved
─────────────
  models/best_model.keras
  models/class_names.json
  models/training_metadata.json
  reports/figures/training_accuracy.png
  reports/figures/training_loss.png
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Set seeds before TF import
from src.utils import set_seeds
set_seeds(42)

import numpy as np

from config.config import (
    BASE_MODEL_NAME,
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    DENSE_UNITS,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS_FINETUNE,
    EPOCHS_HEAD,
    FIGURES_DIR,
    FINETUNE_LAYERS,
    IMAGE_SIZE,
    INPUT_SHAPE,
    LEARNING_RATE_FINETUNE,
    LEARNING_RATE_HEAD,
    MODELS_DIR,
    PRETRAINED_WEIGHTS,
    RANDOM_SEED,
    REDUCE_LR_FACTOR,
    REDUCE_LR_MIN,
    REDUCE_LR_PATIENCE,
    TARGET_CLASSES,
    TRAINING_ACC_PATH,
    TRAINING_LOSS_PATH,
    TRAINING_META_PATH,
)
from src.utils import (
    compute_class_weights,
    ensure_dir,
    get_logger,
    print_section,
    save_json,
    save_class_names,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# 1.  TensorFlow SETUP
# ─────────────────────────────────────────────────────────

def _setup_tensorflow():
    """Configure TF before importing (memory growth, seed)."""
    import tensorflow as tf

    # Enable GPU memory growth to avoid OOM on small GPUs
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    tf.random.set_seed(RANDOM_SEED)
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"GPUs available: {len(gpus)}")
    return tf


# ─────────────────────────────────────────────────────────
# 2.  MODEL BUILDING
# ─────────────────────────────────────────────────────────

def build_model(num_classes: int = 4, trainable_backbone: bool = False):
    """
    Build the transfer-learning model.

    Architecture
    ────────────
    Input (224×224×3)
    → EfficientNetB0 backbone (ImageNet weights, frozen by default)
    → GlobalAveragePooling2D
    → Dense(DENSE_UNITS, relu)
    → Dropout(DROPOUT_RATE)
    → Dense(num_classes, softmax)

    Parameters
    ----------
    num_classes         : number of output classes
    trainable_backbone  : if True, backbone weights are trainable (fine-tune phase)
    """
    import tensorflow as tf
    from tensorflow import keras

    # ── Select backbone ──────────────────────────────────
    backbone_map = {
        "EfficientNetB0": keras.applications.EfficientNetB0,
        "MobileNetV2":    keras.applications.MobileNetV2,
        "ResNet50":       keras.applications.ResNet50,
    }

    if BASE_MODEL_NAME not in backbone_map:
        logger.warning(f"Unknown base model '{BASE_MODEL_NAME}'. Falling back to EfficientNetB0.")
        BackboneClass = keras.applications.EfficientNetB0
    else:
        BackboneClass = backbone_map[BASE_MODEL_NAME]

    logger.info(f"Building model: {BASE_MODEL_NAME} (trainable_backbone={trainable_backbone})")

    try:
        backbone = BackboneClass(
            include_top=False,
            weights=PRETRAINED_WEIGHTS,
            input_shape=INPUT_SHAPE,
        )
    except Exception as exc:
        logger.error(f"Failed to load backbone: {exc}")
        logger.info("Trying EfficientNetB0 without pretrained weights …")
        backbone = keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=INPUT_SHAPE,
        )

    backbone.trainable = trainable_backbone

    # ── Classification head ───────────────────────────────
    inputs  = keras.Input(shape=INPUT_SHAPE, name="retinal_input")
    x       = backbone(inputs, training=trainable_backbone)
    x       = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x       = keras.layers.Dense(DENSE_UNITS, activation="relu", name="dense_1")(x)
    x       = keras.layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="RetinalScreeningModel")

    total_params    = model.count_params()
    trainable_p     = sum(
        np.prod(v.shape) for v in model.trainable_variables
    )
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_p:,}")

    return model


def compile_model(model, learning_rate: float):
    """Compile the model with Adam optimizer."""
    import tensorflow as tf
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────────────────
# 3.  CALLBACKS
# ─────────────────────────────────────────────────────────

def get_callbacks(phase: str = "head"):
    """
    Return Keras callbacks for training.

    phase: "head" or "finetune"
    """
    from tensorflow import keras

    ensure_dir(MODELS_DIR)

    monitor = "val_accuracy"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN,
            verbose=1,
        ),
    ]
    return callbacks


# ─────────────────────────────────────────────────────────
# 4.  PLOTTING TRAINING HISTORY
# ─────────────────────────────────────────────────────────

def plot_history(history_head, history_ft=None) -> None:
    """
    Plot training/validation accuracy and loss curves.
    Saves PNG files to reports/figures/.
    """
    import matplotlib.pyplot as plt

    ensure_dir(FIGURES_DIR)

    # Merge histories if fine-tuning was done
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

    # Add a vertical line at the end of phase 1 if fine-tuning was done
    phase1_end = len(history_head.history.get("accuracy", []))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (train_vals, val_vals, ylabel, title, fpath) in zip(
        axes,
        [
            (acc,     val_acc, "Accuracy",  "Model Accuracy",  TRAINING_ACC_PATH),
            (loss,    val_los, "Loss",      "Model Loss",      TRAINING_LOSS_PATH),
        ],
    ):
        ax.plot(epochs, train_vals, label="Train",      linewidth=2, color="#2196F3")
        ax.plot(epochs, val_vals,   label="Validation", linewidth=2, color="#FF5722", linestyle="--")
        if history_ft is not None and phase1_end < len(epochs):
            ax.axvline(x=phase1_end + 0.5, color="gray", linestyle=":", linewidth=1.5, label="Fine-tune start")
        ax.set_xlabel("Epoch",  fontsize=12)
        ax.set_ylabel(ylabel,   fontsize=12)
        ax.set_title(title,     fontsize=13, fontweight="bold")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(TRAINING_ACC_PATH, dpi=150, bbox_inches="tight")

    # Also save individual plots
    for ax, fpath in zip(axes, [TRAINING_ACC_PATH, TRAINING_LOSS_PATH]):
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(fpath, bbox_inches=extent.expanded(1.1, 1.2), dpi=150)

    plt.close()
    logger.info(f"Training plots saved → {FIGURES_DIR}")


# ─────────────────────────────────────────────────────────
# 5.  SAVE TRAINING METADATA
# ─────────────────────────────────────────────────────────

def save_training_metadata(
    history_head,
    history_ft,
    class_weights: dict,
    n_train: int,
    n_val: int,
    n_test: int,
    duration_seconds: float,
) -> None:
    """Persist training metadata to JSON for reproducibility."""
    best_val_acc  = max(history_head.history.get("val_accuracy", [0]))
    best_val_loss = min(history_head.history.get("val_loss", [9999]))

    if history_ft is not None:
        ft_acc  = history_ft.history.get("val_accuracy", [])
        ft_loss = history_ft.history.get("val_loss", [])
        if ft_acc:
            best_val_acc  = max(best_val_acc,  max(ft_acc))
        if ft_loss:
            best_val_loss = min(best_val_loss, min(ft_loss))

    meta = {
        "model_name":           BASE_MODEL_NAME,
        "pretrained_weights":   PRETRAINED_WEIGHTS,
        "input_shape":          list(INPUT_SHAPE),
        "num_classes":          len(TARGET_CLASSES),
        "class_names":          TARGET_CLASSES,
        "batch_size":           BATCH_SIZE,
        "epochs_head":          EPOCHS_HEAD,
        "epochs_finetune":      EPOCHS_FINETUNE,
        "learning_rate_head":   LEARNING_RATE_HEAD,
        "learning_rate_ft":     LEARNING_RATE_FINETUNE,
        "dropout_rate":         DROPOUT_RATE,
        "dense_units":          DENSE_UNITS,
        "train_samples":        n_train,
        "val_samples":          n_val,
        "test_samples":         n_test,
        "class_weights":        class_weights,
        "best_val_accuracy":    round(best_val_acc, 4),
        "best_val_loss":        round(best_val_loss, 4),
        "training_duration_s":  round(duration_seconds, 1),
        "random_seed":          RANDOM_SEED,
    }
    save_json(meta, TRAINING_META_PATH)


# ─────────────────────────────────────────────────────────
# 6.  MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────

def train():
    """Full training pipeline."""
    print_section("AI RETINAL SCREENING — MODEL TRAINING")
    tf = _setup_tensorflow()

    # ── Load dataset ──────────────────────────────────────
    from src.dataset import (
        assign_four_class_labels,
        build_dataframe,
        load_hf_dataset,
    )
    from src.preprocessing import run_full_pipeline

    ds = load_hf_dataset(streaming=False)
    df_meta = build_dataframe(ds)
    df_labelled, _ = assign_four_class_labels(df_meta)

    # ── Preprocess & split ────────────────────────────────
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     datagen_train) = run_full_pipeline(ds, df_labelled)

    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
    num_classes = len(TARGET_CLASSES)

    # ── Class weights ─────────────────────────────────────
    class_weights = compute_class_weights(y_train, num_classes)
    logger.info(f"Class weights: {class_weights}")

    # ── Phase 1: Train head ───────────────────────────────
    print_section("Phase 1: Training Classification Head")
    model = build_model(num_classes=num_classes, trainable_backbone=False)
    model = compile_model(model, learning_rate=LEARNING_RATE_HEAD)
    model.summary(print_fn=logger.info)

    # Create augmented generator
    train_gen = datagen_train.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=RANDOM_SEED)

    t0 = time.perf_counter()
    history_head = model.fit(
        train_gen,
        steps_per_epoch=max(1, n_train // BATCH_SIZE),
        epochs=EPOCHS_HEAD,
        validation_data=(X_val, y_val),
        callbacks=get_callbacks("head"),
        class_weight=class_weights,
        verbose=1,
    )

    # ── Phase 2: Fine-tune top backbone layers ────────────
    history_ft = None
    if EPOCHS_FINETUNE > 0:
        print_section("Phase 2: Fine-tuning Backbone (top layers)")
        # Load the best weights from phase 1
        if BEST_MODEL_PATH.exists():
            model.load_weights(str(BEST_MODEL_PATH))

        # Unfreeze the top N layers
        backbone_layer = model.get_layer(index=1)   # the EfficientNet layer
        total_layers   = len(backbone_layer.layers)
        for layer in backbone_layer.layers[-(FINETUNE_LAYERS):]:
            layer.trainable = True

        n_trainable = sum(
            np.prod(v.shape) for v in model.trainable_variables
        )
        logger.info(f"Fine-tuning: {FINETUNE_LAYERS} backbone layers unfrozen. "
                    f"Trainable params: {n_trainable:,}")

        model = compile_model(model, learning_rate=LEARNING_RATE_FINETUNE)

        history_ft = model.fit(
            train_gen,
            steps_per_epoch=max(1, n_train // BATCH_SIZE),
            epochs=EPOCHS_FINETUNE,
            validation_data=(X_val, y_val),
            callbacks=get_callbacks("finetune"),
            class_weight=class_weights,
            verbose=1,
        )

    duration = time.perf_counter() - t0
    logger.info(f"Total training time: {duration/60:.1f} minutes")

    # ── Save class names ──────────────────────────────────
    save_class_names(TARGET_CLASSES, CLASS_NAMES_PATH)

    # ── Save training metadata ────────────────────────────
    save_training_metadata(
        history_head, history_ft, class_weights,
        n_train, n_val, n_test, duration,
    )

    # ── Plot history ──────────────────────────────────────
    plot_history(history_head, history_ft)

    print_section("Training Complete")
    print(f"  Best model saved to : {BEST_MODEL_PATH}")
    print(f"  Next step           : python -m src.evaluate")


if __name__ == "__main__":
    train()
