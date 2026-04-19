"""
train_efficientnet.py — Train EfficientNetB4 for Real vs AI Image Detection.

Two-phase training:
  Phase 1: Freeze EfficientNet base → train custom head (LR 1e-4, 10 epochs)
  Phase 2: Unfreeze top 30 layers → fine-tune (LR 1e-5, 15 epochs)

Expects dataset_v2/real/ and dataset_v2/ai/ directories.
Saves model to model/efficientnet_trained.h5.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report

# ──────────────────────────── CONFIG ────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(BASE_DIR, "dataset_v2")
MODEL_DIR     = os.path.join(BASE_DIR, "model")
MODEL_PATH    = os.path.join(MODEL_DIR, "efficientnet_trained.h5")
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS_HEAD   = 10
EPOCHS_FINE   = 15
FINE_TUNE_AT  = -30     # unfreeze last 30 layers
# ────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)


def build_generators():
    """Create training and validation generators with augmentation."""
    datagen = ImageDataGenerator(
        # rescale=1.0 / 255,  <-- REMOVED: EfficientNetB4 has an internal Rescaling layer
        validation_split=0.2,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        shear_range=0.15,
        fill_mode="nearest",
    )

    train_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        classes=["real", "ai"],  # 0 = real, 1 = ai
    )

    val_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        classes=["real", "ai"],
    )

    return train_gen, val_gen


def build_model():
    """Build EfficientNetB4 + custom classifier head."""
    base_model = EfficientNetB4(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model


def plot_combined(h1, h2):
    """Plot combined Phase 1 + Phase 2 accuracy and loss."""
    acc   = h1.history["accuracy"]  + h2.history["accuracy"]
    vacc  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss  = h1.history["loss"]  + h2.history["loss"]
    vloss = h1.history["val_loss"] + h2.history["val_loss"]
    epochs = range(1, len(acc) + 1)
    phase1_end = len(h1.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, acc, label="Train")
    ax1.plot(epochs, vacc, label="Val")
    ax1.axvline(x=phase1_end, color="gray", ls="--", label="Fine-tune start")
    ax1.set_title("EfficientNetB4 — Accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy"); ax1.legend()

    ax2.plot(epochs, loss, label="Train")
    ax2.plot(epochs, vloss, label="Val")
    ax2.axvline(x=phase1_end, color="gray", ls="--", label="Fine-tune start")
    ax2.set_title("EfficientNetB4 — Loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "efficientnet_training_plots.png"), dpi=120)
    plt.close()
    print("📊 Plot saved → efficientnet_training_plots.png")


def evaluate_metrics(model, val_gen):
    """Print precision / recall / F1."""
    val_gen.reset()
    preds = model.predict(val_gen, verbose=0)
    y_pred = (preds > 0.5).astype(int).flatten()
    y_true = val_gen.classes

    print("\n" + "=" * 50)
    print("  Classification Report")
    print("=" * 50)
    print(classification_report(
        y_true, y_pred,
        target_names=["Real", "AI Generated"],
        digits=4,
    ))


def main():
    print("=" * 60)
    print("  EfficientNetB4 Training — Real vs AI Image Detector")
    print("=" * 60)

    train_gen, val_gen = build_generators()
    print(f"\nClasses: {train_gen.class_indices}")
    print(f"Training samples  : {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}\n")

    model, base_model = build_model()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    ]

    # ── Phase 1 ───────────────────────────────────────────────────
    print("🔒 Phase 1 — Training head only (base frozen)…")
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    h1 = model.fit(
        train_gen,
        epochs=EPOCHS_HEAD,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # ── Phase 2 ───────────────────────────────────────────────────
    print("\n🔓 Phase 2 — Fine-tuning more EfficientNet layers…")
    base_model.trainable = True
    # Freeze the first (total_layers - 100)
    for layer in base_model.layers[:-100]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-6),  # Ultra low LR for 95%+ accuracy
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    h2 = model.fit(
        train_gen,
        epochs=EPOCHS_FINE,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # ── Evaluate ──────────────────────────────────────────────────
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"\n✅ Final Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"   Final Validation Loss   : {val_loss:.4f}")

    evaluate_metrics(model, val_gen)

    # ── Save ──────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n💾 Model saved → {MODEL_PATH}")

    plot_combined(h1, h2)


if __name__ == "__main__":
    main()
