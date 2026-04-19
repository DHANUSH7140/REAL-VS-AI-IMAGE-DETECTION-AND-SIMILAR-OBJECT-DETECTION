"""
train_resnet.py — Transfer-learning with ResNet50 for Real vs AI detection.

Optimized for 90%+ accuracy with:
  - Advanced augmentation (noise, color jitter, compression simulation)
  - Label smoothing (0.05)
  - Two-phase training with cosine annealing LR
  - Class weight balancing
  - Classification report + confusion matrix

Phase 1: Freeze ResNet50 base → train custom head (LR 1e-4, 15 epochs)
Phase 2: Unfreeze last 80 layers → fine-tune (LR 1e-6, 15 epochs)

Expects dataset_v2/real/ and dataset_v2/ai/ directories.
Saves the final model as model_resnet.h5.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ──────────────────────────── CONFIG ────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(BASE_DIR, "dataset_v2")   # Fixed: was "dataset"
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS_HEAD   = 15      # Phase 1
EPOCHS_FINE   = 15      # Phase 2
MODEL_PATH    = os.path.join(BASE_DIR, "model_resnet.h5")
LABEL_SMOOTH  = 0.05
# ────────────────────────────────────────────────────────────────


def gaussian_noise_augment(image):
    """Add random Gaussian noise for robustness."""
    image = preprocess_input(image.copy())
    if np.random.random() < 0.3:
        noise = np.random.normal(0, np.random.uniform(3, 15), image.shape)
        image = image + noise
    return image


def build_generators():
    """Create training and validation generators with augmentation."""
    datagen = ImageDataGenerator(
        preprocessing_function=gaussian_noise_augment,
        validation_split=0.2,
        rotation_range=25,
        zoom_range=0.25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        shear_range=0.15,
        brightness_range=[0.7, 1.3],
        channel_shift_range=20.0,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
    )

    train_gen = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        classes=["real", "ai"],
    )

    val_gen = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        classes=["real", "ai"],
    )

    return train_gen, val_gen


def build_resnet_model():
    """Build ResNet50 + custom classifier head with regularization."""
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    # Freeze all base layers initially
    base_model.trainable = False

    # Custom head with stronger regularization
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu",
              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu",
              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model


def compute_weights(train_gen):
    """Compute class weights for balanced training."""
    labels = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(zip(classes, weights))


class CosineAnnealingScheduler(Callback):
    """Cosine annealing learning rate schedule."""

    def __init__(self, initial_lr, min_lr, epochs):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.epochs = epochs

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
             (1 + math.cos(math.pi * epoch / self.epochs))
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


def plot_combined_history(hist1, hist2):
    """Combine Phase 1 + Phase 2 histories and plot."""
    acc  = hist1.history["accuracy"]  + hist2.history["accuracy"]
    vacc = hist1.history["val_accuracy"] + hist2.history["val_accuracy"]
    loss = hist1.history["loss"]  + hist2.history["loss"]
    vloss = hist1.history["val_loss"] + hist2.history["val_loss"]

    epochs = range(1, len(acc) + 1)
    phase1_end = len(hist1.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, acc, label="Train")
    ax1.plot(epochs, vacc, label="Val")
    ax1.axvline(x=phase1_end, color="gray", ls="--", label="Fine-tune start")
    ax1.set_title("ResNet50 — Accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy"); ax1.legend()

    ax2.plot(epochs, loss, label="Train")
    ax2.plot(epochs, vloss, label="Val")
    ax2.axvline(x=phase1_end, color="gray", ls="--", label="Fine-tune start")
    ax2.set_title("ResNet50 — Loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "resnet_training_plots.png"), dpi=120)
    plt.close()
    print("📊 Plot saved → resnet_training_plots.png")


def evaluate_metrics(model, val_gen):
    """Print precision / recall / F1 + confusion matrix."""
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

    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion Matrix:")
    print(f"    Real→Real: {cm[0][0]}  Real→AI: {cm[0][1]}")
    print(f"    AI→Real:   {cm[1][0]}  AI→AI:   {cm[1][1]}")


def main():
    print("=" * 60)
    print("  ResNet50 Transfer Learning — Real vs AI Image Detector")
    print("  (Optimized for 90%+ accuracy)")
    print("=" * 60)

    train_gen, val_gen = build_generators()
    print(f"\nClasses: {train_gen.class_indices}")
    print(f"Training samples  : {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}\n")

    model, base_model = build_resnet_model()
    class_weights = compute_weights(train_gen)
    print(f"Class weights: {class_weights}\n")

    # ── Phase 1: Train head only ──────────────────────────────────
    print("🔒 Phase 1 — Training custom head (base frozen)…")
    model.compile(
        optimizer=Adam(learning_rate=1e-4),  # Fixed: was 1e-3 (too aggressive)
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=["accuracy"],
    )
    model.summary()

    callbacks_p1 = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    ]

    hist1 = model.fit(
        train_gen,
        epochs=EPOCHS_HEAD,
        validation_data=val_gen,
        callbacks=callbacks_p1,
        class_weight=class_weights,
    )

    # ── Phase 2: Fine-tune last 80 layers ─────────────────────────
    print(f"\n🔓 Phase 2 — Fine-tuning ResNet layers (last 80)…")
    base_model.trainable = True
    for layer in base_model.layers[:-80]:
        layer.trainable = False

    trainable_count = sum(1 for l in model.layers if l.trainable)
    total_count = len(model.layers)
    print(f"   Trainable layers: {trainable_count}/{total_count}")

    model.compile(
        optimizer=Adam(learning_rate=1e-6),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=["accuracy"],
    )

    callbacks_p2 = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        CosineAnnealingScheduler(initial_lr=1e-5, min_lr=1e-7, epochs=EPOCHS_FINE),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    ]

    hist2 = model.fit(
        train_gen,
        epochs=EPOCHS_FINE,
        validation_data=val_gen,
        callbacks=callbacks_p2,
        class_weight=class_weights,
    )

    # ── Evaluate ──────────────────────────────────────────────────
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"\n✅ Final Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"   Final Validation Loss   : {val_loss:.4f}")

    evaluate_metrics(model, val_gen)

    # ── Save ──────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n💾 Model saved → {MODEL_PATH}")

    plot_combined_history(hist1, hist2)


if __name__ == "__main__":
    main()
