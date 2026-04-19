"""
train_resnet.py — Transfer-learning with ResNet50 for Real vs AI detection.

Phase 1: Freeze the entire ResNet50 base and train only the custom head.
Phase 2: Unfreeze the last ~20 layers of ResNet50 and fine-tune at a low LR.

Expects the same dataset/ layout as train_cnn.py.
Saves the final model as model_resnet.h5.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ──────────────────────────── CONFIG ────────────────────────────
DATASET_DIR   = os.path.join(os.path.dirname(__file__), "dataset")
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS_HEAD   = 15      # Phase 1
EPOCHS_FINE   = 15      # Phase 2
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "model_resnet.h5")
FINE_TUNE_AT  = -20     # unfreeze last 20 layers
# ────────────────────────────────────────────────────────────────


def build_generators():
    """Create training and validation generators with augmentation."""
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        rotation_range=25,
        zoom_range=0.2,
        horizontal_flip=True,
        shear_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
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


def build_resnet_model():
    """Build ResNet50 + custom classifier head."""
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    # Freeze all base layers initially
    base_model.trainable = False

    # Custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model


def plot_combined_history(hist1, hist2):
    """Combine Phase 1 + Phase 2 histories and plot."""
    acc  = hist1.history["accuracy"]  + hist2.history["accuracy"]
    vacc = hist1.history["val_accuracy"] + hist2.history["val_accuracy"]
    loss = hist1.history["loss"]  + hist2.history["loss"]
    vloss = hist1.history["val_loss"] + hist2.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, vacc, label="Val Accuracy")
    plt.axvline(x=len(hist1.history["accuracy"]), color="gray",
                linestyle="--", label="Fine-tune start")
    plt.title("ResNet50 — Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "resnet_accuracy.png"))
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, vloss, label="Val Loss")
    plt.axvline(x=len(hist1.history["loss"]), color="gray",
                linestyle="--", label="Fine-tune start")
    plt.title("ResNet50 — Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "resnet_loss.png"))
    plt.close()

    print("📊 Plots saved: resnet_accuracy.png, resnet_loss.png")


def main():
    print("=" * 60)
    print("  ResNet50 Transfer Learning — Real vs AI Image Detector")
    print("=" * 60)

    train_gen, val_gen = build_generators()
    print(f"\nClasses: {train_gen.class_indices}")
    print(f"Training samples  : {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}\n")

    model, base_model = build_resnet_model()

    # Shared callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    ]

    # ── Phase 1: Train head only ──────────────────────────────────
    print("\n🔒 Phase 1 — Training custom head (base frozen)…")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    hist1 = model.fit(
        train_gen,
        epochs=EPOCHS_HEAD,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # ── Phase 2: Fine-tune last N layers ──────────────────────────
    print(f"\n🔓 Phase 2 — Fine-tuning more ResNet layers…")
    base_model.trainable = True
    # Freeze initial layers, train last 60
    for layer in base_model.layers[:-60]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),   # much lower LR
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    hist2 = model.fit(
        train_gen,
        epochs=EPOCHS_FINE,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # ── Evaluate ──────────────────────────────────────────────────
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"\n✅ Final Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"   Final Validation Loss   : {val_loss:.4f}")

    # ── Save ──────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n💾 Model saved → {MODEL_PATH}")

    # ── Plots ─────────────────────────────────────────────────────
    plot_combined_history(hist1, hist2)


if __name__ == "__main__":
    main()
