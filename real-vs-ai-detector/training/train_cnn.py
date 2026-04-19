"""
train_cnn.py — Train a custom CNN to classify Real vs AI-generated images.

Directory layout expected:
    dataset/
        real/   ← place real photographs here
        ai/     ← place AI-generated images here

The script applies data augmentation, trains the model, saves it as
model_cnn.h5, and plots accuracy / loss curves.
"""

import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so plots save to file
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ──────────────────────────── CONFIG ────────────────────────────
DATASET_DIR  = os.path.join(os.path.dirname(__file__), "dataset_v2")
IMG_SIZE     = (128, 128)
BATCH_SIZE   = 32
EPOCHS       = 30
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "model_cnn.h5")
# ────────────────────────────────────────────────────────────────


def build_generators():
    """Create training and validation generators with augmentation."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=30,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        shear_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        classes=["real", "ai"],   # 0 = real, 1 = ai
    )

    val_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        classes=["real", "ai"],
    )

    return train_gen, val_gen


def build_cnn_model():
    """Build a CNN with Conv2D (32→64→128→256), BN, MaxPool, Dropout."""
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation="relu", padding="same",
               input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 2
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 3
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Block 4
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Classifier head
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),   # binary output
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    """Save accuracy and loss plots as PNG files."""
    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("CNN — Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "cnn_accuracy.png"))
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("CNN — Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "cnn_loss.png"))
    plt.close()

    print("📊 Plots saved: cnn_accuracy.png, cnn_loss.png")


def main():
    print("=" * 60)
    print("  CNN Training — Real vs AI Image Detector")
    print("=" * 60)

    # 1. Data generators
    train_gen, val_gen = build_generators()
    print(f"\nClasses: {train_gen.class_indices}")
    print(f"Training samples  : {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}\n")

    # 2. Build model
    model = build_cnn_model()
    model.summary()

    # 3. Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    ]

    # 4. Train
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # 5. Evaluate
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"\n✅ Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"   Validation Loss   : {val_loss:.4f}")

    # 6. Save
    model.save(MODEL_PATH)
    print(f"\n💾 Model saved → {MODEL_PATH}")

    # 7. Plots
    plot_history(history)


if __name__ == "__main__":
    main()
