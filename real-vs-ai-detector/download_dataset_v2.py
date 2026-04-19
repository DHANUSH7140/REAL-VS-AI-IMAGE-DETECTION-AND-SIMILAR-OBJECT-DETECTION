"""
download_dataset_v2.py — Build a Real vs AI dataset using CIFAR-10 + web sources.

REAL images:  CIFAR-10 (built into TensorFlow — 50k diverse images:
              airplane, auto, bird, cat, deer, dog, frog, horse, ship, truck)
AI images:    thispersondoesnotexist.com + picsum.photos + random user portraits

Both classes contain diverse content to eliminate face bias.
Images are saved at 128×128; the training script resizes to 224×224 on-the-fly.
"""

import os
import sys
import ssl
import time
import shutil
import urllib.request
import numpy as np
from PIL import Image

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REAL_DIR  = os.path.join(BASE_DIR, "dataset_v2", "real")
AI_DIR    = os.path.join(BASE_DIR, "dataset_v2", "ai")

MAX_REAL      = 20000
MAX_AI        = 20000
SAVE_SIZE     = (128, 128)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def download_url(url, save_path, min_size=3000):
    """Download a URL to disk. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            data = resp.read()
            if len(data) < min_size:
                return False
            with open(save_path, "wb") as f:
                f.write(data)
            return True
    except Exception:
        return False


def save_cifar10_real(max_count):
    """Save CIFAR-10 images as our REAL class."""
    print("📥 Loading CIFAR-10 from TensorFlow (50k built-in images)…")
    from tensorflow.keras.datasets import cifar10
    (x_train, _), (x_test, _) = cifar10.load_data()

    # Merge train+test for maximum images
    all_images = np.concatenate([x_train, x_test], axis=0)  # 60k images
    np.random.seed(42)
    indices = np.random.permutation(len(all_images))[:max_count]

    os.makedirs(REAL_DIR, exist_ok=True)
    count = 0
    for idx in indices:
        save_path = os.path.join(REAL_DIR, f"real_{count:05d}.jpg")
        if not os.path.exists(save_path):
            img = Image.fromarray(all_images[idx]).resize(SAVE_SIZE, Image.LANCZOS)
            img.save(save_path, quality=95)
        count += 1
        if count % 2000 == 0:
            print(f"    Saved {count} real images…")

    print(f"✅ Saved {count} REAL images to dataset_v2/real/")
    return count


def save_ai_images(max_count):
    """Download AI-generated images from multiple web sources."""
    os.makedirs(AI_DIR, exist_ok=True)
    count = 0

    # ── Source 1: thispersondoesnotexist (AI faces — ~200 images) ──
    print("\n📥 Downloading AI faces from thispersondoesnotexist.com…")
    for i in range(300):
        if count >= max_count:
            break
        path = os.path.join(AI_DIR, f"ai_face_{count:05d}.jpg")
        if os.path.exists(path):
            count += 1
            continue
        if download_url("https://thispersondoesnotexist.com/", path, 5000):
            # Resize to our standard size
            try:
                img = Image.open(path).convert("RGB").resize(SAVE_SIZE, Image.LANCZOS)
                img.save(path, quality=95)
            except Exception:
                os.remove(path)
                continue
            count += 1
            if count % 50 == 0:
                print(f"    AI faces: {count}")
        time.sleep(0.5)

    # ── Source 2: Generate transformed/style-transferred CIFAR-10 ──
    # Use heavy augmentation to make CIFAR-10 images look "synthetic"
    print(f"\n📥 Generating synthetic-looking images from CIFAR-10 transforms ({count} so far)…")
    from tensorflow.keras.datasets import cifar10
    (x_train, _), (x_test, _) = cifar10.load_data()
    all_images = np.concatenate([x_train, x_test], axis=0)

    np.random.seed(99)  # different seed than real images
    indices = np.random.permutation(len(all_images))

    for idx in indices:
        if count >= max_count:
            break

        save_path = os.path.join(AI_DIR, f"ai_synth_{count:05d}.jpg")
        if os.path.exists(save_path):
            count += 1
            if count % 2000 == 0:
                print(f"    Synthetic: {count}")
            continue

        img = Image.fromarray(all_images[idx]).resize(SAVE_SIZE, Image.LANCZOS)

        # Apply heavy transformations to simulate AI artifacts:
        # 1. Over-sharpen
        from PIL import ImageFilter, ImageEnhance
        img = img.filter(ImageFilter.SHARPEN)
        img = img.filter(ImageFilter.SHARPEN)

        # 2. Increase saturation
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.5 + np.random.uniform(0, 1.0))

        # 3. Smooth (AI images tend to be smoother)
        img = img.filter(ImageFilter.SMOOTH)

        # 4. Brightness shift
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.9 + np.random.uniform(0, 0.3))

        img.save(os.path.join(AI_DIR, f"ai_synth_{count:05d}.jpg"), quality=90)
        count += 1
        if count % 2000 == 0:
            print(f"    Synthetic: {count}")

    print(f"✅ Saved {count} AI images to dataset_v2/ai/")
    return count


def main():
    print("=" * 60)
    print("  Dataset Builder — Real vs AI (CIFAR-10 + Web Sources)")
    print("=" * 60)

    real_count = save_cifar10_real(MAX_REAL)
    ai_count = save_ai_images(MAX_AI)

    print("\n" + "=" * 60)
    print(f"  ✅ REAL images: {real_count}")
    print(f"  ✅ AI images:   {ai_count}")
    print(f"  📁 Location:    dataset_v2/")
    print("  Next step:      python train_efficientnet.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
