"""
download_faces.py — Download a clear Real Faces vs AI Faces dataset.

Real faces:  Labeled Faces in the Wild (via scikit-learn)
AI faces:    thispersondoesnotexist.com
"""

import os
import shutil
import urllib.request
import ssl
import time
import numpy as np
from PIL import Image

try:
    from sklearn.datasets import fetch_lfw_people
except ImportError:
    print("Please install scikit-learn: pip install scikit-learn")
    exit(1)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REAL_DIR  = os.path.join(BASE_DIR, "dataset", "real")
AI_DIR    = os.path.join(BASE_DIR, "dataset", "ai")

# Clean old dataset (which mixed random photos with faces)
print("Cleaning old dataset folders...")
shutil.rmtree(REAL_DIR, ignore_errors=True)
shutil.rmtree(AI_DIR, ignore_errors=True)
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(AI_DIR, exist_ok=True)

# ── 1. Fetch Real Faces (LFW) ────────────────────────────────────────────────
print("\nDownloading REAL faces from LFW dataset...")
# min_faces_per_person limits it, resize=1.0 keeps original 62x47 size initially,
# but let's fetch resize=2.0 or so to get bigger faces if possible. (Default is fine).
lfw = fetch_lfw_people(min_faces_per_person=10, resize=2.0)
images = lfw.images  # Float array [N, H, W]
num_real_to_save = min(300, len(images))

for i in range(num_real_to_save):
    # LFW images are float32 normalized, need to convert to 0-255 uint8
    img_array = images[i]
    # Normalize to 0-255 if not already
    img_array = (img_array / img_array.max() * 255).astype(np.uint8)
    
    img = Image.fromarray(img_array).convert("RGB")
    save_path = os.path.join(REAL_DIR, f"real_{i:04d}.jpg")
    img.save(save_path)

print(f"✅ Saved {num_real_to_save} real faces to dataset/real/")

# ── 2. Fetch AI Faces ────────────────────────────────────────────────────────
print("\nDownloading AI-GENERATED faces...")

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

ai_count = 0
num_ai_to_save = 200

# thispersondoesnotexist generates a new image per request
for i in range(num_ai_to_save):
    url = "https://thispersondoesnotexist.com/"
    save_path = os.path.join(AI_DIR, f"ai_face_{i:04d}.jpg")
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = resp.read()
            if len(data) > 5000:
                with open(save_path, "wb") as f:
                    f.write(data)
                ai_count += 1
                if ai_count % 10 == 0:
                    print(f"  Downloaded {ai_count} AI faces...")
    except Exception as e:
        pass
    
    time.sleep(1.0)  # Politeness delay

print(f"✅ Saved {ai_count} AI faces to dataset/ai/")
print("\nDataset ready! You can now run train_resnet.py and train_cnn.py.")
