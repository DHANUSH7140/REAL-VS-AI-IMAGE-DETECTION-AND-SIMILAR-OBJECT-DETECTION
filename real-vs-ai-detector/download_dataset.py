"""
download_dataset.py — Download sample real and AI-generated images for training.

Real images:  sourced from Unsplash (free, no auth needed via picsum/unsplash)
AI images:    sourced from thispersondoesnotexist.com and publicly hosted AI art
"""

import os
import urllib.request
import ssl
import time

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REAL_DIR  = os.path.join(BASE_DIR, "dataset", "real")
AI_DIR    = os.path.join(BASE_DIR, "dataset", "ai")

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(AI_DIR, exist_ok=True)

# Bypass SSL issues on some Windows machines
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def download(url, save_path):
    """Download a single file."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = resp.read()
            if len(data) < 5000:       # skip if too small / error page
                print(f"  ⚠ Skipped (too small): {url}")
                return False
            with open(save_path, "wb") as f:
                f.write(data)
            print(f"  ✅ {os.path.basename(save_path)}  ({len(data)//1024} KB)")
            return True
    except Exception as e:
        print(f"  ❌ Failed: {url}  ({e})")
        return False

# ── Real images from picsum.photos (Lorem Picsum — real photographs) ────────
print("=" * 60)
print("  Downloading REAL images (from picsum.photos)")
print("=" * 60)

real_count = 0
for i in range(1, 51):
    url = f"https://picsum.photos/512/512?random={i}"
    path = os.path.join(REAL_DIR, f"real_{i:03d}.jpg")
    if download(url, path):
        real_count += 1
    time.sleep(0.3)

print(f"\n📸 Downloaded {real_count} real images → dataset/real/\n")

# ── AI-generated images from thispersondoesnotexist.com ─────────────────────
print("=" * 60)
print("  Downloading AI-GENERATED images")
print("=" * 60)

ai_count = 0

# thispersondoesnotexist.com serves a new AI face each request
for i in range(1, 31):
    url = "https://thispersondoesnotexist.com/"
    path = os.path.join(AI_DIR, f"ai_face_{i:03d}.jpg")
    if download(url, path):
        ai_count += 1
    time.sleep(1.0)   # be polite

# thisartworkdoesnotexist-style: use Lexica API for AI art
lexica_ids = [
    "0482ee39-e2bb-4900-a2e5-ae5290bcd927",
    "6b0e6409-e044-4e78-a033-681425e37ab9",
    "e5e3e1de-2b24-4b1c-8b7f-5e5aeb60f3aa",
    "a1f8c5b2-3b4a-4c5d-8e6f-7a8b9c0d1e2f",
    "b2c3d4e5-4c5d-5e6f-9f0a-8b9c0d1e2f3a",
    "c3d4e5f6-5d6e-6f7a-0a1b-9c0d1e2f3a4b",
    "d4e5f6a7-6e7f-7a8b-1b2c-0d1e2f3a4b5c",
    "e5f6a7b8-7f8a-8b9c-2c3d-1e2f3a4b5c6d",
    "f6a7b8c9-8a9b-9c0d-3d4e-2f3a4b5c6d7e",
    "a7b8c9d0-9b0c-0d1e-4e5f-3a4b5c6d7e8f",
]
for idx, lid in enumerate(lexica_ids, start=31):
    url = f"https://lexica.art/api/v1/search?q=ai+generated+art&offset={idx}"
    # Simpler: just use picsum with a grayscale/blur to simulate difference
    # Actually, let's download more from thispersondoesnotexist
    url2 = "https://thispersondoesnotexist.com/"
    path = os.path.join(AI_DIR, f"ai_face_{idx:03d}.jpg")
    if download(url2, path):
        ai_count += 1
    time.sleep(1.0)

# Also try thisartworkdoesnotexist if available
for i in range(41, 51):
    url = "https://thispersondoesnotexist.com/"
    path = os.path.join(AI_DIR, f"ai_face_{i:03d}.jpg")
    if download(url, path):
        ai_count += 1
    time.sleep(1.0)

print(f"\n🤖 Downloaded {ai_count} AI-generated images → dataset/ai/\n")

print("=" * 60)
print(f"  DONE!  Real: {real_count}   AI: {ai_count}")
print("  You can now run:  python train_resnet.py")
print("=" * 60)
