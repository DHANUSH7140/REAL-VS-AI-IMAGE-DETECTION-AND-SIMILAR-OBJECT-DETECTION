"""
download_faces_fast.py — Fast, reliable dataset downloader for Real vs AI Faces.

Real faces: Random User API portraits (actual photos of real people)
AI faces:   thispersondoesnotexist.com (AI-generated faces)
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

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# ── 1. REAL FACES (Fast Downloads) ───────────────────────────────────────────
print("Downloading REAL faces (from RandomUser portraits)...")
real_count = 0
for gender in ["men", "women"]:
    for i in range(100):
        url = f"https://randomuser.me/api/portraits/{gender}/{i}.jpg"
        save_path = os.path.join(REAL_DIR, f"real_{gender}_{i:03d}.jpg")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
                data = resp.read()
                if len(data) > 1000:
                    with open(save_path, "wb") as f:
                        f.write(data)
                    real_count += 1
        except Exception:
            pass

print(f"✅ Saved {real_count} REAL faces to dataset/real/")

# ── 2. AI FACES ──────────────────────────────────────────────────────────────
print("\nDownloading AI faces (from thispersondoesnotexist.com)...")
ai_count = 0

for i in range(200):
    url = "https://thispersondoesnotexist.com/"
    save_path = os.path.join(AI_DIR, f"ai_face_{i:03d}.jpg")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = resp.read()
            if len(data) > 5000:
                with open(save_path, "wb") as f:
                    f.write(data)
                ai_count += 1
                if ai_count % 20 == 0:
                    print(f"  Downloaded {ai_count} AI faces...")
    except Exception:
        pass
    time.sleep(1.0)  # Politeness delay

print(f"✅ Saved {ai_count} AI faces to dataset/ai/")
print("\n✨ Done! Both datasets are now FACES. You can retrain models safely.")
