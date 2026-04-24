import os
import glob
import sys
from PIL import Image

sys.path.insert(0, r"d:\SEM 4 PROJECT\real-vs-ai-detector\visionprobe\backend")
from detector.predict import AIDetector

def analyze_errors():
    detector = AIDetector(weights_dir=r"d:\SEM 4 PROJECT\real-vs-ai-detector\visionprobe\backend\weights", device="auto")
    detector.load()
    
    real_imgs = glob.glob(r"d:\SEM 4 PROJECT\real-vs-ai-detector\real-vs-ai-detector\dataset_v2\real\*.jpg")[:50]
    
    failed = []
    print("Evaluating real images...")
    for img_path in real_imgs:
        try:
            pil_img = Image.open(img_path)
            res = detector.predict(pil_img)
            if res["label"] == "AI":
                failed.append((os.path.basename(img_path), res["confidence"], res["raw_probability"]))
        except Exception as e:
            pass
            
    print(f"\nFailed on {len(failed)}/{len(real_imgs)} real images (False Positives):")
    for f in failed:
        print(f"  {f[0]}: Predicted AI with conf {f[1]}% (prob {f[2]:.4f})")

if __name__ == "__main__":
    analyze_errors()
