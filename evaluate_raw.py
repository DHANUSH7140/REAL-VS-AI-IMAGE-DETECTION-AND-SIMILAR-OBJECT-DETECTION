import os
import glob
from tqdm import tqdm
import sys

# Add backend to path
sys.path.insert(0, r"d:\SEM 4 PROJECT\real-vs-ai-detector\visionprobe\backend")
from detector.predict import AIDetector
from PIL import Image

def evaluate():
    detector = AIDetector(weights_dir=r"d:\SEM 4 PROJECT\real-vs-ai-detector\visionprobe\backend\weights", device="auto")
    detector.load()
    
    # FORCE RAW XGBOOST
    detector.calibrated_model = None
    
    real_imgs = glob.glob(r"d:\SEM 4 PROJECT\real-vs-ai-detector\real-vs-ai-detector\dataset_v2\real\*.jpg")[:50]
    ai_imgs = glob.glob(r"d:\SEM 4 PROJECT\real-vs-ai-detector\real-vs-ai-detector\dataset_v2\ai\*.jpg")[:50]
    
    results = []
    
    for img_path in tqdm(real_imgs + ai_imgs):
        try:
            pil_img = Image.open(img_path)
            res = detector.predict_with_features(pil_img)
            true_label = "REAL" if "real\\" in img_path else "AI"
            results.append({
                "path": os.path.basename(img_path),
                "true": true_label,
                "pred": res["label"],
                "prob": res["raw_probability"],
                "conf": res["confidence"]
            })
        except Exception as e:
            pass
            
    # Calculate stats
    real_probs = [r["prob"] for r in results if r["true"] == "REAL"]
    ai_probs = [r["prob"] for r in results if r["true"] == "AI"]
    
    print("\n--- RAW XGBOOST STATS ---")
    print(f"Avg Real Prob: {sum(real_probs)/len(real_probs):.4f}")
    print(f"Avg AI Prob: {sum(ai_probs)/len(ai_probs):.4f}")
    
    correct = sum(1 for r in results if r["true"] == r["pred"])
    print(f"Accuracy: {correct}/{len(results)} ({correct/len(results)*100:.2f}%)")
    
    confs = [r["conf"] for r in results]
    print(f"Avg Confidence: {sum(confs)/len(confs):.2f}%")
    print(f"Min Confidence: {min(confs):.2f}%")
    print(f"Max Confidence: {max(confs):.2f}%")
    
    low_conf = sum(1 for c in confs if c < 60.0)
    print(f"Predictions with <60% confidence: {low_conf}/{len(confs)} ({low_conf/len(confs)*100:.2f}%)")

if __name__ == "__main__":
    evaluate()
