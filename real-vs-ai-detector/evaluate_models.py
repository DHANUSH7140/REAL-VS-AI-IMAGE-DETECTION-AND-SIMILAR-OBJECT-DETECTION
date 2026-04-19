import os
import glob
import random
import numpy as np
import tensorflow as tf
from app import get_model, preprocess_image, BASE_DIR

# Disable tf logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATASET_DIR = os.path.join(BASE_DIR, "dataset_v2")

def evaluate_model(model_name, num_samples=100):
    try:
        model = get_model(model_name)
    except Exception as e:
        print(f"[{model_name.upper()}] Failed to load: {e}")
        return

    real_imgs = glob.glob(os.path.join(DATASET_DIR, "real", "*.*"))
    ai_imgs = glob.glob(os.path.join(DATASET_DIR, "ai", "*.*"))
    
    if not real_imgs or not ai_imgs:
        print("Dataset not found or empty.")
        return

    # Sample randomly
    random.seed(42)
    real_sample = random.sample(real_imgs, min(num_samples, len(real_imgs)))
    ai_sample = random.sample(ai_imgs, min(num_samples, len(ai_imgs)))
    
    correct = 0
    total = 0
    
    print(f"\n--- Evaluating {model_name.upper()} ---")
    
    # Real images -> expected prediction < 0.5
    for img in real_sample:
        arr = preprocess_image(img)
        pred = model.predict(arr, verbose=0)[0][0]
        if pred < 0.5:
            correct += 1
        total += 1
        
    # AI images -> expected prediction >= 0.5
    for img in ai_sample:
        arr = preprocess_image(img)
        pred = model.predict(arr, verbose=0)[0][0]
        if pred >= 0.5:
            correct += 1
        total += 1
        
    accuracy = (correct / total) * 100
    print(f"[{model_name.upper()}] Accuracy on {total} samples: {accuracy:.2f}%")

def main():
    print("Testing models on a subset of dataset_v2...")
    # Using 50 samples each to keep evaluation fast (~100 images total per model)
    evaluate_model("resnet", 50)
    evaluate_model("efficientnet", 50)

if __name__ == "__main__":
    main()
