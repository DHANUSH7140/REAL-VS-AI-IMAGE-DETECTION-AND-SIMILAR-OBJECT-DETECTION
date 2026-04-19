import os
import glob
import numpy as np
from tensorflow.keras.models import load_model
from app import preprocess_image

def main():
    model_path = "model_cnn.h5"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
        
    print(f"Loading {model_path}...")
    model = load_model(model_path)
    
    # Let's test on a few 'real' and 'ai' images
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print(f"Dataset dir not found: {dataset_dir}")
        return
        
    real_imgs = glob.glob(os.path.join(dataset_dir, "real", "*.*"))[:5]
    ai_imgs = glob.glob(os.path.join(dataset_dir, "ai", "*.*"))[:5]
    
    print("\n--- Testing 'Real' images (Expected output ~ 0.0) ---")
    for img_path in real_imgs:
        arr = preprocess_image(img_path)
        pred = model.predict(arr, verbose=0)[0][0]
        print(f"Pred: {pred:.4f}  | Image: {os.path.basename(img_path)}")
        
    print("\n--- Testing 'AI' images (Expected output ~ 1.0) ---")
    for img_path in ai_imgs:
        arr = preprocess_image(img_path)
        pred = model.predict(arr, verbose=0)[0][0]
        print(f"Pred: {pred:.4f}  | Image: {os.path.basename(img_path)}")

if __name__ == "__main__":
    main()
