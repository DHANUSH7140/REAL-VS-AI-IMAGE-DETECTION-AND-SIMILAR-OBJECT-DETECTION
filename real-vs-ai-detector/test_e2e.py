"""Quick end-to-end test for the Real vs AI Detector API."""
import requests

BASE = "http://127.0.0.1:5000"

def test(label, url, image_path):
    print(f"=== {label} ===")
    with open(image_path, "rb") as f:
        resp = requests.post(url, files={"file": ("test.jpg", f, "image/jpeg")})
    print(f"  Status : {resp.status_code}")
    data = resp.json()
    if resp.status_code != 200:
        print(f"  Error  : {data.get('error')}")
        return
    pred = data.get("prediction", data.get("label", "?"))
    conf = data.get("confidence", "?")
    gcam = data.get("gradcam_image", data.get("gradcam_url", "None"))
    fft  = list(data.get("fft_analysis", {}).keys()) if data.get("fft_analysis") else "None"
    print(f"  Predict: {pred}")
    print(f"  Confid : {conf}%")
    print(f"  GradCAM: {gcam}")
    print(f"  FFT    : {fft}")
    print()

# 1) CNN → AI image
test("CNN on AI image",
     f"{BASE}/api/predict?model=cnn",
     "dataset_v2/ai/ai_face_00001.jpg")

# 2) EfficientNet → Real image
test("EfficientNet on Real image",
     f"{BASE}/api/predict?model=efficientnet",
     "dataset_v2/real/real_00001.jpg")

# 3) Ensemble → AI image
test("Ensemble on AI image",
     f"{BASE}/api/predict?model=ensemble",
     "dataset_v2/ai/ai_face_00002.jpg")

print("✅ All tests completed.")
