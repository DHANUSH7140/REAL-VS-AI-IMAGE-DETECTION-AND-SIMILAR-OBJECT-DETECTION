"""Quick end-to-end test of the VisionProbe API."""
import requests
import json
import os
import sys

from PIL import Image
import numpy as np

# Generate a simple test image
test_img_path = os.path.join(os.path.dirname(__file__), "test_random.jpg")
img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
img.save(test_img_path)
print(f"Test image: {test_img_path}")

# Test health
print("\n--- /health ---")
r = requests.get("http://localhost:8000/health", timeout=10)
print(json.dumps(r.json(), indent=2))

# Test model-info
print("\n--- /api/model-info ---")
r = requests.get("http://localhost:8000/api/model-info", timeout=10)
info = r.json()
print(f"Pipeline: {info['pipeline']}")
print(f"Classifier: {info['classifier']}")
for m in info["models"]:
    print(f"  [{m['branch']}] {m['name']} - available: {m['available']}")

# Test analyze
print("\n--- /api/analyze ---")
with open(test_img_path, "rb") as f:
    r = requests.post(
        "http://localhost:8000/api/analyze",
        files={"image": ("test.jpg", f, "image/jpeg")},
        data={"include_gradcam": "true", "include_shap": "true"},
        timeout=120,
    )

result = r.json()

# Summarize without huge fields
print(f"  Verdict:    {result.get('verdict')}")
print(f"  Confidence: {result.get('confidence')}")
print(f"  Scores:")
for k, v in result.get("scores", {}).items():
    print(f"    {k}: {v}")
print(f"  Reasoning:")
for reason in result.get("explanation", {}).get("reasoning", []):
    print(f"    - {reason[:100]}")
has_gradcam = bool(result.get("explanation", {}).get("gradcam_heatmap"))
print(f"  GradCAM heatmap: {'yes' if has_gradcam else 'no'}")
has_freq = bool(result.get("explanation", {}).get("frequency_profile"))
print(f"  Frequency data:  {'yes' if has_freq else 'no'}")
has_shap = bool(result.get("explanation", {}).get("shap_features"))
print(f"  Feature importance: {'yes' if has_shap else 'no'}")
print(f"  Processing time: {result.get('metadata', {}).get('processing_time_ms')}ms")
print(f"  Models used: {result.get('metadata', {}).get('models_used')}")

# Cleanup
os.remove(test_img_path)
print("\n✅ All tests passed!")
