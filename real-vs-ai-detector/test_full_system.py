"""Full system test for all new features."""
import requests, json, sys, time

BASE = "http://127.0.0.1:5000"
IMAGE = "dataset_v2/real/real_00001.jpg"

def test_health():
    print("=== /api/health ===")
    r = requests.get(f"{BASE}/api/health")
    d = r.json()
    print(f"  Status:  {r.status_code} -> {d.get('status')}")
    print(f"  Memory:  {d.get('memory_mb')} MB")
    print(f"  Features: {json.dumps(d.get('features', {}), indent=4)}")
    return r.status_code == 200

def test_similarity():
    print("\n=== /predict/similar (ViT + FAISS) ===")
    with open(IMAGE, "rb") as f:
        r = requests.post(f"{BASE}/predict/similar",
            files={"file": ("test.jpg", f)},
            data={"x1": "10", "y1": "10", "x2": "100", "y2": "100", "threshold": "0.5"})
    d = r.json()
    print(f"  Status:    {r.status_code}")
    print(f"  Backbone:  {d.get('embedding_source', 'N/A')}")
    print(f"  FAISS:     {d.get('faiss_used', 'N/A')}")
    print(f"  Detected:  {d.get('total_detected')}")
    print(f"  Matches:   {d.get('match_count')}")
    print(f"  Time:      {d.get('processing_time_ms')}ms")
    return r.status_code == 200

def test_patch():
    print("\n=== /predict/patch ===")
    with open(IMAGE, "rb") as f:
        r = requests.post(f"{BASE}/predict/patch",
            files={"file": ("test.jpg", f)},
            data={"patch_size": "64"})
    d = r.json()
    print(f"  Status:    {r.status_code}")
    print(f"  Mode:      {d.get('mode')}")
    print(f"  Patches:   {d.get('patch_count')}")
    print(f"  Grid:      {d.get('grid_size')}")
    print(f"  Score:     {d.get('overall_score')}")
    print(f"  Artifacts: {len(d.get('artifact_regions', []))}")
    print(f"  Heatmap:   {d.get('heatmap_url')}")
    print(f"  Summary:   {d.get('summary')}")
    print(f"  Time:      {d.get('processing_time_ms')}ms")
    return r.status_code == 200

def test_standard():
    print("\n=== /predict (ensemble) ===")
    with open(IMAGE, "rb") as f:
        r = requests.post(f"{BASE}/predict",
            files={"file": ("test.jpg", f)},
            data={"model": "efficientnet"})
    d = r.json()
    print(f"  Status:    {r.status_code}")
    print(f"  Label:     {d.get('label')}")
    print(f"  Conf:      {d.get('confidence')}%")
    return r.status_code == 200


results = []
results.append(("Health", test_health()))
results.append(("Standard", test_standard()))
results.append(("Similarity", test_similarity()))
results.append(("Patch", test_patch()))

print("\n" + "=" * 50)
all_ok = True
for name, ok in results:
    status = "[PASS]" if ok else "[FAIL]"
    print(f"  {status}  {name}")
    if not ok: all_ok = False

print("=" * 50)
sys.exit(0 if all_ok else 1)
