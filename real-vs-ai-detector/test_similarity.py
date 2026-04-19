"""Quick end-to-end test for the similarity search API."""
import requests, json, sys

BASE = "http://127.0.0.1:5000"

def test_similarity():
    print("=== Similarity Search API Test ===")
    
    # Use an image with multiple objects (people/faces)
    image_path = "dataset_v2/real/real_00001.jpg"
    
    # First, let's see what YOLO detects
    url = f"{BASE}/predict/similar"
    
    with open(image_path, "rb") as f:
        resp = requests.post(url, files={"file": ("test.jpg", f, "image/jpeg")}, data={
            "x1": "10",
            "y1": "10", 
            "x2": "100",
            "y2": "100",
            "threshold": "0.5",
        })
    
    print(f"  Status    : {resp.status_code}")
    data = resp.json()
    
    if resp.status_code != 200:
        print(f"  Error     : {data.get('error')}")
        return False
    
    print(f"  Mode      : {data.get('mode')}")
    print(f"  ROI       : {data.get('roi')}")
    print(f"  Detected  : {data.get('total_detected')} objects")
    print(f"  Matches   : {data.get('match_count')}")
    print(f"  Threshold : {data.get('threshold')}")
    print(f"  Annotated : {data.get('annotated_image_url')}")
    print(f"  Original  : {data.get('original_image_url')}")
    
    if data.get("matches"):
        print(f"\n  Top matches:")
        for m in data["matches"][:5]:
            print(f"    - {m['class_name']}: {m['similarity']*100:.1f}% (box={m['box']})")
    
    print()
    return True

def test_api_find_similar():
    print("=== REST API /api/find_similar Test ===")
    
    image_path = "dataset_v2/real/real_00001.jpg"
    url = f"{BASE}/api/find_similar"
    
    with open(image_path, "rb") as f:
        resp = requests.post(url, files={"file": ("test.jpg", f, "image/jpeg")}, data={
            "x1": "20",
            "y1": "20",
            "x2": "80",
            "y2": "80",
            "threshold": "0.6",
        })
    
    print(f"  Status    : {resp.status_code}")
    data = resp.json()
    
    if resp.status_code != 200:
        print(f"  Error     : {data.get('error')}")
        return False
    
    print(f"  Detected  : {data.get('total_detected')}")
    print(f"  Matches   : {data.get('match_count')}")
    print(f"  Annotated : {data.get('annotated_image_url')}")
    print()
    return True

ok1 = test_similarity()
ok2 = test_api_find_similar()

if ok1 and ok2:
    print("All similarity tests passed!")
    sys.exit(0)
else:
    print("Some tests failed.")
    sys.exit(1)
