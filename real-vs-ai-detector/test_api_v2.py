import requests
import os

url = "http://127.0.0.1:5000/predict"
# Pick a random image from the dataset to test
img_path = "dataset_v2/real/real_00000.jpg"

if not os.path.exists(img_path):
    print(f"Error: {img_path} not found.")
    exit(1)

files = {"file": open(img_path, "rb")}
data = {"model": "efficientnet"}

try:
    response = requests.post(url, files=files, data=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error during request: {e}")
