import requests
import json

url = "http://127.0.0.1:5000/predict"
image_path = r"d:\SEM 4 PROJECT\real-vs-ai-detector\real-vs-ai-detector\dataset_v2\real\real_00000.jpg"

with open(image_path, "rb") as f:
    files = {"file": f}
    data = {"model": "ensemble"}
    response = requests.post(url, files=files, data=data)

print(json.dumps(response.json(), indent=2))
