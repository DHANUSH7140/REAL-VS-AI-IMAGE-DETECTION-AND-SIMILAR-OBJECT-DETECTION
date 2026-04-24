import requests

url = "http://127.0.0.1:5001/predict"
# Create a dummy image
with open("dummy.jpg", "wb") as f:
    f.write(b"dummy image data")

with open("dummy.jpg", "rb") as f:
    files = {"file": f}
    data = {"model": "ensemble"}
    try:
        response = requests.post(url, files=files, data=data)
        print("Status Code:", response.status_code)
        print("Response:", response.text)
    except Exception as e:
        print("Request failed:", e)
