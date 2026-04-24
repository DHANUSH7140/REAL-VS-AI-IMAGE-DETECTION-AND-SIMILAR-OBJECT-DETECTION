import requests
import json

def get_datasets():
    url = "https://huggingface.co/api/datasets?search=diffusion&sort=downloads&direction=-1&limit=50&filter=task_categories:text-to-image"
    r = requests.get(url)
    data = r.json()
    for d in data:
        print(d.get('id', 'Unknown'))

get_datasets()
