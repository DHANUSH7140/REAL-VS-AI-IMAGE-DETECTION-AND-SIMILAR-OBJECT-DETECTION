import logging
from datasets import load_dataset
logging.basicConfig(level=logging.ERROR)

def test_ds(name, split="train", img_key="image"):
    try:
        print(f"\nTesting {name}...")
        ds = load_dataset(name, split=split, streaming=True)
        for i, item in enumerate(ds):
            img = item.get(img_key)
            if img:
                print(f"Success! Image size: {img.size}")
                return True
            if i > 2: 
                break
    except Exception as e:
        print(f"Failed: {e}")
    return False

test_ds("hanruijiang/civitai-stable-diffusion-2.5m")
test_ds("jlbaker361/midjourney-v5-images")
test_ds("lambdalabs/pokemon-blip-captions")
test_ds("Kokei/sdxl-high-quality-images")
test_ds("nroggendorff/midjourney-v5")
