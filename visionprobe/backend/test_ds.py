import logging
from datasets import load_dataset
logging.basicConfig(level=logging.INFO)

try:
    ds = load_dataset('poloclub/diffusiondb', '2m_random_1k', split='train', streaming=True, trust_remote_code=True)
    item = next(iter(ds))
    img = item.get('image')
    print(f"Success! Image size: {img.size}")
except Exception as e:
    print(f"Failed: {e}")
