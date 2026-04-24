import re

with open('visionprobe/backend/train_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

replacement = '''def download_ai_images(output_dir: str, count: int) -> str:
    """Download AI-generated images from DiffusionDB directly via Zip files."""
    import requests
    import zipfile
    import io
    
    ai_dir = os.path.join(output_dir, "ai")
    os.makedirs(ai_dir, exist_ok=True)

    existing = len(collect_images(ai_dir, shuffle=False))
    if existing >= count:
        logger.info(f"  AI images already exist ({existing} found)")
        return ai_dir

    needed = count - existing
    downloaded = 0
    
    # DiffusionDB has parts from 000001 to 002000. Each part has 1000 images.
    part = 1
    
    while downloaded < needed and part <= 2000:
        url = f"https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-{part:06d}.zip"
        logger.info(f"  Downloading DiffusionDB part {part}...")
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    for filename in z.namelist():
                        if downloaded >= needed:
                            break
                        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                            try:
                                with z.open(filename) as f_img:
                                    img = Image.open(f_img).convert("RGB")
                                    img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
                                    save_path = os.path.join(ai_dir, f"diffusion_{existing + downloaded:06d}.jpg")
                                    img.save(save_path, "JPEG", quality=95)
                                    downloaded += 1
                                    if downloaded % 100 == 0:
                                        logger.info(f"    Downloaded {downloaded}/{needed} AI images")
                            except Exception as e:
                                continue
            else:
                logger.warning(f"  Failed to download part {part}: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"  Error downloading part {part}: {e}")
        
        part += 1

    logger.info(f"  Total AI images downloaded: {downloaded}")
    return ai_dir'''

content = re.sub(r'def download_ai_images\(output_dir: str, count: int\) -> str:.*?return ai_dir', replacement, content, flags=re.DOTALL)

with open('visionprobe/backend/train_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)
