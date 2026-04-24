"""
Shared preprocessing utilities for the detector module.
"""

import hashlib
import io
import logging

import numpy as np
import torch
from PIL import Image, ExifTags
from torchvision import transforms

logger = logging.getLogger("visionprobe.preprocessing")

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def decode_image(image_bytes: bytes) -> tuple:
    """
    Decode raw bytes → (PIL.Image in RGB, numpy RGB array).
    Handles JPEG, PNG, WEBP, BMP, TIFF.
    """
    pil_img = Image.open(io.BytesIO(image_bytes))
    if pil_img.mode == "RGBA":
        background = Image.new("RGB", pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        pil_img = background
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    np_img = np.array(pil_img)
    return pil_img, np_img


def extract_exif(image_bytes: bytes) -> dict:
    """
    Extract EXIF metadata from image bytes.
    Returns dict of tag_name → value. Empty dict if no EXIF found.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        raw_exif = img._getexif()
        if raw_exif is None:
            return {}
        result = {}
        for tag_id, value in raw_exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            # Convert bytes to string for JSON serialization
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8", errors="replace")
                except Exception:
                    value = str(value)
            elif isinstance(value, tuple) and len(value) == 2:
                # IFDRational
                try:
                    value = float(value[0]) / float(value[1]) if value[1] != 0 else 0.0
                except Exception:
                    value = str(value)
            result[tag_name] = value
        return result
    except Exception:
        return {}


def resize_for_processing(image: Image.Image, max_side: int = 2048) -> Image.Image:
    """Resize image if largest side exceeds max_side, maintaining aspect ratio."""
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def image_to_tensor(image: Image.Image, size: int = 448) -> torch.Tensor:
    """
    PIL Image → normalized torch tensor suitable for model input.
    Resizes to (size, size), applies ImageNet mean/std normalization.
    Returns tensor of shape (1, 3, size, size).
    """
    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(image).unsqueeze(0)  # (1, 3, size, size)
    return tensor


def compute_image_hash(image_bytes: bytes) -> str:
    """SHA256 hash of raw image bytes for caching."""
    return hashlib.sha256(image_bytes).hexdigest()
