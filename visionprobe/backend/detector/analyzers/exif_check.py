"""
EXIF metadata scoring for AI vs Real detection.
Scores image authenticity based on camera metadata presence and content.
"""

import logging

logger = logging.getLogger("visionprobe.detector.analyzers.exif_check")

AI_SOFTWARE_LIST = [
    "Stable Diffusion", "DALL-E", "Midjourney",
    "Adobe Firefly", "Canva", "NightCafe", "RunwayML",
    "Imagen", "Gemini", "ChatGPT", "Bing Image Creator",
    "DALL·E", "ComfyUI", "Automatic1111", "InvokeAI",
    "Leonardo.AI", "Craiyon", "Jasper Art",
]


def score_exif(exif_meta: dict) -> dict:
    """
    Score image authenticity from EXIF metadata.
    
    Returns dict with:
      - score: float (0=real, 1=AI) — the exif_ai_score
      - real_score: float (raw real-ness score before inversion)
      - fields_found: int
      - has_gps: bool
      - has_camera: bool
      - software_detected: str or None
    """
    real_score = 0.0
    has_gps = False
    has_camera = False
    software_detected = None

    fields_found = len(exif_meta)

    if fields_found == 0:
        # No EXIF - common in AI, but also common if stripped (e.g. WhatsApp, web)
        # Should be neutral, maybe slightly leaning AI (0.6) but not 1.0!
        real_score += 0.40
    else:
        # Camera hardware
        if "Make" in exif_meta or "Model" in exif_meta:
            real_score += 0.30
            has_camera = True

        # GPS data
        gps_keys = ["GPSInfo", "GPSLatitude", "GPSLongitude", "GPSLatitudeRef"]
        if any(k in exif_meta for k in gps_keys):
            real_score += 0.20
            has_gps = True

        # Software check
        software = exif_meta.get("Software", "")
        if isinstance(software, str) and software:
            software_detected = software
            is_ai_software = any(
                ai_name.lower() in software.lower()
                for ai_name in AI_SOFTWARE_LIST
            )
            if is_ai_software:
                real_score -= 0.20
            else:
                real_score += 0.15

        # Date/time
        if "DateTime" in exif_meta or "DateTimeOriginal" in exif_meta:
            real_score += 0.15

        # Camera settings
        camera_keys = ["FocalLength", "ExposureTime", "FNumber", "ISOSpeedRatings"]
        if any(k in exif_meta for k in camera_keys):
            real_score += 0.10

        # Color space
        color_space = exif_meta.get("ColorSpace")
        if color_space == 1 or color_space == "sRGB":
            real_score += 0.10

    # Clamp and invert
    real_score_clamped = max(0.0, min(1.0, real_score))
    exif_ai_score = 1.0 - real_score_clamped

    return {
        "score": exif_ai_score,
        "real_score": real_score_clamped,
        "fields_found": fields_found,
        "has_gps": has_gps,
        "has_camera": has_camera,
        "software_detected": software_detected,
    }
