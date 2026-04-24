from setuptools import setup, find_packages

setup(
    name="visionprobe",
    version="1.0.0",
    description="VisionProbe: AI vs Real Image Detector",
    author="Expert AI Engineer",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "timm>=0.9.12",
        "transformers>=4.37.0",
        "lightgbm>=4.2.0",
        "numpy",
        "opencv-python-headless",
        "scipy",
        "pillow",
        "python-multipart>=0.0.7",
    ],
    extras_require={
        "explain": ["shap", "grad-cam"],
    }
)
