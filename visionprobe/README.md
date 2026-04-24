# VisionProbe

**"See what AI can't hide"**

VisionProbe is a production-ready web application featuring a state-of-the-art AI image detection system:

**AI vs Real Image Detector**: An ensemble of 6 models (CLIP, SwinV2, EfficientNet, CNNDetection, Frequency Analysis, SRM Noise) with a LightGBM meta-learner to detect generative AI images.

## Features
- **Detector**: 
  - Ensemble voting for robust detection.
  - GradCAM++ explainability maps.
  - SHAP feature importance.
  - 1D and 2D DCT Frequency analysis.
- **Architecture**:
  - FastAPI asynchronous backend.
  - Thread-safe lazy-loading Singleton Model Cache.
  - Graceful degradation (works even if models are missing).

## Setup Instructions

### Prerequisites
- Python 3.11+
- CUDA Toolkit (if using GPU).

### Installation
1. Navigate to the `backend` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Download required model weights into `backend/weights/`. If weights are missing, the system uses robust fallback mechanisms automatically.

### Running the Server
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
Then navigate to `http://localhost:8000` in your browser.

### Docker
Alternatively, deploy using Docker Compose:
```bash
cd backend
docker-compose up --build
```
