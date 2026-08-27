# PPE Detection System — backend API container.
#
# Serves the FastAPI app (app/api/main.py) and the static frontend
# (frontend/) from a single container.
#
# Build:
#   docker build -t ppe-detection-system .
# Run:
#   docker run -p 8000:8000 -e MODEL_PATH=/app/models/yolov8n.pt ppe-detection-system
#
# To deploy the real trained model instead of the temporary one, mount it
# and point MODEL_PATH at it, e.g.:
#   docker run -p 8000:8000 \
#     -v $(pwd)/models/trained:/app/models/trained \
#     -e MODEL_PATH=/app/models/trained/best.pt \
#     -e IS_CUSTOM_PPE_MODEL=true \
#     -e REQUIRED_PPE_CLASSES=helmet,vest \
#     ppe-detection-system

FROM python:3.12-slim

# System deps required by opencv-python-headless at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (dataset/, runs/, venv/, outputs/ are excluded via .dockerignore)
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY data/ppe.yaml ./data/ppe.yaml
COPY models/yolov8n.pt ./models/yolov8n.pt

# Runtime-configurable
ENV MODEL_PATH=/app/models/yolov8n.pt
ENV IS_CUSTOM_PPE_MODEL=false
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV OUTPUT_DIR=/app/outputs

RUN mkdir -p /app/outputs /app/models/trained

EXPOSE 8000

# Basic container healthcheck against our own /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
