"""
Central configuration for the PPE Detection System.

The application uses the custom-trained PPE YOLO model by default.

The model was trained using the verified:
Kaggle: shlokraval/ppe-dataset-yolov8

Dataset classes:
    Fall-Detected
    Gloves
    Goggles
    Hardhat
    Ladder
    Mask
    NO-Gloves
    NO-Goggles
    NO-Hardhat
    NO-Mask
    NO-Safety Vest
    Person
    Safety Cone
    Safety Vest
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Default: custom PPE-trained YOLO model.
#
# MODEL_PATH can still be overridden through an environment variable.
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(BASE_DIR / "models" / "trained" / "best.pt"),
)

# Indicates whether the loaded model is the custom PPE model.
IS_CUSTOM_PPE_MODEL = (
    os.environ.get("IS_CUSTOM_PPE_MODEL", "true").lower() == "true"
)

# "cpu", "cuda", "cuda:0", or "auto".
DEVICE = os.environ.get("DEVICE", "auto")

# Minimum confidence for a detection to be kept.
CONFIDENCE_THRESHOLD = float(
    os.environ.get("CONFIDENCE_THRESHOLD", "0.5")
)

# IoU threshold used for NMS.
IOU_THRESHOLD = float(
    os.environ.get("IOU_THRESHOLD", "0.45")
)

# ---------------------------------------------------------------------------
# PPE compliance configuration
# ---------------------------------------------------------------------------

# Dataset uses "Person" with a capital P.
PERSON_CLASS_NAMES = [
    c.strip()
    for c in os.environ.get(
        "PERSON_CLASS_NAMES",
        "Person",
    ).split(",")
    if c.strip()
]

# PPE classes that indicate required protective equipment.
#
# This is configurable through the environment so the compliance
# rules can be changed without modifying application code.
REQUIRED_PPE_CLASSES = [
    c.strip()
    for c in os.environ.get(
        "REQUIRED_PPE_CLASSES",
        "Hardhat,Gloves,Goggles,Mask,Safety Vest",
    ).split(",")
    if c.strip()
]

# Explicitly represent classes indicating missing PPE.
MISSING_PPE_CLASSES = [
    c.strip()
    for c in os.environ.get(
        "MISSING_PPE_CLASSES",
        "NO-Gloves,NO-Goggles,NO-Hardhat,NO-Mask,NO-Safety Vest",
    ).split(",")
    if c.strip()
]

# ---------------------------------------------------------------------------
# Paths / storage
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        str(BASE_DIR / "outputs"),
    )
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATASET_CONFIG_PATH = os.environ.get(
    "DATASET_CONFIG_PATH",
    str(BASE_DIR / "data" / "ppe.yaml"),
)

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------

API_HOST = os.environ.get(
    "API_HOST",
    "0.0.0.0",
)

API_PORT = int(
    os.environ.get(
        "API_PORT",
        "8000",
    )
)

CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS",
        "*",
    ).split(",")
    if o.strip()
]

MAX_UPLOAD_SIZE_MB = int(
    os.environ.get(
        "MAX_UPLOAD_SIZE_MB",
        "50",
    )
)

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}

# Analyze every Nth video frame rather than every frame.
# This keeps CPU-only deployments reasonably fast.
VIDEO_FRAME_SAMPLE_RATE = int(
    os.environ.get(
        "VIDEO_FRAME_SAMPLE_RATE",
        "5",
    )
)