"""
Training pipeline for the custom PPE YOLO model.

Dataset:
    Kaggle: shlokraval/ppe-dataset-yolov8

Verified dataset:
    - 30,765 training images
    - 8,814 validation images
    - 4,423 test images
    - 14 object classes

The training pipeline is designed for:
    - NVIDIA CUDA GPUs
    - Laptop GPUs such as RTX 3050 6GB
    - CPU fallback
    - Reduced DataLoader memory usage
    - Safe checkpoint generation
    - Resume-friendly training

Typical GPU training:
    python -m app.train --epochs 50 --imgsz 640 --batch 4 --workers 2 --device 0

Smoke test:
    python -m app.train --smoke-test

Training artifacts:
    runs/detect/<name>/

Stable checkpoints:
    models/trained/best.pt
    models/trained/last.pt
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ppe.train")

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Project directories
# ---------------------------------------------------------------------------

RUNS_DIR = BASE_DIR / "runs" / "detect"
TRAINED_DIR = BASE_DIR / "models" / "trained"


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """
    Automatically select CUDA when available.

    Returns:
        "0"   -> first NVIDIA CUDA GPU
        "cpu" -> CPU fallback
    """

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)

            logger.info(
                "CUDA GPU detected: %s",
                gpu_name,
            )

            return "0"

    except ImportError:
        logger.warning(
            "PyTorch is not installed. Falling back to CPU."
        )

    return "cpu"


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def validate_dataset(data_yaml: str) -> dict:
    """
    Validate the YOLO dataset before training.

    Supports:
        - relative dataset paths
        - absolute dataset paths

    Prevents training from accidentally starting against
    an empty or incorrect dataset.
    """

    import yaml

    path = Path(data_yaml)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_yaml}"
        )

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    configured_path = Path(
        cfg.get("path", ".")
    )

    # Absolute path:
    # C:/Users/Dell/.cache/kagglehub/...
    if configured_path.is_absolute():
        dataset_root = configured_path

    # Relative path:
    # data/ppe_dataset
    else:
        dataset_root = BASE_DIR / configured_path

    train_dir = dataset_root / cfg.get(
        "train",
        "images/train",
    )

    val_dir = dataset_root / cfg.get(
        "val",
        "images/val",
    )

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    # -----------------------------------------------------------------------
    # Training images
    # -----------------------------------------------------------------------

    if train_dir.exists():

        train_images = [
            p
            for p in train_dir.iterdir()
            if (
                p.is_file()
                and p.suffix.lower() in image_extensions
            )
        ]

    else:
        train_images = []

    # -----------------------------------------------------------------------
    # Validation images
    # -----------------------------------------------------------------------

    if val_dir.exists():

        val_images = [
            p
            for p in val_dir.iterdir()
            if (
                p.is_file()
                and p.suffix.lower() in image_extensions
            )
        ]

    else:
        val_images = []

    # -----------------------------------------------------------------------
    # Fail fast
    # -----------------------------------------------------------------------

    if not train_images:

        raise RuntimeError(
            f"No training images found at {train_dir}. "
            "Check data/ppe.yaml and make sure the verified "
            "PPE dataset is available."
        )

    if not val_images:

        logger.warning(
            "No validation images found at %s. "
            "Evaluation will be unreliable.",
            val_dir,
        )

    logger.info(
        "Dataset OK: %d train images, %d val images, classes=%s",
        len(train_images),
        len(val_images),
        cfg.get("names"),
    )

    return cfg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 4,
    device: str | None = None,
    name: str = "ppe_train",
    workers: int = 2,
    cache: bool = False,
):
    """
    Train YOLO on the verified PPE dataset.

    Memory-safe defaults are intentionally used for laptop GPUs:

        batch   = 4
        workers = 2
        cache   = False

    This reduces system RAM pressure while keeping GPU training enabled.
    """

    from ultralytics import YOLO

    # -----------------------------------------------------------------------
    # 1. Validate dataset
    # -----------------------------------------------------------------------

    validate_dataset(data_yaml)

    # -----------------------------------------------------------------------
    # 2. Detect device
    # -----------------------------------------------------------------------

    device = device or detect_device()

    if device == "cpu":

        logger.warning(
            "No GPU detected — training on CPU will be slow."
        )

    else:

        logger.info(
            "Training on CUDA device: %s",
            device,
        )

    # -----------------------------------------------------------------------
    # 3. Log configuration
    # -----------------------------------------------------------------------

    logger.info(
        "Starting training:"
        " model=%s"
        " epochs=%d"
        " imgsz=%d"
        " batch=%d"
        " workers=%d"
        " cache=%s"
        " device=%s",
        base_model,
        epochs,
        imgsz,
        batch,
        workers,
        cache,
        device,
    )

    # -----------------------------------------------------------------------
    # 4. Prepare directories
    # -----------------------------------------------------------------------

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRAINED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # 5. Load model
    # -----------------------------------------------------------------------

    logger.info(
        "Loading YOLO model from: %s",
        base_model,
    )

    model = YOLO(base_model)

    # -----------------------------------------------------------------------
    # 6. Train
    # -----------------------------------------------------------------------

    results = model.train(

        data=data_yaml,

        epochs=epochs,

        imgsz=imgsz,

        batch=batch,

        device=device,

        project=str(RUNS_DIR),

        name=name,

        # ---------------------------------------------------------------
        # Memory safety
        # ---------------------------------------------------------------

        workers=workers,

        cache=cache,

        # ---------------------------------------------------------------
        # Checkpoint safety
        # ---------------------------------------------------------------

        save=True,

        save_period=1,

        # ---------------------------------------------------------------
        # Keep existing run directory
        # ---------------------------------------------------------------

        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # 7. Locate training artifacts
    # -----------------------------------------------------------------------

    run_dir = RUNS_DIR / name

    best = (
        run_dir
        / "weights"
        / "best.pt"
    )

    last = (
        run_dir
        / "weights"
        / "last.pt"
    )

    logger.info(
        "Training directory: %s",
        run_dir,
    )

    # -----------------------------------------------------------------------
    # 8. Copy best checkpoint
    # -----------------------------------------------------------------------

    if best.exists():

        shutil.copy2(
            best,
            TRAINED_DIR / "best.pt",
        )

        logger.info(
            "Copied best.pt -> %s",
            TRAINED_DIR / "best.pt",
        )

    else:

        logger.warning(
            "best.pt was not found at %s",
            best,
        )

    # -----------------------------------------------------------------------
    # 9. Copy last checkpoint
    # -----------------------------------------------------------------------

    if last.exists():

        shutil.copy2(
            last,
            TRAINED_DIR / "last.pt",
        )

        logger.info(
            "Copied last.pt -> %s",
            TRAINED_DIR / "last.pt",
        )

    else:

        logger.warning(
            "last.pt was not found at %s",
            last,
        )

    # -----------------------------------------------------------------------
    # 10. Final status
    # -----------------------------------------------------------------------

    logger.info(
        "Training complete."
    )

    logger.info(
        "Training results: %s",
        run_dir,
    )

    logger.info(
        "Best model: %s",
        TRAINED_DIR / "best.pt",
    )

    logger.info(
        "Last model: %s",
        TRAINED_DIR / "last.pt",
    )

    return results


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Train the custom PPE YOLO model."
    )

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--data",
        default=str(
            BASE_DIR
            / "data"
            / "ppe.yaml"
        ),
        help="Path to dataset YAML",
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base YOLO model to fine-tune",
    )

    # -----------------------------------------------------------------------
    # Training parameters
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Training batch size",
    )

    # -----------------------------------------------------------------------
    # GPU
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "'cpu', '0' for first GPU, etc. "
            "Auto-detected when omitted."
        ),
    )

    # -----------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "Number of DataLoader workers. "
            "Keep low on laptops to reduce RAM usage."
        ),
    )

    # -----------------------------------------------------------------------
    # Dataset caching
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--cache",
        action="store_true",
        help=(
            "Cache dataset images in RAM. "
            "Disabled by default to reduce memory usage."
        ),
    )

    # -----------------------------------------------------------------------
    # Run name
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--name",
        default="ppe_train",
        help="Training run name",
    )

    # -----------------------------------------------------------------------
    # Smoke test
    # -----------------------------------------------------------------------

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run a small 1-epoch training job "
            "to verify the complete pipeline."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Default training parameters
    # -----------------------------------------------------------------------

    epochs = args.epochs

    imgsz = args.imgsz

    batch = args.batch

    # -----------------------------------------------------------------------
    # Smoke-test configuration
    # -----------------------------------------------------------------------

    if args.smoke_test:

        epochs = 1

        imgsz = 320

        batch = 4

        logger.info(
            "Smoke-test mode: "
            "epochs=1 "
            "imgsz=320 "
            "batch=4 "
            "workers=2 "
            "cache=False"
        )

    # -----------------------------------------------------------------------
    # Start training
    # -----------------------------------------------------------------------

    try:

        train(

            data_yaml=args.data,

            base_model=args.model,

            epochs=epochs,

            imgsz=imgsz,

            batch=batch,

            device=args.device,

            name=args.name,

            workers=args.workers,

            cache=args.cache,
        )

    except (
        FileNotFoundError,
        RuntimeError,
    ) as exc:

        logger.error(
            "Training aborted: %s",
            exc,
        )

        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()