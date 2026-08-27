"""
Image inference CLI.

Refactored from the original script (which did `YOLO("yolov8n.pt")` and
ran a single hardcoded example) into a reusable function on top of the
shared ModelService, plus a thin CLI entry point.

Usage:
    python -m app.inference path/to/image.jpg
    python -m app.inference path/to/image.jpg --conf 0.4 --out outputs/result.jpg
"""

from __future__ import annotations

import argparse
i   mport logging
import sys
from pathlib import Path

import cv2

from app import config
from app.model_service import get_model_service
from app.ppe_logic import evaluate_compliance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ppe.inference")


def run_image_inference(image_path: str, save_path: str = None, confidence: float = None):
    """Run detection + PPE compliance on a single image file. Returns (detections, compliance, annotated_bgr)."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    service = get_model_service()
    detections, annotated = service.predict_and_annotate(image, confidence_threshold=confidence)
    compliance = evaluate_compliance(detections)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, annotated)
        logger.info("Annotated image saved to %s", save_path)

    return detections, compliance, annotated


def main():
    parser = argparse.ArgumentParser(description="Run PPE/object detection on a single image.")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold override")
    parser.add_argument("--out", default=None, help="Path to save annotated image")
    args = parser.parse_args()

    out_path = args.out or str(config.OUTPUT_DIR / f"annotated_{Path(args.image).name}")

    detections, compliance, _ = run_image_inference(args.image, save_path=out_path, confidence=args.conf)

    print(f"\nDetections ({len(detections)}):")
    for d in detections:
        print(f"  - {d.class_name} | confidence={d.confidence:.2f} | box={[round(v, 1) for v in d.box_xyxy]}")

    print(f"\nCompliance: {compliance.overall_status}")
    if compliance.violations:
        print("Violations:")
        for v in compliance.violations:
            print(f"  - {v}")

    print(f"\nAnnotated output: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
