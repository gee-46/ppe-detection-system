"""
Detection analysis utilities.

Originally a standalone script that ran YOLO on a fixed "bus.jpg" and
printed each detection. Refactored into reusable analysis functions on
top of the shared ModelService so the same summary logic can be used by
the API, tests, or ad-hoc CLI runs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Dict, List

import cv2

from app.model_service import Detection, get_model_service


def summarize_detections(detections: List[Detection]) -> Dict:
    """Produce a compact summary: counts per class and average confidence per class."""
    counts = Counter(d.class_name for d in detections)
    conf_totals: Dict[str, float] = {}
    for d in detections:
        conf_totals[d.class_name] = conf_totals.get(d.class_name, 0.0) + d.confidence

    summary = {
        "total_detections": len(detections),
        "unique_classes": len(counts),
        "per_class": {
            name: {
                "count": count,
                "avg_confidence": round(conf_totals[name] / count, 4),
            }
            for name, count in counts.items()
        },
    }
    return summary


def analyze_image(image_path: str) -> Dict:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    service = get_model_service()
    detections = service.predict(image)
    return summarize_detections(detections)


def main():
    parser = argparse.ArgumentParser(description="Analyze detections on an image.")
    parser.add_argument("image", help="Path to input image")
    args = parser.parse_args()

    summary = analyze_image(args.image)
    print(f"\nTotal detections: {summary['total_detections']}")
    print(f"Unique classes: {summary['unique_classes']}")
    for name, stats in summary["per_class"].items():
        print(f"  - {name}: count={stats['count']} avg_confidence={stats['avg_confidence']:.2f}")


if __name__ == "__main__":
    main()
