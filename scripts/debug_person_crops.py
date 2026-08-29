"""
Debug PPE detection on individual tracked-person crops.

Purpose:
    Determine whether the custom PPE model can detect PPE
    when each person is cropped from the full frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.model_service import get_model_service
from app.tracker import PersonTracker
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "multi_person_test.mp4"
)

PERSON_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolov8n.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "person_crops"
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("=" * 70)
    print("PERSON CROP PPE DEBUG")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------
    # Load models
    # ----------------------------------------------------------------------

    print()
    print("Loading PPE model...")

    ppe_model = get_model_service()

    print(
        "PPE Model:",
        ppe_model.model_path,
    )

    print()
    print("Loading Person model...")

    person_yolo = YOLO(
        str(PERSON_MODEL_PATH)
    )

    print(
        "Person Model:",
        PERSON_MODEL_PATH,
    )

    # ----------------------------------------------------------------------
    # Tracker
    # ----------------------------------------------------------------------

    tracker = PersonTracker(
        model=ppe_model,
        person_model=person_yolo,
        confidence_threshold=0.20,
        iou_threshold=0.70,
    )

    # ----------------------------------------------------------------------
    # Video
    # ----------------------------------------------------------------------

    capture = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open:\n{VIDEO_PATH}"
        )

    # ----------------------------------------------------------------------
    # Read first frame
    # ----------------------------------------------------------------------

    ok, frame = capture.read()

    if not ok:

        raise RuntimeError(
            "Could not read first video frame."
        )

    capture.release()

    # ----------------------------------------------------------------------
    # Run person tracking
    # ----------------------------------------------------------------------

    result = tracker.update(frame)

    print()
    print("=" * 70)
    print("TRACKED PEOPLE")
    print("=" * 70)

    for person in result.tracked_people:

        print(
            f"ID={person.track_id} "
            f"conf={person.confidence:.4f} "
            f"box="
            f"{[round(v, 1) for v in person.box_xyxy]}"
        )

    # ----------------------------------------------------------------------
    # Run PPE model on each person crop
    # ----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("PERSON CROP PPE DETECTIONS")
    print("=" * 70)

    for person in result.tracked_people:

        x1, y1, x2, y2 = [
            int(v)
            for v in person.box_xyxy
        ]

        # ---------------------------------------------------------------
        # Add padding around person
        # ---------------------------------------------------------------

        width = x2 - x1
        height = y2 - y1

        pad_x = int(width * 0.20)
        pad_y = int(height * 0.10)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame.shape[1], x2 + pad_x)
        y2 = min(frame.shape[0], y2 + pad_y)

        crop = frame[
            y1:y2,
            x1:x2,
        ]

        if crop.size == 0:
            continue

        # ---------------------------------------------------------------
        # Save crop
        # ---------------------------------------------------------------

        crop_path = (
            OUTPUT_DIR
            / f"person_{person.track_id}.jpg"
        )

        cv2.imwrite(
            str(crop_path),
            crop,
        )

        # ---------------------------------------------------------------
        # PPE inference
        # ---------------------------------------------------------------

        detections = ppe_model.predict(
            crop,
            confidence_threshold=0.01,
            iou_threshold=0.70,
        )

        print()
        print(
            f"PERSON ID={person.track_id}"
        )

        print(
            f"Crop: "
            f"{crop.shape[1]}x{crop.shape[0]}"
        )

        print(
            f"Saved: {crop_path}"
        )

        if not detections:

            print(
                "  NO PPE DETECTIONS"
            )

            continue

        for detection in detections:

            print(
                f"  "
                f"{detection.class_name:<20}"
                f"conf={detection.confidence:.4f} "
                f"box="
                f"{[round(v, 1) for v in detection.box_xyxy]}"
            )

    print()
    print("=" * 70)
    print("CROP DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()