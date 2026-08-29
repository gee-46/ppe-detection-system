"""
Debug two-model Person + PPE association.

Uses:

    yolov8n.pt
        -> Person detection + ByteTrack

    best.pt
        -> PPE detection

Then prints which PPE detections are associated with
each tracked person.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.model_service import get_model_service
from app.tracker import PersonTracker


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "multi_person_test.mp4"
)


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def box_iou(
    box_a,
    box_b,
) -> float:

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ---------------------------------------------------------------------------
# Center helper
# ---------------------------------------------------------------------------

def center_inside(
    person_box,
    ppe_box,
) -> bool:

    px1, py1, px2, py2 = person_box
    x1, y1, x2, y2 = ppe_box

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    return (
        px1 <= cx <= px2
        and
        py1 <= cy <= py2
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("=" * 70)
    print("TWO-MODEL PERSON + PPE ASSOCIATION DEBUG")
    print("=" * 70)

    print()
    print("Video:")
    print(VIDEO_PATH)

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------

    print()
    print("Loading PPE model...")

    ppe_model = get_model_service()

    print(
        "PPE Model:",
        ppe_model.model_path,
    )

    print()
    print("Loading Person model...")

    person_model_path = (
        PROJECT_ROOT
        / "models"
        / "yolov8n.pt"
    )

    person_model = YOLO(
        str(person_model_path)
    )

    print(
        "Person Model:",
        person_model_path,
    )

    # -----------------------------------------------------------------------
    # Tracker
    # -----------------------------------------------------------------------

    tracker = PersonTracker(
        model=ppe_model,
        person_model=person_model,
        confidence_threshold=0.20,
    )

    # -----------------------------------------------------------------------
    # Open video
    # -----------------------------------------------------------------------

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open:\n{VIDEO_PATH}"
        )

    # -----------------------------------------------------------------------
    # Test selected frames
    # -----------------------------------------------------------------------

    test_frames = {
        1,
        25,
        50,
        75,
        100,
    }

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        if frame_number not in test_frames:
            continue

        result = tracker.update(
            frame
        )

        print()
        print("=" * 70)
        print(
            f"FRAME {frame_number}"
        )
        print("=" * 70)

        # -------------------------------------------------------------------
        # People
        # -------------------------------------------------------------------

        print()
        print("TRACKED PEOPLE")
        print("-" * 70)

        for person in result.tracked_people:

            print(
                f"ID={person.track_id} "
                f"conf={person.confidence:.4f} "
                f"box={[round(v, 1) for v in person.box_xyxy]}"
            )

        # -------------------------------------------------------------------
        # PPE
        # -------------------------------------------------------------------

        print()
        print("PPE DETECTIONS")
        print("-" * 70)

        for detection in result.detections:

            print(
                f"{detection.class_name:<18} "
                f"conf={detection.confidence:.4f} "
                f"box={[round(v, 1) for v in detection.box_xyxy]}"
            )

        # -------------------------------------------------------------------
        # Association
        # -------------------------------------------------------------------

        print()
        print("ASSOCIATION")
        print("-" * 70)

        for person in result.tracked_people:

            print()
            print(
                f"PERSON ID={person.track_id}"
            )

            print(
                "Person box:",
                [
                    round(v, 1)
                    for v in person.box_xyxy
                ],
            )

            found = False

            for detection in result.detections:

                if detection.class_name in {
                    "Person",
                    "person",
                }:
                    continue

                inside = center_inside(
                    person.box_xyxy,
                    detection.box_xyxy,
                )

                iou = box_iou(
                    person.box_xyxy,
                    detection.box_xyxy,
                )

                if inside or iou >= 0.01:

                    found = True

                    print(
                        f"  {detection.class_name:<18} "
                        f"conf={detection.confidence:.4f} "
                        f"center_inside={inside} "
                        f"IoU={iou:.4f} "
                        f"box={[round(v, 1) for v in detection.box_xyxy]}"
                    )

            if not found:

                print(
                    "  NO PPE ASSOCIATED"
                )

    cap.release()

    tracker.reset()

    print()
    print("=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()