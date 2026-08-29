"""
Two-model video PPE compliance debug.

Person model:
    YOLOv8n COCO -> Person -> ByteTrack

PPE model:
    Custom trained YOLO -> PPE detections

Compliance:
    Tracked persons + PPE detections
        -> TrackComplianceManager
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


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from app.model_service import get_model_service
from app.tracker import PersonTracker
from app.tracked_compliance import TrackComplianceManager


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("=" * 70)
    print("TWO-MODEL VIDEO PPE COMPLIANCE DEBUG")
    print("=" * 70)

    print()
    print("Video:")
    print(VIDEO_PATH)

    # ======================================================================
    # LOAD CUSTOM PPE MODEL
    # ======================================================================

    print()
    print("Loading PPE model...")

    ppe_model = get_model_service()

    print(
        "PPE Model:",
        ppe_model.model_path,
    )

    # ======================================================================
    # LOAD COCO PERSON MODEL
    # ======================================================================

    print()
    print("Loading Person model...")

    person_yolo = YOLO(
        str(PERSON_MODEL_PATH)
    )

    print(
        "Person Model:",
        PERSON_MODEL_PATH,
    )

    # ======================================================================
    # CREATE TRACKER
    # ======================================================================

    tracker = PersonTracker(
        model=ppe_model,
        person_model=person_yolo,
        confidence_threshold=0.20,
        iou_threshold=0.70,
    )

    # ======================================================================
    # CREATE COMPLIANCE MANAGER
    # ======================================================================

    compliance_manager = (
        TrackComplianceManager()
    )

    # ======================================================================
    # OPEN VIDEO
    # ======================================================================

    capture = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print()
    print(
        f"Total frames: {total_frames}"
    )

    frame_number = 0

    # ======================================================================
    # PROCESS VIDEO
    # ======================================================================

    while True:

        ok, frame = capture.read()

        if not ok:
            break

        frame_number += 1

        # ==============================================================
        # PERSON + BYTETRACK + PPE
        # ==============================================================

        tracking_result = tracker.update(
            frame
        )

        tracked_people = (
            tracking_result.tracked_people
        )

        detections = (
            tracking_result.detections
        )

        # ==============================================================
        # COMPLIANCE
        # ==============================================================

        compliance_results = (
            compliance_manager.update(
                tracked_people,
                detections,
            )
        )

        # ==============================================================
        # DEBUG SELECTED FRAMES
        # ==============================================================

        if frame_number in (
            1,
            25,
            50,
            75,
            100,
        ):

            print()
            print("=" * 70)
            print(
                f"FRAME {frame_number}"
            )
            print("=" * 70)

            # ----------------------------------------------------------
            # TRACKED PEOPLE
            # ----------------------------------------------------------

            print()
            print("TRACKED PEOPLE")
            print("-" * 70)

            for person in tracked_people:

                print(
                    f"ID={person.track_id} "
                    f"conf={person.confidence:.4f} "
                    f"box="
                    f"{[round(v, 1) for v in person.box_xyxy]}"
                )

            # ----------------------------------------------------------
            # PPE DETECTIONS
            # ----------------------------------------------------------

            print()
            print("PPE DETECTIONS")
            print("-" * 70)

            for detection in detections:

                print(
                    f"{detection.class_name:<20} "
                    f"conf={detection.confidence:.4f} "
                    f"box="
                    f"{[round(v, 1) for v in detection.box_xyxy]}"
                )

            # ----------------------------------------------------------
            # COMPLIANCE
            # ----------------------------------------------------------

            print()
            print("COMPLIANCE")
            print("-" * 70)

            for state in compliance_results:

                print(
                    f"Track ID       : "
                    f"{state.track_id}"
                )

                print(
                    f"Person conf    : "
                    f"{state.person_confidence:.4f}"
                )

                print(
                    f"Compliant      : "
                    f"{state.compliant}"
                )

                print(
                    f"Present PPE    : "
                    f"{state.present_ppe}"
                )

                print(
                    f"Missing PPE    : "
                    f"{state.missing_ppe}"
                )

                print(
                    f"Violations     : "
                    f"{state.violations}"
                )

                print()

    capture.release()

    # ======================================================================
    # COMPLETE
    # ======================================================================

    print()
    print("=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()