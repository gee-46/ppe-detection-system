"""
Debug tracker + PPE compliance association.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

TEST_DIR = PROJECT_ROOT / "data" / "ppe_tests"

TEST_IMAGES = [
    TEST_DIR / "hardhat_test.jpg",
    TEST_DIR / "vest_test.jpg",
    TEST_DIR / "mask_test.jpg",
    TEST_DIR / "violation_test.jpg",
]

REQUIRED_PPE = [
    "Hardhat",
    "Gloves",
    "Goggles",
    "Mask",
    "Safety Vest",
]


def main():

    print("=" * 70)
    print("TRACKER + PPE COMPLIANCE DEBUG")
    print("=" * 70)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Loading model...")

    model_service = get_model_service()

    print("Model:")
    print(model_service.model_path)

    tracker = PersonTracker(
        model=model_service,
        confidence_threshold=0.10,
        iou_threshold=0.7,
    )

    manager = TrackComplianceManager(
        required_ppe_classes=REQUIRED_PPE,
    )

    import cv2

    for image_path in TEST_IMAGES:

        print()
        print("=" * 70)
        print(f"IMAGE: {image_path.name}")
        print("=" * 70)

        if not image_path.exists():
            print("ERROR: Image does not exist:")
            print(image_path)
            continue

        frame = cv2.imread(str(image_path))

        if frame is None:
            print("ERROR: Could not read image.")
            continue

        # ---------------------------------------------------------------
        # YOLO + ByteTrack
        # ---------------------------------------------------------------

        tracking = tracker.update(frame)

        print()
        print("TRACKING")
        print("-" * 70)

        print(
            f"Tracked people: "
            f"{len(tracking.tracked_people)}"
        )

        for person in tracking.tracked_people:

            print(
                f"  ID={person.track_id} "
                f"conf={person.confidence:.4f} "
                f"box={person.box_xyxy}"
            )

        # ---------------------------------------------------------------
        # All detections
        # ---------------------------------------------------------------

        print()
        print("ALL DETECTIONS")
        print("-" * 70)

        for detection in tracking.detections:

            print(
                f"  {detection.class_name:<18} "
                f"conf={detection.confidence:.4f} "
                f"box={detection.box_xyxy}"
            )

        # ---------------------------------------------------------------
        # Association
        # ---------------------------------------------------------------

        print()
        print("ASSOCIATION CHECK")
        print("-" * 70)

        for person in tracking.tracked_people:

            print()
            print(
                f"PERSON ID={person.track_id}"
            )

            print(
                f"Person box: {person.box_xyxy}"
            )

            for detection in tracking.detections:

                if detection.class_name == "Person":
                    continue

                if detection.class_name in {
                    "Ladder",
                    "Safety Cone",
                }:
                    continue

                associated = manager._is_inside_person(
                    person.box_xyxy,
                    detection.box_xyxy,
                )

                print(
                    f"  {detection.class_name:<18} "
                    f"conf={detection.confidence:.4f} "
                    f"associated={associated} "
                    f"box={detection.box_xyxy}"
                )

        # ---------------------------------------------------------------
        # Compliance
        # ---------------------------------------------------------------

        compliance = manager.update(
            tracking.tracked_people,
            tracking.detections,
        )

        print()
        print("COMPLIANCE")
        print("-" * 70)

        if not compliance:

            print("NO COMPLIANCE RESULT")

        else:

            for state in compliance:

                print(
                    f"  Track ID    : {state.track_id}"
                )

                print(
                    f"  Compliant   : {state.compliant}"
                )

                print(
                    f"  Present PPE : {state.present_ppe}"
                )

                print(
                    f"  Missing PPE : {state.missing_ppe}"
                )

                print(
                    f"  Violations  : {state.violations}"
                )

        # Reset between independent images.
        tracker.reset()

    print()
    print("=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()