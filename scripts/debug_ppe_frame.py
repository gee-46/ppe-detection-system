"""
Debug PPE detection on a specific video frame.

Purpose:
    Determine whether Person detections are being suppressed
    by the confidence threshold.

Tests multiple confidence thresholds on the same frame.
"""

from pathlib import Path
import sys

import cv2


# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.model_service import get_model_service


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "person_test.mp4"
)

FRAME_NUMBER = 67

CONFIDENCE_THRESHOLDS = [
    0.01,
    0.05,
    0.10,
    0.20,
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("=" * 70)
    print("PPE MODEL FRAME DIAGNOSTIC")
    print("=" * 70)

    print(f"Video: {VIDEO_PATH}")
    print(f"Frame: {FRAME_NUMBER}")
    print()

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    # ------------------------------------------------------------------
    # Open video
    # ------------------------------------------------------------------

    capture = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        FRAME_NUMBER - 1,
    )

    success, frame = capture.read()

    capture.release()

    if not success:
        raise RuntimeError(
            f"Could not read frame {FRAME_NUMBER}"
        )

    print(
        f"Frame size: "
        f"{frame.shape[1]}x{frame.shape[0]}"
    )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    print()
    print("Loading model...")

    model = get_model_service()

    print(
        f"Model: {model.model_path}"
    )

    print()

    # ------------------------------------------------------------------
    # Run detection at different thresholds
    # ------------------------------------------------------------------

    for threshold in CONFIDENCE_THRESHOLDS:

        print("-" * 70)

        print(
            f"CONFIDENCE THRESHOLD: "
            f"{threshold}"
        )

        detections = model.predict(
            frame,
            confidence_threshold=threshold,
        )

        print(
            f"Total detections: "
            f"{len(detections)}"
        )

        person_count = 0

        for detection in detections:

            if detection.class_name in {
                "Person",
                "person",
            }:
                person_count += 1

            print(
                detection.to_dict()
            )

        print(
            f"Person detections: "
            f"{person_count}"
        )

    # ------------------------------------------------------------------
    # Save original frame
    # ------------------------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "data"
        / "debug_frame_67.jpg"
    )

    cv2.imwrite(
        str(output_path),
        frame,
    )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print(
        f"Saved frame: {output_path}"
    )


if __name__ == "__main__":
    main()