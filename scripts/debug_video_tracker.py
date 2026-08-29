"""
Debug the exact tracker integration used by the video pipeline.

Purpose:
    Verify whether PersonTracker.update() returns tracked people
    when called from the same project/environment as video_pipeline.py.
"""

from pathlib import Path
import sys
import cv2


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.model_service import get_model_service
from app.tracker import PersonTracker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "multi_person_test.mp4"
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("=" * 70)
    print("VIDEO PIPELINE TRACKER DEBUG")
    print("=" * 70)

    print(f"Video: {VIDEO_PATH}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    print()
    print("Loading YOLO model...")

    model = get_model_service()

    print(f"Model: {model.model_path}")

    # ------------------------------------------------------------------
    # Create tracker exactly like video_pipeline
    # ------------------------------------------------------------------

    tracker = PersonTracker(
        model=model,
        confidence_threshold=0.10,
    )

    print("Tracker: ByteTrack")

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

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(
        f"Total frames: {total_frames}"
    )

    print()
    print("Testing first 10 frames...")
    print("-" * 70)

    # ------------------------------------------------------------------
    # Process first 10 frames
    # ------------------------------------------------------------------

    for frame_number in range(1, 11):

        success, frame = capture.read()

        if not success:
            print(
                f"Could not read frame {frame_number}"
            )
            break

        tracking_result = tracker.update(
            frame
        )

        tracked_people = (
            tracking_result.tracked_people
        )

        detections = (
            tracking_result.detections
        )

        person_detections = [
            detection
            for detection in detections
            if detection.class_name.lower()
            == "person"
        ]

        print()
        print(
            f"FRAME {frame_number:04d}"
        )

        print(
            f"  Total detections : "
            f"{len(detections)}"
        )

        print(
            f"  Person detections: "
            f"{len(person_detections)}"
        )

        print(
            f"  Tracked people   : "
            f"{len(tracked_people)}"
        )

        if tracked_people:

            for person in tracked_people:

                print(
                    f"    ID={person.track_id} "
                    f"confidence={person.confidence:.4f} "
                    f"box={person.box_xyxy}"
                )

        else:

            print(
                "    NO TRACKED PEOPLE"
            )

        if person_detections:

            for detection in person_detections:

                print(
                    f"    YOLO Person "
                    f"confidence="
                    f"{detection.confidence:.4f} "
                    f"box="
                    f"{detection.box_xyxy}"
                )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    capture.release()

    tracker.reset()

    print()
    print("=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()