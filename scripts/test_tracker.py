"""
Manual multi-person tracking test.

Pipeline:

    Video
      ↓
    YOLO detection + ByteTrack
      ↓
    Persistent Person IDs
      ↓
    TrackingResult
"""

from pathlib import Path
import sys

import cv2


# ---------------------------------------------------------------------------
# Add project root to Python path
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
    """Run the multi-person tracking test."""

    print("=" * 60)
    print("MULTI-PERSON TRACKING TEST")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Check video
    # ------------------------------------------------------------------

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    print(f"Video : {VIDEO_PATH}")

    # ------------------------------------------------------------------
    # 2. Load trained model
    # ------------------------------------------------------------------

    print("Loading YOLO model...")

    model = get_model_service()

    print(
        f"Model : {model.model_path}"
    )

    # ------------------------------------------------------------------
    # 3. Create ByteTrack tracker
    # ------------------------------------------------------------------

    tracker = PersonTracker(
        model=model,
        confidence_threshold=0.10,
    )

    print("Tracker: ByteTrack")
    print()

    # ------------------------------------------------------------------
    # 4. Open video
    # ------------------------------------------------------------------

    capture = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video:\n"
            f"{VIDEO_PATH}"
        )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    print(
        f"Total frames : {total_frames}"
    )

    print(
        f"FPS          : {fps:.2f}"
    )

    print()
    print("Starting tracking...")
    print("-" * 60)

    # ------------------------------------------------------------------
    # 5. Process frames
    # ------------------------------------------------------------------

    frame_number = 0

    seen_track_ids = set()

    frames_with_people = 0

    while True:

        success, frame = capture.read()

        if not success:
            break

        frame_number += 1

        # --------------------------------------------------------------
        # Run YOLO + ByteTrack
        # --------------------------------------------------------------

        tracking_result = (
            tracker.update(frame)
        )

        # --------------------------------------------------------------
        # Extract tracked people
        # --------------------------------------------------------------

        people = (
            tracking_result.tracked_people
        )

        # --------------------------------------------------------------
        # Track IDs currently visible
        # --------------------------------------------------------------

        current_ids = {
            person.track_id
            for person in people
        }

        seen_track_ids.update(
            current_ids
        )

        if people:
            frames_with_people += 1

        # --------------------------------------------------------------
        # Print frame information
        # --------------------------------------------------------------

        print(
            f"Frame {frame_number:04d}: "
            f"{len(people)} people | "
            f"IDs: {sorted(current_ids)}"
        )

        # --------------------------------------------------------------
        # Print individual people
        # --------------------------------------------------------------

        for person in people:

            box_string = ", ".join(
                f"{value:.1f}"
                for value in person.box_xyxy
            )

            print(
                f"    ID={person.track_id} "
                f"confidence="
                f"{person.confidence:.3f} "
                f"box=[{box_string}]"
            )

    # ------------------------------------------------------------------
    # 6. Release video
    # ------------------------------------------------------------------

    capture.release()

    # ------------------------------------------------------------------
    # 7. Reset tracker
    # ------------------------------------------------------------------

    tracker.reset()

    # ------------------------------------------------------------------
    # 8. Final summary
    # ------------------------------------------------------------------

    print()
    print("=" * 60)
    print("TRACKING TEST COMPLETE")
    print("=" * 60)

    print(
        f"Frames processed       : "
        f"{frame_number}"
    )

    print(
        f"Frames with people     : "
        f"{frames_with_people}"
    )

    print(
        f"Unique track IDs       : "
        f"{sorted(seen_track_ids)}"
    )

    print(
        f"Total unique people    : "
        f"{len(seen_track_ids)}"
    )


if __name__ == "__main__":
    main()