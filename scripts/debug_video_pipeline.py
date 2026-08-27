from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.model_service import get_model_service
from app.tracker import PersonTracker


VIDEO_PATH = PROJECT_ROOT / "data" / "test.mp4"


def main():

    print("=" * 60)
    print("VIDEO PIPELINE TRACKER DEBUG")
    print("=" * 60)

    print("Video:", VIDEO_PATH)

    model = get_model_service()

    print("Model:", model.model_path)
    print()

    tracker = PersonTracker(
        model=model,
        confidence_threshold=0.10,
    )

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

    print("Total frames:", total_frames)
    print("Starting...")
    print("-" * 60)

    frame_number = 0

    while True:

        success, frame = capture.read()

        if not success:
            break

        frame_number += 1

        result = tracker.update(frame)

        people = result.tracked_people
        detections = result.detections

        # Print selected frames so the terminal doesn't get flooded.
        if frame_number in {
            1,
            25,
            50,
            67,
            75,
            100,
            150,
            200,
            250,
            300,
        }:

            print()
            print(
                f"FRAME {frame_number}"
            )

            print(
                "Total detections:",
                len(detections),
            )

            print(
                "Tracked people:",
                len(people),
            )

            for detection in detections:

                print(
                    "  DETECTION:",
                    detection.to_dict(),
                )

            for person in people:

                print(
                    "  TRACKED PERSON:",
                    person.to_dict(),
                )

        # Stop as soon as a tracked person is found.
        if people:

            print()
            print("=" * 60)
            print("PERSON FOUND")
            print("=" * 60)

            print(
                "Frame:",
                frame_number,
            )

            print(
                "Tracked people:",
                [
                    person.to_dict()
                    for person in people
                ],
            )

            print(
                "All detections:",
                [
                    detection.to_dict()
                    for detection in detections
                ],
            )

            break

    capture.release()

    tracker.reset()

    print()
    print("=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()