from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model_service import get_model_service


VIDEO_PATH = PROJECT_ROOT / "data" / "test.mp4"


def main():

    model = get_model_service()

    capture = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open:\n{VIDEO_PATH}"
        )

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print("=" * 60)
    print("RAW YOLO VIDEO DETECTION DEBUG")
    print("=" * 60)

    print(f"Video: {VIDEO_PATH}")
    print(f"Frames: {total_frames}")
    print()

    for frame_number in range(1, total_frames + 1):

        success, frame = capture.read()

        if not success:
            break

        detections = model.predict(
            frame,
            confidence_threshold=0.10,
        )

        persons = [
            d
            for d in detections
            if d.class_name == "Person"
        ]

        if persons:

            print(
                f"Frame {frame_number}: "
                f"{len(persons)} PERSON(S)"
            )

            for person in persons:

                print(
                    f"    confidence="
                    f"{person.confidence:.3f} "
                    f"box={person.box_xyxy}"
                )

            # Stop once we prove Person exists.
            break

        if frame_number % 25 == 0:

            print(
                f"Checked frame {frame_number}: "
                "no Person"
            )

    capture.release()

    print()
    print("=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()