from pathlib import Path
import sys
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model_service import get_model_service
from app.tracker import PersonTracker


VIDEO_PATH = PROJECT_ROOT / "data" / "multi_person_test.mp4"


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def main():

    print("=" * 70)
    print("PPE / PERSON ASSOCIATION DEBUG")
    print("=" * 70)

    model = get_model_service()

    print(f"Model: {model.model_path}")

    tracker = PersonTracker(
        model=model,
        confidence_threshold=0.10,
    )

    capture = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open:\n{VIDEO_PATH}"
        )

    # Inspect representative frames.
    frames_to_check = {1, 25, 50, 75, 100}

    frame_number = 0

    while True:

        success, frame = capture.read()

        if not success:
            break

        frame_number += 1

        if frame_number not in frames_to_check:
            continue

        result = tracker.update(frame)

        print()
        print("=" * 70)
        print(f"FRAME {frame_number}")
        print("=" * 70)

        print(
            f"Total detections: "
            f"{len(result.detections)}"
        )

        print(
            f"Tracked people: "
            f"{len(result.tracked_people)}"
        )

        for person in result.tracked_people:

            print()
            print(
                f"PERSON ID={person.track_id}"
            )

            print(
                f"  confidence = "
                f"{person.confidence:.4f}"
            )

            print(
                f"  box = "
                f"{[round(v, 2) for v in person.box_xyxy]}"
            )

            print()
            print("  Detections overlapping person:")

            matches = []

            for detection in result.detections:

                if detection.class_name.lower() == "person":
                    continue

                overlap = iou(
                    person.box_xyxy,
                    detection.box_xyxy,
                )

                if overlap > 0:

                    matches.append(
                        (
                            overlap,
                            detection,
                        )
                    )

            matches.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            if not matches:

                print(
                    "    NONE"
                )

            for overlap, detection in matches:

                print(
                    f"    {detection.class_name:<18} "
                    f"conf={detection.confidence:.4f} "
                    f"IoU={overlap:.3f} "
                    f"box="
                    f"{[round(v, 2) for v in detection.box_xyxy]}"
                )

        print()
        print("ALL DETECTIONS:")

        for detection in result.detections:

            print(
                f"  {detection.class_name:<18} "
                f"conf={detection.confidence:.4f} "
                f"box="
                f"{[round(v, 2) for v in detection.box_xyxy]}"
            )

    capture.release()

    tracker.reset()

    print()
    print("=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()