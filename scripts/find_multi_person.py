from pathlib import Path
import sys
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model_service import get_model_service


DATASET_ROOT = Path(
    r"C:\Users\Dell\.cache\kagglehub\datasets"
    r"\shlokraval\ppe-dataset-yolov8\versions\1"
)

IMAGE_ROOT = DATASET_ROOT / "valid" / "images"


def main():

    print("=" * 70)
    print("FIND MULTI-PERSON VALIDATION IMAGE")
    print("=" * 70)

    model = get_model_service()

    images = list(IMAGE_ROOT.iterdir())

    print(
        f"Searching {len(images)} validation images..."
    )

    for index, image_path in enumerate(images, 1):

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:
            continue

        detections = model.predict(
            frame,
            confidence_threshold=0.10,
        )

        people = [
            detection
            for detection in detections
            if detection.class_name in {
                "Person",
                "person",
            }
        ]

        if len(people) >= 2:

            print()
            print("=" * 70)
            print("FOUND MULTI-PERSON IMAGE")
            print("=" * 70)

            print(
                f"Image: {image_path}"
            )

            print(
                f"People detected: "
                f"{len(people)}"
            )

            for person in people:
                print(
                    person.to_dict()
                )

            output_path = (
                PROJECT_ROOT
                / "data"
                / "multi_person_test.jpg"
            )

            cv2.imwrite(
                str(output_path),
                frame,
            )

            print()
            print(
                f"Saved test image:"
            )
            print(output_path)

            return

        if index % 500 == 0:
            print(
                f"Checked {index} images..."
            )

    print()
    print("No multi-person image found.")


if __name__ == "__main__":
    main()