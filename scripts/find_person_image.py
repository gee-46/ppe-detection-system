"""
Find a validation image where the trained PPE model detects Person.
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET_ROOT = Path(
    r"C:\Users\Dell\.cache\kagglehub\datasets"
    r"\shlokraval\ppe-dataset-yolov8\versions\1"
)

IMAGE_DIR = DATASET_ROOT / "valid" / "images"

OUTPUT = PROJECT_ROOT / "data" / "person_test.jpg"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Validation image directory not found:\n{IMAGE_DIR}"
        )

    model = get_model_service()

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    images = [
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in image_extensions
    ]

    print(
        f"Searching {len(images)} validation images..."
    )

    for index, image_path in enumerate(images):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        detections = model.predict(
            image,
            confidence_threshold=0.10,
        )

        people = [
            detection
            for detection in detections
            if detection.class_name == "Person"
        ]

        if people:

            print()
            print("=" * 60)
            print("FOUND PERSON IMAGE")
            print("=" * 60)

            print(
                f"Image: {image_path}"
            )

            print(
                f"People detected: {len(people)}"
            )

            print()

            for person in people:
                print(
                    person.to_dict()
                )

            OUTPUT.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            cv2.imwrite(
                str(OUTPUT),
                image,
            )

            print()
            print(
                f"Saved test image:"
            )
            print(
                OUTPUT.resolve()
            )

            return

        if (index + 1) % 25 == 0:

            print(
                f"Checked {index + 1} images..."
            )

    print()
    print("=" * 60)
    print("NO PERSON DETECTION FOUND")
    print("=" * 60)


if __name__ == "__main__":
    main()