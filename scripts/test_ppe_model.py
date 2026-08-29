"""
Test the custom PPE model independently.

This verifies whether best.pt can detect the PPE classes
before person association and tracking are involved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.model_service import get_model_service


# ---------------------------------------------------------------------------
# Test images
# ---------------------------------------------------------------------------

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "ppe_tests"
)

TEST_IMAGES = [
    "hardhat_test.jpg",
    "vest_test.jpg",
    "mask_test.jpg",
    "violation_test.jpg",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("=" * 70)
    print("CUSTOM PPE MODEL TEST")
    print("=" * 70)

    model = get_model_service()

    print()
    print("Model:")
    print(model.model_path)

    print()

    for image_name in TEST_IMAGES:

        image_path = (
            TEST_DIR
            / image_name
        )

        print("=" * 70)
        print(
            f"IMAGE: {image_name}"
        )
        print("=" * 70)

        if not image_path.exists():

            print(
                "NOT FOUND:",
                image_path,
            )

            continue

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:

            print(
                "Could not read image"
            )

            continue

        # ---------------------------------------------------------------
        # Run custom PPE model directly.
        #
        # Very low confidence is intentional.
        # We want to see what the model is capable of detecting.
        # ---------------------------------------------------------------

        results = model._model.predict(
            source=frame,
            conf=0.01,
            iou=0.7,
            verbose=False,
        )

        if not results:

            print(
                "No YOLO result"
            )

            continue

        result = results[0]

        if result.boxes is None:

            print(
                "No detections"
            )

            continue

        boxes = result.boxes

        print()
        print(
            f"Detections: {len(boxes)}"
        )

        print()

        for index in range(
            len(boxes)
        ):

            class_id = int(
                boxes.cls[index].item()
            )

            confidence = float(
                boxes.conf[index].item()
            )

            box = [
                round(float(v), 1)
                for v in boxes.xyxy[
                    index
                ].tolist()
            ]

            class_names = model.class_names

            if isinstance(
                class_names,
                dict,
            ):

                class_name = (
                    class_names.get(
                        class_id,
                        str(class_id),
                    )
                )

            else:

                class_name = (
                    class_names[class_id]
                )

            print(
                f"{class_id:>3} "
                f"{class_name:<20} "
                f"conf={confidence:.4f} "
                f"box={box}"
            )

    print()
    print("=" * 70)
    print("PPE MODEL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()