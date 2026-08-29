from pathlib import Path
import sys
import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model_service import get_model_service


TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "ppe_tests"
)


TEST_FILES = [
    "hardhat_test.jpg",
    "vest_test.jpg",
    "mask_test.jpg",
    "violation_test.jpg",
]


def main():

    print("=" * 70)
    print("PPE DETECTOR CONFIDENCE DIAGNOSTIC")
    print("=" * 70)

    model_service = get_model_service()
    yolo = model_service._model

    print()
    print("Model:", model_service.model_path)
    print("Classes:", model_service.class_names)

    thresholds = [
        0.01,
        0.03,
        0.05,
        0.10,
        0.20,
    ]

    for filename in TEST_FILES:

        image_path = TEST_DIR / filename

        print()
        print("=" * 70)
        print(f"IMAGE: {filename}")
        print("=" * 70)

        if not image_path.exists():
            print("FILE NOT FOUND:", image_path)
            continue

        frame = cv2.imread(str(image_path))

        if frame is None:
            print("COULD NOT READ IMAGE")
            continue

        for threshold in thresholds:

            results = yolo.predict(
                source=frame,
                conf=threshold,
                verbose=False,
            )

            if not results or results[0].boxes is None:
                print()
                print(
                    f"CONF {threshold:.2f}: "
                    "NO DETECTIONS"
                )
                continue

            boxes = results[0].boxes

            print()
            print(
                f"CONF {threshold:.2f}: "
                f"{len(boxes)} detections"
            )

            for i in range(len(boxes)):

                class_id = int(
                    boxes.cls[i].item()
                )

                confidence = float(
                    boxes.conf[i].item()
                )

                name = model_service.class_names.get(
                    class_id,
                    str(class_id),
                )

                box = [
                    round(float(v), 1)
                    for v in boxes.xyxy[i].tolist()
                ]

                print(
                    f"  {name:18}"
                    f" conf={confidence:.4f}"
                    f" box={box}"
                )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()