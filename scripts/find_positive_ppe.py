from pathlib import Path
import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALID_DIR = Path(
    r"C:\Users\Dell\.cache\kagglehub\datasets\shlokraval\ppe-dataset-yolov8\versions\1\valid\images"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "trained"
    / "best.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "ppe_tests"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGETS = {
    "hardhat": {"Person", "Hardhat"},
    "gloves": {"Person", "Gloves"},
    "goggles": {"Person", "Goggles"},
    "mask": {"Person", "Mask"},
    "vest": {"Person", "Safety Vest"},
    "violation": {
        "Person",
        "NO-Hardhat",
    },
}


def main():

    print("=" * 70)
    print("SEARCHING FOR PPE TEST IMAGES")
    print("=" * 70)

    print("Model :", MODEL_PATH)
    print("Images:", VALID_DIR)

    model = YOLO(str(MODEL_PATH))

    image_paths = [
        p
        for p in VALID_DIR.iterdir()
        if p.suffix.lower()
        in {".jpg", ".jpeg", ".png"}
    ]

    print(
        f"Validation images: {len(image_paths)}"
    )

    found = {
        name: False
        for name in TARGETS
    }

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        if all(found.values()):
            break

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:
            continue

        results = model.predict(
            source=frame,
            conf=0.10,
            verbose=False,
        )

        if not results:
            continue

        boxes = results[0].boxes

        if boxes is None:
            continue

        detected = set()

        for i in range(len(boxes)):

            class_id = int(
                boxes.cls[i].item()
            )

            class_name = model.names[
                class_id
            ]

            detected.add(class_name)

        for target_name, required in TARGETS.items():

            if found[target_name]:
                continue

            if required.issubset(detected):

                output_path = (
                    OUTPUT_DIR
                    / f"{target_name}_test.jpg"
                )

                cv2.imwrite(
                    str(output_path),
                    frame,
                )

                found[target_name] = True

                print()
                print(
                    f"[FOUND] {target_name}"
                )
                print(
                    "Image   :",
                    image_path,
                )
                print(
                    "Detected:",
                    sorted(detected),
                )
                print(
                    "Saved   :",
                    output_path,
                )

        if index % 500 == 0:

            print(
                f"Checked {index}/{len(image_paths)}..."
            )

    print()
    print("=" * 70)
    print("SEARCH COMPLETE")
    print("=" * 70)

    for name, status in found.items():

        print(
            f"{name:12} : "
            f"{'FOUND' if status else 'NOT FOUND'}"
        )


if __name__ == "__main__":
    main()