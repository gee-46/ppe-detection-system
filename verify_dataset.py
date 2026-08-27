from pathlib import Path
from collections import Counter

DATASET = Path(
    r"C:\Users\Dell\.cache\kagglehub\datasets"
    r"\shlokraval\ppe-dataset-yolov8\versions\1"
)

CLASS_NAMES = [
    "Fall-Detected",
    "Gloves",
    "Goggles",
    "Hardhat",
    "Ladder",
    "Mask",
    "NO-Gloves",
    "NO-Goggles",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ["train", "valid", "test"]:
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))

    class_counts = Counter()
    malformed = 0
    invalid_classes = 0
    invalid_coordinates = 0
    empty_labels = 0

    for label_file in labels:
        lines = label_file.read_text(encoding="utf-8").splitlines()

        if not lines:
            empty_labels += 1
            continue

        for line in lines:
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                malformed += 1
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                malformed += 1
                continue

            if not 0 <= class_id < len(CLASS_NAMES):
                invalid_classes += 1
            else:
                class_counts[class_id] += 1

            if not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                invalid_coordinates += 1

    print()
    print("=" * 55)
    print(f"{split.upper()} DATASET")
    print("=" * 55)

    print(f"Images:              {len(images)}")
    print(f"Labels:              {len(labels)}")
    print(f"Empty labels:        {empty_labels}")
    print(f"Malformed rows:      {malformed}")
    print(f"Invalid class IDs:   {invalid_classes}")
    print(f"Invalid coordinates: {invalid_coordinates}")

    print("\nClass distribution:")

    for class_id, name in enumerate(CLASS_NAMES):
        print(f"{class_id:2}  {name:<18} {class_counts[class_id]}")

print()
print("Dataset verification complete.")