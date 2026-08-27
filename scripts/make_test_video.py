"""
Create a simple local test video from PPE validation images.

This is only for testing the video/tracking pipeline.
It does not replace a real surveillance video.
"""

from pathlib import Path

import cv2


DATASET_ROOT = Path(
    r"C:\Users\Dell\.cache\kagglehub\datasets"
    r"\shlokraval\ppe-dataset-yolov8\versions\1"
)

IMAGE_DIR = DATASET_ROOT / "valid" / "images"

OUTPUT_VIDEO = Path("data/test.mp4")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def main():

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Validation image directory not found:\n{IMAGE_DIR}"
        )

    images = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError(
            f"No images found in:\n{IMAGE_DIR}"
        )

    # Use a manageable number of images for the test video.
    images = images[:100]

    first_image = cv2.imread(str(images[0]))

    if first_image is None:
        raise RuntimeError(
            f"Could not read image:\n{images[0]}"
        )

    height, width = first_image.shape[:2]

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        10.0,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create video:\n{OUTPUT_VIDEO}"
        )

    written = 0

    for image_path in images:

        frame = cv2.imread(
            str(image_path)
        )

        if frame is None:
            continue

        # Ensure every frame has the same dimensions.
        if (
            frame.shape[1] != width
            or frame.shape[0] != height
        ):
            frame = cv2.resize(
                frame,
                (width, height),
            )

        # Hold each image for several frames so the tracker
        # has consecutive frames to process.
        for _ in range(3):
            writer.write(frame)
            written += 1

    writer.release()

    print("Test video created successfully.")
    print(f"Output: {OUTPUT_VIDEO.resolve()}")
    print(f"Images used: {len(images)}")
    print(f"Frames written: {written}")


if __name__ == "__main__":
    main()