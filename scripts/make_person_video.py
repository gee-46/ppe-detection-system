"""
Create a simple continuous video from a known Person image.

Used only to verify that ByteTrack assigns and maintains
a track ID across consecutive frames.
"""

from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_IMAGE = PROJECT_ROOT / "data" / "person_test.jpg"
OUTPUT_VIDEO = PROJECT_ROOT / "data" / "person_test.mp4"


def main():

    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{INPUT_IMAGE}"
        )

    image = cv2.imread(
        str(INPUT_IMAGE)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image:\n{INPUT_IMAGE}"
        )

    height, width = image.shape[:2]

    OUTPUT_VIDEO.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create video:\n{OUTPUT_VIDEO}"
        )

    # 100 consecutive frames of the same scene.
    for _ in range(100):
        writer.write(image)

    writer.release()

    print("=" * 60)
    print("PERSON TEST VIDEO CREATED")
    print("=" * 60)
    print(f"Input : {INPUT_IMAGE}")
    print(f"Output: {OUTPUT_VIDEO}")
    print("Frames: 100")
    print("FPS   : 10")


if __name__ == "__main__":
    main()