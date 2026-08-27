from pathlib import Path
import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "multi_person_test.jpg"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "multi_person_test.mp4"
)

FRAME_COUNT = 100
FPS = 10


def main():

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found:\n{IMAGE_PATH}"
        )

    frame = cv2.imread(
        str(IMAGE_PATH)
    )

    if frame is None:
        raise RuntimeError(
            "Could not read test image."
        )

    height, width = frame.shape[:2]

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not create output video."
        )

    for _ in range(FRAME_COUNT):
        writer.write(frame)

    writer.release()

    print("=" * 60)
    print("MULTI-PERSON TEST VIDEO CREATED")
    print("=" * 60)
    print(f"Input : {IMAGE_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Frames: {FRAME_COUNT}")
    print(f"FPS   : {FPS}")


if __name__ == "__main__":
    main()