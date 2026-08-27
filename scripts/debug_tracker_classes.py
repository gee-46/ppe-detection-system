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
# Configuration
# ---------------------------------------------------------------------------

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "multi_person_test.mp4"
)

TRACKER_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "bytetrack_ppe.yaml"
)


# ---------------------------------------------------------------------------
# Result printer
# ---------------------------------------------------------------------------

def print_result(label, result):

    print()
    print("=" * 70)
    print(label)
    print("=" * 70)

    if not result:
        print("No results")
        return

    boxes = result[0].boxes

    if boxes is None:
        print("No boxes")
        return

    print(
        f"Boxes: {len(boxes)}"
    )

    for i in range(len(boxes)):

        class_id = int(
            boxes.cls[i].item()
        )

        confidence = float(
            boxes.conf[i].item()
        )

        box = [
            round(float(v), 2)
            for v in boxes.xyxy[i].tolist()
        ]

        track_id = None

        if boxes.id is not None:
            track_id = int(
                boxes.id[i].item()
            )

        print(
            f"class={class_id} "
            f"conf={confidence:.4f} "
            f"track_id={track_id} "
            f"box={box}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("=" * 70)
    print("TRACKER CLASS FILTER DEBUG")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Check files
    # -----------------------------------------------------------------------

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    if not TRACKER_CONFIG.exists():
        raise FileNotFoundError(
            f"Tracker configuration not found:\n"
            f"{TRACKER_CONFIG}"
        )

    print(
        f"Video   : {VIDEO_PATH}"
    )

    print(
        f"Tracker : {TRACKER_CONFIG}"
    )

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------

    model = get_model_service()

    print(
        f"Model   : {model.model_path}"
    )

    # -----------------------------------------------------------------------
    # Read first frame
    # -----------------------------------------------------------------------

    capture = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not capture.isOpened():
        raise RuntimeError(
            "Could not open test video."
        )

    success, frame = capture.read()

    capture.release()

    if not success:
        raise RuntimeError(
            "Could not read first frame."
        )

    # -----------------------------------------------------------------------
    # Underlying Ultralytics model
    # -----------------------------------------------------------------------

    yolo = model._model

    # -----------------------------------------------------------------------
    # 1. Normal YOLO detection
    # -----------------------------------------------------------------------

    result = yolo.predict(
        source=frame,
        conf=0.01,
        verbose=False,
    )

    print_result(
        "NORMAL YOLO DETECTION",
        result,
    )

    # -----------------------------------------------------------------------
    # 2. Person-only YOLO detection
    # -----------------------------------------------------------------------

    result = yolo.predict(
        source=frame,
        conf=0.01,
        classes=[11],
        verbose=False,
    )

    print_result(
        "PERSON-ONLY YOLO DETECTION",
        result,
    )

    # -----------------------------------------------------------------------
    # 3. Person-only ByteTrack
    # -----------------------------------------------------------------------

    yolo.predictor = None

    result = yolo.track(
        source=frame,
        persist=True,
        tracker=str(TRACKER_CONFIG),
        conf=0.01,
        iou=0.7,
        classes=[11],
        verbose=False,
    )

    print_result(
        "PERSON-ONLY BYTETRACK",
        result,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()