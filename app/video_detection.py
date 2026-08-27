"""
Video / webcam inference.

CONSOLIDATION NOTE: the original repo had three near-identical scripts that
each reimplemented "open a cv2.VideoCapture, loop frames, run YOLO, draw
boxes, imshow":
  - video_detection.py   (video file + FPS overlay, plus an unused
                           `import mediapipe` that was never called)
  - webcam_detection.py  (webcam, all classes)
  - person_detection.py  (webcam, filtered to the COCO "person" class only)

They are merged here into one module with a single shared frame-processing
loop (`_run_capture_loop`). `class_filter` reproduces person_detection.py's
behavior generically (works for any class name(s), not just "person"), and
video file vs. webcam is just a different `source` argument to
cv2.VideoCapture, which is all that differed between the other two.

This file is for live/local demo use (cv2.imshow window with a GUI). The
API's /predict/video endpoint (app/api/main.py) is the non-interactive,
headless equivalent used for server-side video file processing.
"""

from __future__ import annotations

import argparse
import time
from typing import List, Optional

import cv2

from app.model_service import get_model_service
from app.ppe_logic import evaluate_compliance


def _run_capture_loop(source, class_filter: Optional[List[str]] = None, window_name: str = "Detection"):
    """Shared loop used for both video files and webcam. Displays an OpenCV window; press 'q' to quit."""
    service = get_model_service()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            start_time = time.time()
            detections, annotated = service.predict_and_annotate(frame)

            if class_filter:
                # Redraw only filtered classes on a clean copy of the frame,
                # mirroring the old person_detection.py behavior.
                annotated = frame.copy()
                for d in detections:
                    if d.class_name not in class_filter:
                        continue
                    x1, y1, x2, y2 = map(int, d.box_xyxy)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated,
                        f"{d.class_name} {d.confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            fps = 1.0 / max(time.time() - start_time, 1e-6)
            cv2.putText(
                annotated, f"FPS: {fps:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def process_video_file(path: str):
    """Equivalent of the old video_detection.py."""
    _run_capture_loop(path, window_name="Video Detection")


def process_webcam(class_filter: Optional[List[str]] = None):
    """Equivalent of the old webcam_detection.py (class_filter=None) or
    person_detection.py (class_filter=["person"])."""
    _run_capture_loop(0, class_filter=class_filter, window_name="Webcam Detection")


def process_video_file_headless(path: str, sample_rate: int = 5, max_frames: int = None):
    """
    Non-interactive version for server-side use (no GUI window).
    Samples every `sample_rate`-th frame and returns aggregate detection +
    compliance stats. Used by the FastAPI /predict/video endpoint.
    """
    service = get_model_service()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")

    frame_index = 0
    analyzed = 0
    class_counts = {}
    violation_frames = 0
    any_compliant_frame = False

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_index % sample_rate == 0:
                detections = service.predict(frame)
                compliance = evaluate_compliance(detections)

                for d in detections:
                    class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

                if compliance.overall_status == "non_compliant":
                    violation_frames += 1
                elif compliance.overall_status == "compliant":
                    any_compliant_frame = True

                analyzed += 1
                if max_frames and analyzed >= max_frames:
                    break

            frame_index += 1
    finally:
        cap.release()

    if violation_frames > 0:
        overall_status = "non_compliant"
    elif any_compliant_frame:
        overall_status = "compliant"
    else:
        overall_status = "not_configured"

    return {
        "frames_total": frame_index,
        "frames_analyzed": analyzed,
        "detections_by_class": class_counts,
        "violation_frame_count": violation_frames,
        "overall_compliance_status": overall_status,
    }


def main():
    parser = argparse.ArgumentParser(description="Run video/webcam detection (interactive GUI window).")
    parser.add_argument("--source", default="webcam", help="'webcam' or path to a video file")
    parser.add_argument(
        "--filter", nargs="*", default=None, help="Only draw these class names (e.g. --filter person)"
    )
    args = parser.parse_args()

    if args.source == "webcam":
        process_webcam(class_filter=args.filter)
    else:
        process_video_file(args.source)


if __name__ == "__main__":
    main()
