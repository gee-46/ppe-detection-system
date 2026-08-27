"""
End-to-end PPE video processing pipeline.

Pipeline:

    Video
      ↓
    YOLO + ByteTrack
      ↓
    Person tracking + YOLO detections
      ↓
    Track-aware PPE compliance
      ↓
    Annotated output video

The tracker performs the YOLO inference and returns both:
    - tracked people
    - all detections

This avoids running a second model.predict() call for the same frame.

This module is intentionally separate from FastAPI so the same
processing engine can later be reused for:

    - webcam
    - RTSP/CCTV
    - multiple cameras
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow:
#
#     python app/video_pipeline.py
#
# in addition to:
#
#     python -m app.video_pipeline
#
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.model_service import Detection, get_model_service
from app.tracker import PersonTracker
from app.tracked_compliance import (
    TrackCompliance,
    TrackComplianceManager,
)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "person_test.mp4"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "ppe_result.mp4"
)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_person(
    frame,
    person,
    compliance: TrackCompliance | None,
):
    """
    Draw a tracked person and their PPE compliance status.
    """

    x1, y1, x2, y2 = [
        int(value)
        for value in person.box_xyxy
    ]

    # ------------------------------------------------------------------
    # Determine status
    # ------------------------------------------------------------------

    if compliance is None:
        status = "UNKNOWN"
    elif compliance.compliant:
        status = "COMPLIANT"
    else:
        status = "NON-COMPLIANT"

    # ------------------------------------------------------------------
    # Person bounding box
    # ------------------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2,
    )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    label = (
        f"ID {person.track_id} | "
        f"{status}"
    )

    cv2.rectangle(
        frame,
        (x1, max(0, y1 - 30)),
        (x1 + 250, y1),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x1 + 5, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    # ------------------------------------------------------------------
    # PPE information
    # ------------------------------------------------------------------

    if compliance is None:
        return

    y = y2 + 20

    if compliance.present_ppe:

        text = (
            "PPE: "
            + ", ".join(
                compliance.present_ppe
            )
        )

        cv2.putText(
            frame,
            text,
            (x1, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        y += 18

    if compliance.missing_ppe:

        text = (
            "Missing: "
            + ", ".join(
                compliance.missing_ppe
            )
        )

        cv2.putText(
            frame,
            text,
            (x1, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )


def _draw_ppe_detection(
    frame,
    detection: Detection,
):
    """
    Draw a PPE detection that is not a Person.
    """

    if not detection.box_xyxy:
        return

    x1, y1, x2, y2 = [
        int(value)
        for value in detection.box_xyxy
    ]

    label = (
        f"{detection.class_name} "
        f"{detection.confidence:.2f}"
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        1,
    )

    cv2.putText(
        frame,
        label,
        (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_video(
    input_path: str | Path,
    output_path: str | Path,
    confidence_threshold: float = 0.10,
    imgsz: int = 640,
) -> dict:
    """
    Process a complete video through the PPE pipeline.

    The pipeline performs one YOLO + ByteTrack call per frame.

    Returns processing statistics.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input video not found:\n{input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("PPE VIDEO PROCESSING PIPELINE")
    print("=" * 70)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print()

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    print("Loading YOLO model...")

    model = get_model_service()

    print(
        f"Model : {model.model_path}"
    )

    # ------------------------------------------------------------------
    # Create tracker
    # ------------------------------------------------------------------

    tracker = PersonTracker(
        model=model,
        confidence_threshold=confidence_threshold,
    )

    print("Tracker: ByteTrack")

    # ------------------------------------------------------------------
    # Create compliance manager
    # ------------------------------------------------------------------

    compliance_manager = (
        TrackComplianceManager()
    )

    # ------------------------------------------------------------------
    # Open input video
    # ------------------------------------------------------------------

    capture = cv2.VideoCapture(
        str(input_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open input video:\n"
            f"{input_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print()
    print(
        f"Resolution    : "
        f"{width}x{height}"
    )
    print(
        f"FPS           : "
        f"{fps:.2f}"
    )
    print(
        f"Total frames  : "
        f"{total_frames}"
    )
    print()
    print("Processing...")
    print("-" * 70)

    # ------------------------------------------------------------------
    # Output writer
    # ------------------------------------------------------------------

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():

        capture.release()

        raise RuntimeError(
            f"Could not create output video:\n"
            f"{output_path}"
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    frame_count = 0

    frames_with_people = 0

    unique_track_ids = set()

    compliant_frames = 0

    non_compliant_frames = 0

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------

    while True:

        success, frame = capture.read()

        if not success:
            break

        frame_count += 1

        # --------------------------------------------------------------
        # ONE YOLO + ByteTrack inference
        # --------------------------------------------------------------

        tracking_result = tracker.update(
            frame
        )

        tracked_people = (
            tracking_result.tracked_people
        )

        detections = (
            tracking_result.detections
        )

        # --------------------------------------------------------------
        # Track statistics
        # --------------------------------------------------------------

        if tracked_people:
            frames_with_people += 1

        current_ids = {
            person.track_id
            for person in tracked_people
        }

        unique_track_ids.update(
            current_ids
        )

        # --------------------------------------------------------------
        # Track-aware PPE compliance
        # --------------------------------------------------------------

        compliance_results = (
            compliance_manager.update(
                tracked_people=tracked_people,
                detections=detections,
            )
        )

        compliance_by_id = {
            result.track_id: result
            for result in compliance_results
        }

        # --------------------------------------------------------------
        # Frame compliance statistics
        # --------------------------------------------------------------

        if any(
            result.compliant
            for result in compliance_results
        ):
            compliant_frames += 1

        if any(
            not result.compliant
            for result in compliance_results
        ):
            non_compliant_frames += 1

        # --------------------------------------------------------------
        # Draw PPE detections
        # --------------------------------------------------------------

        for detection in detections:

            if detection.class_name in {
                "Person",
                "person",
            }:
                continue

            _draw_ppe_detection(
                frame,
                detection,
            )

        # --------------------------------------------------------------
        # Draw tracked people
        # --------------------------------------------------------------

        for person in tracked_people:

            compliance = (
                compliance_by_id.get(
                    person.track_id
                )
            )

            _draw_person(
                frame,
                person,
                compliance,
            )

        # --------------------------------------------------------------
        # Global frame information
        # --------------------------------------------------------------

        info = (
            f"Frame: "
            f"{frame_count}/{total_frames} | "
            f"People: "
            f"{len(tracked_people)}"
        )

        cv2.putText(
            frame,
            info,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # --------------------------------------------------------------
        # Write output frame
        # --------------------------------------------------------------

        writer.write(frame)

        # --------------------------------------------------------------
        # Progress
        # --------------------------------------------------------------

        if (
            frame_count == 1
            or frame_count % 25 == 0
            or frame_count == total_frames
        ):

            print(
                f"Frame "
                f"{frame_count:04d}/"
                f"{total_frames} | "
                f"People: "
                f"{len(tracked_people)} | "
                f"IDs: "
                f"{sorted(current_ids)}"
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    capture.release()
    writer.release()

    # Reset ByteTrack state after the video finishes.
    tracker.reset()

    # ------------------------------------------------------------------
    # Final statistics
    # ------------------------------------------------------------------

    statistics = {
        "frames_processed": frame_count,

        "frames_with_people": (
            frames_with_people
        ),

        "unique_track_ids": sorted(
            unique_track_ids
        ),

        "total_unique_people": len(
            unique_track_ids
        ),

        "compliant_frames": (
            compliant_frames
        ),

        "non_compliant_frames": (
            non_compliant_frames
        ),

        "output_path": str(
            output_path.resolve()
        ),
    }

    print()
    print("=" * 70)
    print("VIDEO PROCESSING COMPLETE")
    print("=" * 70)

    for key, value in statistics.items():

        print(
            f"{key:<25}: {value}"
        )

    return statistics


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    process_video(
        input_path=DEFAULT_INPUT,
        output_path=DEFAULT_OUTPUT,
    )