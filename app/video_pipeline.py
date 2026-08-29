"""
End-to-end PPE video processing pipeline.

Architecture:

    Video
      |
      +------------------------+
      |                        |
      v                        v
  YOLOv8n COCO          Custom PPE YOLO
      |                 on person crops
      v                        |
  ByteTrack                    |
      |                        |
      v                        v
 Person IDs              PPE detections
      |                        |
      +-----------+------------+
                  |
                  v
        Track-aware compliance
                  |
                  v
        Clean annotated video

The PersonTracker is responsible for:

    YOLOv8n person detection
    ByteTrack tracking
    person-crop PPE detection
    duplicate PPE suppression

This pipeline is responsible for:

    compliance
    statistics
    clean visualization
    output video
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import cv2
from ultralytics import YOLO


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.model_service import get_model_service
from app.tracker import PersonTracker
from app.tracked_compliance import TrackComplianceManager


# ============================================================================
# DEFAULT PATHS
# ============================================================================

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "multi_person_test.mp4"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "ppe_result.mp4"
)

PERSON_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolov8n.pt"
)


# ============================================================================
# VISUALIZATION
# ============================================================================

def _draw_text_box(
    frame,
    text,
    x,
    y,
    width,
    height=28,
):
    """
    Draw a black background behind text and then draw white text.

    Coordinates are clipped to the video frame.
    """

    frame_height, frame_width = frame.shape[:2]

    x = max(0, min(int(x), frame_width - 1))
    y = max(height, min(int(y), frame_height))

    width = min(
        width,
        frame_width - x,
    )

    top = max(
        0,
        y - height,
    )

    cv2.rectangle(
        frame,
        (x, top),
        (x + width, y),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x + 6, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_person(
    frame,
    person,
    compliance,
):
    """
    Draw one tracked person.

    The annotation is intentionally compact so multiple people
    do not overwrite each other's information.
    """

    x1, y1, x2, y2 = [
        int(v)
        for v in person.box_xyxy
    ]

    frame_height, frame_width = frame.shape[:2]

    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width - 1))
    y2 = max(0, min(y2, frame_height - 1))

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    if compliance is None:
        status = "UNKNOWN"
    elif compliance.compliant:
        status = "COMPLIANT"
    else:
        status = "NON-COMPLIANT"

    # ------------------------------------------------------------------
    # Person box
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

    header = (
        f"ID {person.track_id} | {status}"
    )

    # Keep header inside the frame.
    header_width = 235

    header_x = x1

    if header_x + header_width > frame_width:
        header_x = max(
            0,
            frame_width - header_width,
        )

    header_y = max(
        32,
        y1,
    )

    _draw_text_box(
        frame,
        header,
        header_x,
        header_y,
        header_width,
        30,
    )

    if compliance is None:
        return

    # ------------------------------------------------------------------
    # Compact PPE summary
    # ------------------------------------------------------------------

    present = list(
        compliance.present_ppe
    )

    missing = list(
        compliance.missing_ppe
    )

    if present:
        ppe_text = (
            "PPE: "
            + ", ".join(present)
        )
    else:
        ppe_text = "PPE: None"

    # Only show missing PPE if there is room.
    if missing:
        missing_text = (
            "Missing: "
            + ", ".join(missing)
        )
    else:
        missing_text = "Missing: None"

    # ------------------------------------------------------------------
    # Draw summary ABOVE or BELOW person depending on available space.
    # ------------------------------------------------------------------

    line_height = 20

    required_height = (
        line_height * 2
        + 8
    )

    if (
        y2 + required_height
        < frame_height
    ):
        text_x = x1

        if text_x + 300 > frame_width:
            text_x = max(
                0,
                frame_width - 300,
            )

        text_y = y2 + 20

        cv2.putText(
            frame,
            ppe_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            missing_text,
            (text_x, text_y + line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    else:
        # If the person reaches the bottom of the image,
        # put the summary near the top of their box.
        text_x = x1

        if text_x + 300 > frame_width:
            text_x = max(
                0,
                frame_width - 300,
            )

        text_y = max(
            55,
            y1 + 55,
        )

        cv2.putText(
            frame,
            ppe_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            missing_text,
            (text_x, text_y + line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _draw_ppe_detection(
    frame,
    detection,
):
    """
    Draw one PPE detection.

    Person detections and irrelevant classes have already been
    filtered by PersonTracker.
    """

    if not detection.box_xyxy:
        return

    x1, y1, x2, y2 = [
        int(v)
        for v in detection.box_xyxy
    ]

    frame_height, frame_width = frame.shape[:2]

    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width - 1))
    y2 = max(0, min(y2, frame_height - 1))

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
        (
            x1,
            max(
                15,
                y1 - 5,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_frame_info(
    frame,
    frame_number,
    total_frames,
    people_count,
):
    """
    Draw global frame information.
    """

    text = (
        f"Frame {frame_number}/{total_frames}"
        f" | People: {people_count}"
    )

    cv2.rectangle(
        frame,
        (0, 0),
        (290, 38),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def process_video(
    input_path: str | Path,
    output_path: str | Path,
    confidence_threshold: float = 0.20,
    imgsz: int = 640,
) -> dict:
    """
    Process an entire video.

    Person detection:
        YOLOv8n COCO

    Tracking:
        ByteTrack

    PPE:
        Custom PPE model on individual person crops

    Compliance:
        TrackComplianceManager
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    # ------------------------------------------------------------------
    # Validate
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
    # PPE model
    # ------------------------------------------------------------------

    print("Loading PPE model...")

    ppe_model = get_model_service()

    print(
        f"PPE Model    : "
        f"{ppe_model.model_path}"
    )

    # ------------------------------------------------------------------
    # Person model
    # ------------------------------------------------------------------

    print("Loading Person model...")

    if not PERSON_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Person model not found:\n"
            f"{PERSON_MODEL_PATH}"
        )

    person_model = YOLO(
        str(PERSON_MODEL_PATH)
    )

    print(
        f"Person Model : "
        f"{PERSON_MODEL_PATH}"
    )

    # ------------------------------------------------------------------
    # Tracker
    # ------------------------------------------------------------------

    tracker = PersonTracker(
        model=ppe_model,
        person_model=person_model,
        confidence_threshold=confidence_threshold,
        ppe_confidence_threshold=0.01,
        crop_padding=0.15,
    )

    print(
        "Tracker      : "
        "YOLOv8n Person + ByteTrack"
    )

    print(
        "PPE strategy : "
        "Per-person crop inference"
    )

    # ------------------------------------------------------------------
    # Compliance
    # ------------------------------------------------------------------

    compliance_manager = (
        TrackComplianceManager()
    )

    # ------------------------------------------------------------------
    # Open video
    # ------------------------------------------------------------------

    capture = cv2.VideoCapture(
        str(input_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video:\n"
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

    track_stats = defaultdict(
        lambda: {
            "frames_seen": 0,
            "compliant_frames": 0,
            "non_compliant_frames": 0,
            "ppe_observations": defaultdict(int),
            "missing_observations": defaultdict(int),
            "violations": defaultdict(int),
            "confidence_sum": 0.0,
        }
    )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    while True:

        success, frame = capture.read()

        if not success:
            break

        frame_count += 1

        # --------------------------------------------------------------
        # TRACK + PPE
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
        # IDs
        # --------------------------------------------------------------

        current_ids = {
            person.track_id
            for person in tracked_people
        }

        unique_track_ids.update(
            current_ids
        )

        if tracked_people:
            frames_with_people += 1

        # --------------------------------------------------------------
        # COMPLIANCE
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
        # FRAME STATISTICS
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
        # TRACK STATISTICS
        # --------------------------------------------------------------

        for result in compliance_results:

            track_id = result.track_id

            stats = track_stats[
                track_id
            ]

            stats["frames_seen"] += 1

            stats["confidence_sum"] += (
                result.person_confidence
            )

            if result.compliant:
                stats[
                    "compliant_frames"
                ] += 1
            else:
                stats[
                    "non_compliant_frames"
                ] += 1

            for ppe in result.present_ppe:
                stats[
                    "ppe_observations"
                ][ppe] += 1

            for ppe in result.missing_ppe:
                stats[
                    "missing_observations"
                ][ppe] += 1

            for violation in result.violations:
                stats[
                    "violations"
                ][violation] += 1

        # --------------------------------------------------------------
        # DRAW PPE
        # --------------------------------------------------------------

        for detection in detections:

            _draw_ppe_detection(
                frame,
                detection,
            )

        # --------------------------------------------------------------
        # DRAW PEOPLE
        #
        # Draw larger person boxes AFTER PPE boxes.
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
        # GLOBAL INFO
        # --------------------------------------------------------------

        _draw_frame_info(
            frame,
            frame_count,
            total_frames,
            len(tracked_people),
        )

        # --------------------------------------------------------------
        # WRITE FRAME
        # --------------------------------------------------------------

        writer.write(frame)

        # --------------------------------------------------------------
        # PROGRESS
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
    # CLEANUP
    # ------------------------------------------------------------------

    capture.release()
    writer.release()

    tracker.reset()
    compliance_manager.clear()

    # ------------------------------------------------------------------
    # BUILD TRACK SUMMARIES
    # ------------------------------------------------------------------

    track_summaries = {}

    for track_id in sorted(
        track_stats.keys()
    ):

        stats = track_stats[
            track_id
        ]

        frames_seen = (
            stats["frames_seen"]
        )

        compliance_rate = (
            (
                stats["compliant_frames"]
                / frames_seen
                * 100.0
            )
            if frames_seen
            else 0.0
        )

        average_confidence = (
            (
                stats["confidence_sum"]
                / frames_seen
            )
            if frames_seen
            else 0.0
        )

        track_summaries[
            str(track_id)
        ] = {
            "frames_seen": frames_seen,

            "compliant_frames": (
                stats[
                    "compliant_frames"
                ]
            ),

            "non_compliant_frames": (
                stats[
                    "non_compliant_frames"
                ]
            ),

            "compliance_rate": round(
                compliance_rate,
                2,
            ),

            "average_person_confidence": round(
                average_confidence,
                4,
            ),

            "ppe_observations": dict(
                sorted(
                    stats[
                        "ppe_observations"
                    ].items()
                )
            ),

            "missing_observations": dict(
                sorted(
                    stats[
                        "missing_observations"
                    ].items()
                )
            ),

            "violations": dict(
                sorted(
                    stats[
                        "violations"
                    ].items()
                )
            ),
        }

    # ------------------------------------------------------------------
    # FINAL STATISTICS
    # ------------------------------------------------------------------

    statistics = {
        "frames_processed": frame_count,
        "frames_with_people": frames_with_people,
        "unique_track_ids": sorted(
            unique_track_ids
        ),
        "total_unique_people": len(
            unique_track_ids
        ),
        "compliant_frames": compliant_frames,
        "non_compliant_frames": non_compliant_frames,
        "track_summaries": track_summaries,
        "output_path": str(
            output_path.resolve()
        ),
    }

    # ------------------------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("VIDEO PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"frames_processed"
        f"{'':<9}: "
        f"{frame_count}"
    )

    print(
        f"frames_with_people"
        f"{'':<7}: "
        f"{frames_with_people}"
    )

    print(
        f"unique_track_ids"
        f"{'':<10}: "
        f"{sorted(unique_track_ids)}"
    )

    print(
        f"total_unique_people"
        f"{'':<5}: "
        f"{len(unique_track_ids)}"
    )

    print(
        f"compliant_frames"
        f"{'':<9}: "
        f"{compliant_frames}"
    )

    print(
        f"non_compliant_frames"
        f"{'':<5}: "
        f"{non_compliant_frames}"
    )

    print(
        f"output_path"
        f"{'':<17}: "
        f"{output_path.resolve()}"
    )

    print()
    print("PER-TRACK COMPLIANCE")
    print("-" * 70)

    for track_id, summary in (
        track_summaries.items()
    ):

        print(
            f"Track {track_id}: "
            f"frames={summary['frames_seen']} | "
            f"compliant="
            f"{summary['compliant_frames']} | "
            f"non_compliant="
            f"{summary['non_compliant_frames']} | "
            f"rate="
            f"{summary['compliance_rate']:.2f}%"
        )

        print(
            f"  PPE: "
            f"{summary['ppe_observations']}"
        )

        print(
            f"  Missing: "
            f"{summary['missing_observations']}"
        )

        if summary["violations"]:

            print(
                f"  Violations: "
                f"{summary['violations']}"
            )

    print()

    return statistics


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":

    process_video(
        input_path=DEFAULT_INPUT,
        output_path=DEFAULT_OUTPUT,
    )