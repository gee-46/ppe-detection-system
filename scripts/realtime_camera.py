
"""
Real-time PPE monitoring using a webcam.

Pipeline:

    Webcam
      ↓
    YOLO + ByteTrack
      ↓
    Track-aware PPE compliance
      ↓
    Temporal violation detection
      ↓
    Safety events
      ↓
    Live annotated display

Controls:

    Q / ESC → quit
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import cv2


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.model_service import get_model_service
from app.tracker import PersonTracker
from app.tracked_compliance import TrackComplianceManager
from app.temporal_violations import TemporalViolationEngine
from app.events import SafetyEventManager


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.10

VIOLATION_CONFIRM_FRAMES = 3

WINDOW_NAME = "PPE Safety Monitor"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_person(
    frame,
    person,
    compliance,
):
    """
    Draw a tracked person and PPE status.
    """

    x1, y1, x2, y2 = [
        int(value)
        for value in person.box_xyxy
    ]

    # --------------------------------------------------------------
    # Determine display status.
    # --------------------------------------------------------------

    if compliance is None:

        status = "UNKNOWN"

    elif compliance.compliant:

        status = "COMPLIANT"

    else:

        status = "NON-COMPLIANT"

    # --------------------------------------------------------------
    # Draw bounding box.
    # --------------------------------------------------------------

    if status == "COMPLIANT":
        thickness = 2
    else:
        thickness = 3

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        thickness,
    )

    # --------------------------------------------------------------
    # Main label.
    # --------------------------------------------------------------

    label = (
        f"ID {person.track_id} | "
        f"{status}"
    )

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------------
    # Violation text.
    # --------------------------------------------------------------

    if (
        compliance is not None
        and compliance.violations
    ):

        violations = ", ".join(
            compliance.violations
        )

        cv2.putText(
            frame,
            violations,
            (x1, min(y2 + 20, frame.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run real-time PPE monitoring."""

    print("=" * 70)
    print("REAL-TIME PPE SAFETY MONITOR")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    print("Loading YOLO model...")

    model = get_model_service()

    print(
        f"Model : {model.model_path}"
    )

    # ------------------------------------------------------------------
    # Create pipeline components
    # ------------------------------------------------------------------

    tracker = PersonTracker(
        model=model,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    compliance_manager = (
        TrackComplianceManager()
    )

    temporal_engine = (
        TemporalViolationEngine(
            violation_confirm_frames=(
                VIOLATION_CONFIRM_FRAMES
            )
        )
    )

    event_manager = (
        SafetyEventManager()
    )

    # ------------------------------------------------------------------
    # Open webcam
    # ------------------------------------------------------------------

    print()
    print(
        f"Opening camera index {CAMERA_INDEX}..."
    )

    capture = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Could not open camera "
            f"index {CAMERA_INDEX}"
        )

    print("Camera opened successfully.")
    print()
    print("Press Q or ESC to exit.")
    print("-" * 70)

    # ------------------------------------------------------------------
    # Runtime state
    # ------------------------------------------------------------------

    frame_number = 0

    previous_time = time.perf_counter()

    fps = 0.0

    try:

        while True:

            success, frame = (
                capture.read()
            )

            if not success:

                print(
                    "Could not read frame."
                )

                break

            frame_number += 1

            # ----------------------------------------------------------
            # YOLO + ByteTrack
            # ----------------------------------------------------------

            tracking_result = (
                tracker.update(frame)
            )

            people = (
                tracking_result.tracked_people
            )

            detections = (
                tracking_result.detections
            )

            # ----------------------------------------------------------
            # Track-aware PPE compliance
            # ----------------------------------------------------------

            compliance_results = (
                compliance_manager.update(
                    tracked_people=people,
                    detections=detections,
                )
            )

            compliance_by_id = {
                result.track_id: result
                for result in compliance_results
            }

            # ----------------------------------------------------------
            # Temporal violations
            # ----------------------------------------------------------

            active_track_ids = set()

            for compliance in compliance_results:

                track_id = (
                    compliance.track_id
                )

                active_track_ids.add(
                    track_id
                )

                new_events = (
                    temporal_engine.update(
                        frame_number=frame_number,
                        track_id=track_id,
                        violations=(
                            compliance.violations
                        ),
                    )
                )

                # ------------------------------------------------------
                # Convert confirmed violations into safety events.
                # ------------------------------------------------------

                for violation_event in new_events:

                    event = (
                        event_manager.create_from_violation(
                            track_id=(
                                violation_event.track_id
                            ),
                            violation=(
                                violation_event.violation
                            ),
                            start_frame=(
                                violation_event.start_frame
                            ),
                            confirmed_frame=(
                                violation_event.confirmed_frame
                            ),
                            duration_frames=(
                                violation_event.duration_frames
                            ),
                        )
                    )

                    print(
                        f"\n🚨 SAFETY EVENT "
                        f"{event.event_id} | "
                        f"Track {event.track_id} | "
                        f"{event.violation} | "
                        f"{event.severity}"
                    )

            # ----------------------------------------------------------
            # Draw tracked people.
            # ----------------------------------------------------------

            for person in people:

                compliance = (
                    compliance_by_id.get(
                        person.track_id
                    )
                )

                draw_person(
                    frame,
                    person,
                    compliance,
                )

            # ----------------------------------------------------------
            # Display statistics.
            # ----------------------------------------------------------

            current_time = (
                time.perf_counter()
            )

            elapsed = (
                current_time
                - previous_time
            )

            previous_time = current_time

            if elapsed > 0:

                instant_fps = (
                    1.0 / elapsed
                )

                if fps == 0:

                    fps = instant_fps

                else:

                    fps = (
                        0.9 * fps
                        + 0.1 * instant_fps
                    )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"People: {len(people)}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            active_events = (
                event_manager.get_active_events()
            )

            cv2.putText(
                frame,
                f"Active violations: "
                f"{len(active_events)}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # ----------------------------------------------------------
            # Show frame.
            # ----------------------------------------------------------

            cv2.imshow(
                WINDOW_NAME,
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in {
                ord("q"),
                27,
            }:

                break

    finally:

        # --------------------------------------------------------------
        # Cleanup
        # --------------------------------------------------------------

        capture.release()

        cv2.destroyAllWindows()

        tracker.reset()

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    all_events = (
        event_manager.get_all_events()
    )

    print()
    print("=" * 70)
    print("REAL-TIME MONITOR STOPPED")
    print("=" * 70)

    print(
        f"Frames processed : "
        f"{frame_number}"
    )

    print(
        f"Average FPS      : "
        f"{fps:.2f}"
    )

    print(
        f"Safety events    : "
        f"{len(all_events)}"
    )

    print(
        f"Active events    : "
        f"{len(event_manager.get_active_events())}"
    )


if __name__ == "__main__":
    main()