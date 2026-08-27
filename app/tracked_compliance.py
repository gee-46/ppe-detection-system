"""
Track-aware PPE compliance.

Connects:
    PersonTracker
        +
    YOLO PPE detections
        +
    existing ppe_logic

The goal is to maintain PPE compliance separately for each
tracked person.

Example:

    Track ID 5
        -> Hardhat
        -> Gloves
        -> NO-Goggles

    Track ID 7
        -> Hardhat
        -> Gloves
        -> Goggles

Each person receives an independent compliance result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from app.model_service import Detection
from app.tracker import TrackedPerson
from app.ppe_logic import PersonCompliance


@dataclass
class TrackCompliance:
    """
    PPE compliance state associated with one tracked person.
    """

    track_id: int
    person_confidence: float

    compliant: bool

    present_ppe: List[str] = field(
        default_factory=list
    )

    missing_ppe: List[str] = field(
        default_factory=list
    )

    violations: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "person_confidence": round(
                self.person_confidence,
                4,
            ),
            "compliant": self.compliant,
            "present_ppe": self.present_ppe,
            "missing_ppe": self.missing_ppe,
            "violations": self.violations,
        }


class TrackComplianceManager:
    """
    Maintains PPE compliance for multiple tracked people.

    The manager does not implement new PPE rules.

    It reuses the existing PPE decision layer and associates
    detections with individual tracked person bounding boxes.
    """

    def __init__(
        self,
        required_ppe_classes: List[str] | None = None,
    ):
        self.required_ppe_classes = (
            required_ppe_classes
        )

        self.states: Dict[
            int,
            TrackCompliance,
        ] = {}

    def update(
        self,
        tracked_people: List[TrackedPerson],
        detections: List[Detection],
    ) -> List[TrackCompliance]:
        """
        Calculate compliance for every currently tracked person.

        PPE detections are restricted to each person's bounding box
        before passing them through the existing PPE decision logic.
        """

        results: List[TrackCompliance] = []

        for person in tracked_people:

            person_detection = Detection(
                class_id=-1,
                class_name="Person",
                confidence=person.confidence,
                box_xyxy=person.box_xyxy,
            )

            # ----------------------------------------------------------
            # Find detections associated with this person.
            #
            # Keep the Person detection plus PPE detections whose
            # bounding boxes belong to this tracked person.
            # ----------------------------------------------------------

            associated_detections = [
                person_detection
            ]

            for detection in detections:

                if detection.class_name == "Person":
                    continue

                if self._is_inside_person(
                    person.box_xyxy,
                    detection.box_xyxy,
                ):
                    associated_detections.append(
                        detection
                    )

            # ----------------------------------------------------------
            # Use the existing PPE decision layer.
            # ----------------------------------------------------------

            from app.ppe_logic import evaluate_compliance

            compliance = evaluate_compliance(
                associated_detections,
                person_class_names=["Person"],
                required_ppe_classes=(
                    self.required_ppe_classes
                ),
            )

            # ----------------------------------------------------------
            # Extract this person's result.
            # ----------------------------------------------------------

            if compliance.people:

                person_result = compliance.people[0]

            else:

                person_result = PersonCompliance(
                    person_index=0,
                    person_confidence=person.confidence,
                    compliant=False,
                    missing_ppe=(
                        self.required_ppe_classes
                        or []
                    ),
                    violations=[
                        f"missing_{ppe}"
                        for ppe in (
                            self.required_ppe_classes
                            or []
                        )
                    ],
                )

            state = TrackCompliance(
                track_id=person.track_id,
                person_confidence=person.confidence,
                compliant=person_result.compliant,
                present_ppe=list(
                    person_result.present_ppe
                ),
                missing_ppe=list(
                    person_result.missing_ppe
                ),
                violations=list(
                    person_result.violations
                ),
            )

            # ----------------------------------------------------------
            # Store state by persistent track ID.
            # ----------------------------------------------------------

            self.states[
                person.track_id
            ] = state

            results.append(state)

        # Remove tracks that are no longer active.
        active_ids = {
            person.track_id
            for person in tracked_people
        }

        self.states = {
            track_id: state
            for track_id, state in self.states.items()
            if track_id in active_ids
        }

        return results

    @staticmethod
    def _is_inside_person(
        person_box: List[float],
        detection_box: List[float],
    ) -> bool:
        """
        Determine whether a detection belongs to a person.

        A detection is associated when its center point lies
        inside the person's bounding box.
        """

        px1, py1, px2, py2 = person_box

        dx1, dy1, dx2, dy2 = detection_box

        center_x = (
            dx1 + dx2
        ) / 2.0

        center_y = (
            dy1 + dy2
        ) / 2.0

        return (
            px1 <= center_x <= px2
            and
            py1 <= center_y <= py2
        )

    def get(
        self,
        track_id: int,
    ) -> TrackCompliance | None:
        """Return the latest state for a tracked person."""

        return self.states.get(track_id)

    def clear(self) -> None:
        """Clear all tracked compliance state."""

        self.states.clear()