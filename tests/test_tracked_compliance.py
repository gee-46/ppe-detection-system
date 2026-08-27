"""
Tests for track-aware PPE compliance.

These tests verify that PPE detections are evaluated separately
for different tracked people.
"""

from app.tracked_compliance import (
    TrackComplianceManager,
)
from app.model_service import Detection
from app.tracker import TrackedPerson


def make_detection(
    class_id: int,
    class_name: str,
    confidence: float,
    box: list[float],
) -> Detection:

    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        box_xyxy=box,
    )


def make_person(
    track_id: int,
    box: list[float],
    confidence: float = 0.90,
) -> TrackedPerson:

    return TrackedPerson(
        track_id=track_id,
        confidence=confidence,
        box_xyxy=box,
    )


def test_two_people_receive_independent_compliance():

    manager = TrackComplianceManager(
        required_ppe_classes=[
            "Hardhat",
            "Gloves",
        ]
    )

    people = [
        # Person 1
        make_person(
            track_id=1,
            box=[0, 0, 200, 300],
        ),

        # Person 2
        make_person(
            track_id=2,
            box=[300, 0, 500, 300],
        ),
    ]

    detections = [
        # --------------------------------------------------------------
        # Person 1 has Hardhat + Gloves
        # --------------------------------------------------------------

        make_detection(
            class_id=3,
            class_name="Hardhat",
            confidence=0.90,
            box=[50, 30, 120, 100],
        ),

        make_detection(
            class_id=1,
            class_name="Gloves",
            confidence=0.90,
            box=[50, 150, 100, 220],
        ),

        # --------------------------------------------------------------
        # Person 2 has only Hardhat
        # --------------------------------------------------------------

        make_detection(
            class_id=3,
            class_name="Hardhat",
            confidence=0.90,
            box=[350, 30, 420, 100],
        ),
    ]

    results = manager.update(
        tracked_people=people,
        detections=detections,
    )

    assert len(results) == 2

    # --------------------------------------------------------------
    # Person 1
    # --------------------------------------------------------------

    person_1 = manager.get(1)

    assert person_1 is not None

    assert person_1.track_id == 1

    assert person_1.compliant is True

    assert "Hardhat" in person_1.present_ppe

    assert "Gloves" in person_1.present_ppe

    assert person_1.missing_ppe == []

    # --------------------------------------------------------------
    # Person 2
    # --------------------------------------------------------------

    person_2 = manager.get(2)

    assert person_2 is not None

    assert person_2.track_id == 2

    assert person_2.compliant is False

    assert "Hardhat" in person_2.present_ppe

    assert "Gloves" in person_2.missing_ppe


def test_compliance_state_is_stored_by_track_id():

    manager = TrackComplianceManager(
        required_ppe_classes=[
            "Hardhat",
        ]
    )

    person = make_person(
        track_id=42,
        box=[0, 0, 200, 300],
    )

    detections = [
        make_detection(
            class_id=3,
            class_name="Hardhat",
            confidence=0.95,
            box=[50, 30, 120, 100],
        )
    ]

    results = manager.update(
        tracked_people=[person],
        detections=detections,
    )

    assert len(results) == 1

    state = manager.get(42)

    assert state is not None

    assert state.track_id == 42

    assert state.compliant is True

    assert state.present_ppe == ["Hardhat"]


def test_inactive_tracks_are_removed():

    manager = TrackComplianceManager(
        required_ppe_classes=[
            "Hardhat",
        ]
    )

    person = make_person(
        track_id=10,
        box=[0, 0, 200, 300],
    )

    manager.update(
        tracked_people=[person],
        detections=[],
    )

    assert manager.get(10) is not None

    # No currently tracked people.
    manager.update(
        tracked_people=[],
        detections=[],
    )

    assert manager.get(10) is None