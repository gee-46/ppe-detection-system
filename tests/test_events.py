"""
Tests for the Safety Event Manager.
"""

from app.events import (
    SafetyEventManager,
)


# ---------------------------------------------------------------------------
# Event creation
# ---------------------------------------------------------------------------

def test_create_safety_event():

    manager = SafetyEventManager()

    event = manager.create_from_violation(
        track_id=5,
        violation="missing_Hardhat",
        start_frame=10,
        confirmed_frame=12,
        duration_frames=3,
    )

    assert event.event_id == "evt_000001"
    assert event.track_id == 5
    assert event.event_type == "PPE_VIOLATION"
    assert event.violation == "missing_Hardhat"
    assert event.severity == "HIGH"
    assert event.status == "confirmed"

    assert event.start_frame == 10
    assert event.confirmed_frame == 12
    assert event.duration_frames == 3


# ---------------------------------------------------------------------------
# Event dictionary
# ---------------------------------------------------------------------------

def test_event_to_dict():

    manager = SafetyEventManager()

    event = manager.create_from_violation(
        track_id=5,
        violation="missing_Gloves",
        start_frame=20,
        confirmed_frame=22,
        duration_frames=3,
    )

    data = event.to_dict()

    assert data["event_id"] == "evt_000001"
    assert data["track_id"] == 5
    assert data["event_type"] == "PPE_VIOLATION"
    assert data["violation"] == "missing_Gloves"
    assert data["severity"] == "MEDIUM"
    assert data["status"] == "confirmed"


# ---------------------------------------------------------------------------
# Duplicate prevention
# ---------------------------------------------------------------------------

def test_duplicate_active_violation_returns_same_event():

    manager = SafetyEventManager()

    first = manager.create_from_violation(
        track_id=5,
        violation="missing_Hardhat",
        start_frame=10,
        confirmed_frame=12,
        duration_frames=3,
    )

    second = manager.create_from_violation(
        track_id=5,
        violation="missing_Hardhat",
        start_frame=13,
        confirmed_frame=15,
        duration_frames=3,
    )

    assert first is second

    assert len(
        manager.get_all_events()
    ) == 1

    assert len(
        manager.get_active_events()
    ) == 1


# ---------------------------------------------------------------------------
# Different violations
# ---------------------------------------------------------------------------

def test_different_violations_create_separate_events():

    manager = SafetyEventManager()

    hardhat = manager.create_from_violation(
        track_id=5,
        violation="missing_Hardhat",
        start_frame=10,
        confirmed_frame=12,
        duration_frames=3,
    )

    gloves = manager.create_from_violation(
        track_id=5,
        violation="missing_Gloves",
        start_frame=10,
        confirmed_frame=12,
        duration_frames=3,
    )

    assert hardhat.event_id != gloves.event_id

    assert len(
        manager.get_all_events()
    ) == 2

    assert len(
        manager.get_active_events()
    ) == 2


# ---------------------------------------------------------------------------
# Different tracks
# ---------------------------------------------------------------------------

def test_different_tracks_create_separate_events():

    manager = SafetyEventManager()

    person_5 = manager.create_from_violation(
        track_id=5,
        violation="missing_Hardhat",
        start_frame=10,
        confirmed_frame=12,
        duration_frames=3,
    )

    person_7 = manager.create_from_violation(
        track_id=7,
        violation="missing_Hardhat",
        start_frame=10,
        confirmed_frame=12,
        duration_frames=3,
    )

    assert person_5.event_id != person_7.event_id

    assert len(
        manager.get_all_events()
    ) == 2


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_resolve_violation():

    manager = SafetyEventManager()

    event = manager.create_from_violation(
        track_id=5,
        violation="missing_Hardhat",
        start_frame=10,
        confirmed_frame=12,
        duration_frames=3,
    )

    resolved = manager.resolve_violation(
        track_id=5,
        violation="missing_Hardhat",
        frame_number=25,
    )

    assert resolved is event

    assert event.status == "resolved"
    assert event.resolved_frame == 25

    assert manager.get_active_events() == []

    # Event remains in history.
    assert len(
        manager.get_all_events()
    ) == 1


# ---------------------------------------------------------------------------
# Resolving nonexistent event
# ---------------------------------------------------------------------------

def test_resolve_nonexistent_violation():

    manager = SafetyEventManager()

    result = manager.resolve_violation(
        track_id=5,
        violation="missing_Hardhat",
        frame_number=20,
    )

    assert result is None


# ---------------------------------------------------------------------------
# Remove track
# ---------------------------------------------------------------------------

def test_remove_track_resolves_all_active_events():

    manager = SafetyEventManager()

    hardhat = manager.create_from_violation(
        track_id=5,
        violation="missing_Hardhat",
        start_frame=10,
        confirmed_frame=12,
        duration_frames=3,
    )

    gloves = manager.create_from_violation(
        track_id=5,
        violation="missing_Gloves",
        start_frame=10,
        confirmed_frame=12,
        duration_frames=3,
    )

    # Different person should remain active.
    other_person = manager.create_from_violation(
        track_id=7,
        violation="missing_Hardhat",
        start_frame=10,
        confirmed_frame=12,
        duration_frames=3,
    )

    manager.remove_track(
        track_id=5,
        frame_number=30,
    )

    assert hardhat.status == "resolved"
    assert hardhat.resolved_frame == 30

    assert gloves.status == "resolved"
    assert gloves.resolved_frame == 30

    assert other_person.status == "confirmed"

    active = manager.get_active_events()

    assert active == [other_person]


# ---------------------------------------------------------------------------
# Severity fallback
# ---------------------------------------------------------------------------

def test_unknown_violation_uses_medium_severity():

    manager = SafetyEventManager()

    event = manager.create_from_violation(
        track_id=5,
        violation="unknown_violation",
        start_frame=1,
        confirmed_frame=3,
        duration_frames=3,
    )

    assert event.severity == "MEDIUM"


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

def test_clear():

    manager = SafetyEventManager()

    manager.create_from_violation(
        track_id=5,
        violation="missing_Hardhat",
        start_frame=1,
        confirmed_frame=3,
        duration_frames=3,
    )

    manager.clear()

    assert manager.get_all_events() == []
    assert manager.get_active_events() == []


# ---------------------------------------------------------------------------
# Event IDs
# ---------------------------------------------------------------------------

def test_event_ids_increment():

    manager = SafetyEventManager()

    event_1 = manager.create_from_violation(
        track_id=1,
        violation="missing_Hardhat",
        start_frame=1,
        confirmed_frame=3,
        duration_frames=3,
    )

    event_2 = manager.create_from_violation(
        track_id=2,
        violation="missing_Goggles",
        start_frame=1,
        confirmed_frame=3,
        duration_frames=3,
    )

    assert event_1.event_id == "evt_000001"
    assert event_2.event_id == "evt_000002"