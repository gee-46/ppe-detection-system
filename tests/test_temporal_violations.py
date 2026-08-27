"""
Tests for the temporal PPE violation engine.

The engine should:

1. Ignore short-lived violations.
2. Confirm persistent violations.
3. Reset a violation when PPE returns.
4. Keep different tracks independent.
5. Reject invalid configuration.
"""

import pytest

from app.temporal_violations import (
    TemporalViolationEngine,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_invalid_confirmation_frames():

    with pytest.raises(ValueError):

        TemporalViolationEngine(
            violation_confirm_frames=0
        )


# ---------------------------------------------------------------------------
# Single-frame violation
# ---------------------------------------------------------------------------

def test_single_frame_violation_not_confirmed():

    engine = TemporalViolationEngine(
        violation_confirm_frames=3
    )

    events = engine.update(
        frame_number=1,
        track_id=5,
        violations=["missing_Hardhat"],
    )

    assert events == []

    state = engine.get_state(
        track_id=5,
        violation="missing_Hardhat",
    )

    assert state is not None
    assert state.consecutive_frames == 1
    assert state.confirmed is False


# ---------------------------------------------------------------------------
# Persistent violation
# ---------------------------------------------------------------------------

def test_persistent_violation_is_confirmed():

    engine = TemporalViolationEngine(
        violation_confirm_frames=3
    )

    assert engine.update(
        frame_number=1,
        track_id=5,
        violations=["missing_Hardhat"],
    ) == []

    assert engine.update(
        frame_number=2,
        track_id=5,
        violations=["missing_Hardhat"],
    ) == []

    events = engine.update(
        frame_number=3,
        track_id=5,
        violations=["missing_Hardhat"],
    )

    assert len(events) == 1

    event = events[0]

    assert event.track_id == 5
    assert event.violation == "missing_Hardhat"
    assert event.start_frame == 1
    assert event.confirmed_frame == 3
    assert event.duration_frames == 3
    assert event.status == "confirmed"


# ---------------------------------------------------------------------------
# No duplicate events
# ---------------------------------------------------------------------------

def test_confirmed_violation_does_not_repeat():

    engine = TemporalViolationEngine(
        violation_confirm_frames=3
    )

    for frame in range(1, 6):

        events = engine.update(
            frame_number=frame,
            track_id=5,
            violations=["missing_Hardhat"],
        )

        if frame < 3:
            assert events == []

        else:
            assert len(events) == 0 or (
                frame == 3
            )

    confirmed = engine.get_confirmed_events()

    assert len(confirmed) == 1


# ---------------------------------------------------------------------------
# Violation recovery
# ---------------------------------------------------------------------------

def test_violation_resets_when_ppe_returns():

    engine = TemporalViolationEngine(
        violation_confirm_frames=3
    )

    engine.update(
        frame_number=1,
        track_id=5,
        violations=["missing_Hardhat"],
    )

    engine.update(
        frame_number=2,
        track_id=5,
        violations=["missing_Hardhat"],
    )

    # PPE returns.
    engine.update(
        frame_number=3,
        track_id=5,
        violations=[],
    )

    state = engine.get_state(
        track_id=5,
        violation="missing_Hardhat",
    )

    assert state is None


# ---------------------------------------------------------------------------
# Independent tracks
# ---------------------------------------------------------------------------

def test_tracks_are_independent():

    engine = TemporalViolationEngine(
        violation_confirm_frames=3
    )

    # Person 5 has a violation.
    engine.update(
        frame_number=1,
        track_id=5,
        violations=["missing_Hardhat"],
    )

    engine.update(
        frame_number=2,
        track_id=5,
        violations=["missing_Hardhat"],
    )

    # Person 7 is compliant.
    engine.update(
        frame_number=1,
        track_id=7,
        violations=[],
    )

    state_5 = engine.get_state(
        track_id=5,
        violation="missing_Hardhat",
    )

    state_7 = engine.get_state(
        track_id=7,
        violation="missing_Hardhat",
    )

    assert state_5 is not None
    assert state_5.consecutive_frames == 2

    assert state_7 is None


# ---------------------------------------------------------------------------
# Multiple violations for one person
# ---------------------------------------------------------------------------

def test_multiple_violations_are_tracked_independently():

    engine = TemporalViolationEngine(
        violation_confirm_frames=2
    )

    events = engine.update(
        frame_number=1,
        track_id=5,
        violations=[
            "missing_Hardhat",
            "missing_Gloves",
        ],
    )

    assert events == []

    events = engine.update(
        frame_number=2,
        track_id=5,
        violations=[
            "missing_Hardhat",
            "missing_Gloves",
        ],
    )

    assert len(events) == 2

    violation_names = {
        event.violation
        for event in events
    }

    assert violation_names == {
        "missing_Hardhat",
        "missing_Gloves",
    }


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

def test_clear_removes_all_state():

    engine = TemporalViolationEngine(
        violation_confirm_frames=2
    )

    engine.update(
        frame_number=1,
        track_id=5,
        violations=["missing_Hardhat"],
    )

    engine.update(
        frame_number=2,
        track_id=5,
        violations=["missing_Hardhat"],
    )

    assert len(
        engine.get_confirmed_events()
    ) == 1

    engine.clear()

    assert engine.get_confirmed_events() == []

    assert engine.get_state(
        track_id=5,
        violation="missing_Hardhat",
    ) is None