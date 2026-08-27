"""
Temporal PPE violation engine.

Converts frame-level PPE compliance results into
time-consistent safety events.

Pipeline:

    TrackCompliance
          ↓
    TemporalViolationEngine
          ↓
    Confirmed violations
          ↓
    Resolved violations

A violation is NOT confirmed immediately.

It must persist for a configurable number of
consecutive frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class ViolationEvent:
    """
    Represents a confirmed PPE violation event.
    """

    track_id: int
    violation: str
    start_frame: int
    confirmed_frame: int
    duration_frames: int
    status: str = "confirmed"

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "violation": self.violation,
            "start_frame": self.start_frame,
            "confirmed_frame": self.confirmed_frame,
            "duration_frames": self.duration_frames,
            "status": self.status,
        }


@dataclass
class ViolationState:
    """
    Internal temporal state for one track + violation.
    """

    track_id: int
    violation: str
    first_seen_frame: int
    consecutive_frames: int = 0
    confirmed: bool = False


class TemporalViolationEngine:
    """
    Converts frame-level violations into persistent events.

    Example:

        violation_confirm_frames = 3

        Frame 10 → missing_Hardhat
        Frame 11 → missing_Hardhat
        Frame 12 → missing_Hardhat

        Frame 12 → violation confirmed
    """

    def __init__(
        self,
        violation_confirm_frames: int = 3,
    ):
        if violation_confirm_frames < 1:
            raise ValueError(
                "violation_confirm_frames must be >= 1"
            )

        self.violation_confirm_frames = (
            violation_confirm_frames
        )

        self.states: Dict[
            tuple[int, str],
            ViolationState,
        ] = {}

        self.confirmed_events: List[
            ViolationEvent
        ] = []

    def update(
        self,
        frame_number: int,
        track_id: int,
        violations: List[str],
    ) -> List[ViolationEvent]:
        """
        Update temporal state for one tracked person.

        Returns newly confirmed violation events.
        """

        current_violations: Set[str] = set(
            violations
        )

        new_events: List[ViolationEvent] = []

        # --------------------------------------------------------------
        # Update violations currently visible.
        # --------------------------------------------------------------

        for violation in current_violations:

            key = (
                track_id,
                violation,
            )

            state = self.states.get(key)

            if state is None:

                state = ViolationState(
                    track_id=track_id,
                    violation=violation,
                    first_seen_frame=frame_number,
                    consecutive_frames=1,
                )

                self.states[key] = state

            else:

                state.consecutive_frames += 1

            # ----------------------------------------------------------
            # Confirm only after persistence threshold.
            # ----------------------------------------------------------

            if (
                not state.confirmed
                and state.consecutive_frames
                >= self.violation_confirm_frames
            ):

                state.confirmed = True

                event = ViolationEvent(
                    track_id=track_id,
                    violation=violation,
                    start_frame=state.first_seen_frame,
                    confirmed_frame=frame_number,
                    duration_frames=(
                        state.consecutive_frames
                    ),
                )

                self.confirmed_events.append(
                    event
                )

                new_events.append(event)

        # --------------------------------------------------------------
        # Handle violations that disappeared.
        # --------------------------------------------------------------

        track_keys = [
            key
            for key in self.states
            if key[0] == track_id
        ]

        for key in track_keys:

            violation = key[1]

            if violation not in current_violations:

                del self.states[key]

        return new_events

    def remove_track(
        self,
        track_id: int,
    ) -> None:
        """
        Remove all temporal state associated
        with a track that disappeared.
        """

        keys = [
            key
            for key in self.states
            if key[0] == track_id
        ]

        for key in keys:
            del self.states[key]

    def get_state(
        self,
        track_id: int,
        violation: str,
    ) -> ViolationState | None:
        """Return current temporal state."""

        return self.states.get(
            (track_id, violation)
        )

    def get_confirmed_events(
        self,
    ) -> List[ViolationEvent]:
        """Return all confirmed events."""

        return list(
            self.confirmed_events
        )

    def clear(self) -> None:
        """Clear all temporal state and events."""

        self.states.clear()
        self.confirmed_events.clear()