"""
Safety event management.

Converts confirmed temporal PPE violations into
application-level safety events.

Pipeline:

    TemporalViolationEngine
            ↓
       SafetyEventManager
            ↓
    Confirmed Safety Events
            ↓
       Active / Resolved
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List


# ---------------------------------------------------------------------------
# Severity configuration
# ---------------------------------------------------------------------------

VIOLATION_SEVERITY = {
    "missing_Hardhat": "HIGH",
    "violation_Hardhat": "HIGH",

    "missing_Goggles": "HIGH",
    "violation_Goggles": "HIGH",

    "missing_Gloves": "MEDIUM",
    "violation_Gloves": "MEDIUM",

    "missing_Mask": "MEDIUM",
    "violation_Mask": "MEDIUM",

    "missing_Safety Vest": "HIGH",
    "violation_Safety Vest": "HIGH",
}


# ---------------------------------------------------------------------------
# Safety event
# ---------------------------------------------------------------------------

@dataclass
class SafetyEvent:
    """
    Application-level representation of a PPE safety event.
    """

    event_id: str

    track_id: int

    event_type: str

    violation: str

    severity: str

    start_frame: int

    confirmed_frame: int

    duration_frames: int

    status: str = "confirmed"

    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    resolved_frame: int | None = None

    def resolve(
        self,
        frame_number: int,
    ) -> None:
        """Mark the event as resolved."""

        self.status = "resolved"

        self.resolved_frame = frame_number

    def to_dict(self) -> dict:
        """Convert event to JSON-compatible dictionary."""

        return {
            "event_id": self.event_id,
            "track_id": self.track_id,
            "event_type": self.event_type,
            "violation": self.violation,
            "severity": self.severity,
            "start_frame": self.start_frame,
            "confirmed_frame": self.confirmed_frame,
            "duration_frames": self.duration_frames,
            "status": self.status,
            "created_at": self.created_at,
            "resolved_frame": self.resolved_frame,
        }


# ---------------------------------------------------------------------------
# Safety event manager
# ---------------------------------------------------------------------------

class SafetyEventManager:
    """
    Maintains confirmed PPE safety events.

    Responsibilities:

    - Create events from temporal confirmations.
    - Avoid duplicate active events.
    - Track active violations.
    - Resolve events when violations disappear.
    - Maintain event history.
    """

    def __init__(self):
        self.events: List[SafetyEvent] = []

        self.active_events: Dict[
            tuple[int, str],
            SafetyEvent,
        ] = {}

        self._next_event_id = 1

    # ------------------------------------------------------------------
    # Event creation
    # ------------------------------------------------------------------

    def create_from_violation(
        self,
        track_id: int,
        violation: str,
        start_frame: int,
        confirmed_frame: int,
        duration_frames: int,
    ) -> SafetyEvent:
        """
        Create a safety event from a confirmed temporal violation.

        If the same violation is already active for the same track,
        the existing event is returned instead of creating a duplicate.
        """

        key = (
            track_id,
            violation,
        )

        existing = self.active_events.get(key)

        if existing is not None:
            return existing

        event_id = (
            f"evt_{self._next_event_id:06d}"
        )

        self._next_event_id += 1

        severity = VIOLATION_SEVERITY.get(
            violation,
            "MEDIUM",
        )

        event = SafetyEvent(
            event_id=event_id,
            track_id=track_id,
            event_type="PPE_VIOLATION",
            violation=violation,
            severity=severity,
            start_frame=start_frame,
            confirmed_frame=confirmed_frame,
            duration_frames=duration_frames,
        )

        self.events.append(event)

        self.active_events[key] = event

        return event

    # ------------------------------------------------------------------
    # Resolve violation
    # ------------------------------------------------------------------

    def resolve_violation(
        self,
        track_id: int,
        violation: str,
        frame_number: int,
    ) -> SafetyEvent | None:
        """
        Resolve an active violation.

        Returns the resolved event or None if no active event exists.
        """

        key = (
            track_id,
            violation,
        )

        event = self.active_events.pop(
            key,
            None,
        )

        if event is None:
            return None

        event.resolve(
            frame_number
        )

        return event

    # ------------------------------------------------------------------
    # Query active events
    # ------------------------------------------------------------------

    def get_active_events(
        self,
    ) -> List[SafetyEvent]:
        """Return currently active safety events."""

        return list(
            self.active_events.values()
        )

    # ------------------------------------------------------------------
    # Query history
    # ------------------------------------------------------------------

    def get_all_events(
        self,
    ) -> List[SafetyEvent]:
        """Return complete event history."""

        return list(
            self.events
        )

    # ------------------------------------------------------------------
    # Track cleanup
    # ------------------------------------------------------------------

    def remove_track(
        self,
        track_id: int,
        frame_number: int,
    ) -> None:
        """
        Resolve all active violations associated
        with a disappeared track.
        """

        keys = [
            key
            for key in self.active_events
            if key[0] == track_id
        ]

        for key in keys:

            event = self.active_events.pop(
                key
            )

            event.resolve(
                frame_number
            )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all events and active state."""

        self.events.clear()

        self.active_events.clear()

        self._next_event_id = 1