from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List

from app.model_service import Detection
from app.tracker import TrackedPerson

POSITIVE_TO_NEGATIVE = {
    "Hardhat": "NO-Hardhat",
    "Gloves": "NO-Gloves",
    "Goggles": "NO-Goggles",
    "Mask": "NO-Mask",
    "Safety Vest": "NO-Safety Vest",
}

DEFAULT_REQUIRED_PPE = ["Hardhat", "Gloves", "Goggles", "Mask", "Safety Vest"]

MIN_CONFIDENCE = {
    "Hardhat": 0.10, "Gloves": 0.10, "Goggles": 0.10,
    "Mask": 0.10, "Safety Vest": 0.10,
    "NO-Hardhat": 0.01, "NO-Gloves": 0.01, "NO-Goggles": 0.01,
    "NO-Mask": 0.01, "NO-Safety Vest": 0.01,
}

HISTORY_SIZE = 5
VIOLATION_CONFIRMATIONS = 3

@dataclass
class TrackCompliance:
    track_id: int
    person_confidence: float
    compliant: bool
    present_ppe: List[str] = field(default_factory=list)
    missing_ppe: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "person_confidence": round(self.person_confidence, 4),
            "compliant": self.compliant,
            "present_ppe": self.present_ppe,
            "missing_ppe": self.missing_ppe,
            "violations": self.violations,
        }

class TrackComplianceManager:
    """Convert per-person PPE detections into track-specific compliance."""

    def __init__(self, required_ppe_classes: List[str] | None = None):
        self.required_ppe_classes = (
            list(required_ppe_classes)
            if required_ppe_classes is not None
            else list(DEFAULT_REQUIRED_PPE)
        )
        self.states: Dict[int, TrackCompliance] = {}
        self._negative_history: Dict[int, Dict[str, deque[bool]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=HISTORY_SIZE))
        )

    def _evaluate(self, track_id: int, detections: List[Detection]):
        best: Dict[str, Detection] = {}
        for detection in detections:
            name = detection.class_name
            threshold = MIN_CONFIDENCE.get(name)
            if threshold is None or detection.confidence < threshold:
                continue
            previous = best.get(name)
            if previous is None or detection.confidence > previous.confidence:
                best[name] = detection

        present, missing, violations = [], [], []
        for required in self.required_ppe_classes:
            negative = POSITIVE_TO_NEGATIVE.get(required)
            positive_detection = best.get(required)
            negative_detection = best.get(negative) if negative else None

            if negative:
                self._negative_history[track_id][negative].append(
                    negative_detection is not None
                )
                history = self._negative_history[track_id][negative]
            else:
                history = deque()

            confirmed_negative = (
                negative is not None
                and sum(history) >= VIOLATION_CONFIRMATIONS
            )

            if confirmed_negative:
                missing.append(required)
                violations.append(f"violation_{required}")
            elif positive_detection is not None:
                present.append(required)
            else:
                missing.append(required)
                violations.append(f"missing_{required}")

        return present, missing, violations

    @staticmethod
    def _assignment_score(person_box: List[float], detection_box: List[float]) -> float:
        px1, py1, px2, py2 = person_box
        dx1, dy1, dx2, dy2 = detection_box
        dcx, dcy = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
        if px1 <= dcx <= px2 and py1 <= dcy <= py2:
            return 1.0

        ix1, iy1 = max(px1, dx1), max(py1, dy1)
        ix2, iy2 = min(px2, dx2), min(py2, dy2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        person_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
        detection_area = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)
        union = person_area + detection_area - intersection
        return intersection / union if union > 0 else 0.0

    def update(self, tracked_people: List[TrackedPerson], detections: List[Detection]) -> List[TrackCompliance]:
        results: List[TrackCompliance] = []
        per_track: Dict[int, List[Detection]] = {p.track_id: [] for p in tracked_people}

        for detection in detections:
            if not detection.box_xyxy:
                continue
            best_person, best_score = None, 0.0
            for person in tracked_people:
                score = self._assignment_score(person.box_xyxy, detection.box_xyxy)
                if score > best_score:
                    best_score, best_person = score, person
            if best_person is not None and best_score > 0.0:
                per_track[best_person.track_id].append(detection)

        active_ids = {p.track_id for p in tracked_people}
        for track_id in list(self._negative_history):
            if track_id not in active_ids:
                del self._negative_history[track_id]

        for person in tracked_people:
            present, missing, violations = self._evaluate(
                person.track_id, per_track.get(person.track_id, [])
            )
            state = TrackCompliance(
                track_id=person.track_id,
                person_confidence=person.confidence,
                compliant=(not missing and not violations),
                present_ppe=present,
                missing_ppe=missing,
                violations=violations,
            )
            self.states[person.track_id] = state
            results.append(state)

        self.states = {k: v for k, v in self.states.items() if k in active_ids}
        return results

    def get(self, track_id: int) -> TrackCompliance | None:
        return self.states.get(track_id)

    def clear(self) -> None:
        self.states.clear()
        self._negative_history.clear()
