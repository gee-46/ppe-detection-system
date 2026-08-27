"""
PPE compliance decision layer.

The YOLO model detects objects/classes. This module converts those
detections into PPE compliance decisions.

Dataset classes:
    0  Fall-Detected
    1  Gloves
    2  Goggles
    3  Hardhat
    4  Ladder
    5  Mask
    6  NO-Gloves
    7  NO-Goggles
    8  NO-Hardhat
    9  NO-Mask
    10 NO-Safety Vest
    11 Person
    12 Safety Cone
    13 Safety Vest

Required PPE:
    Hardhat
    Gloves
    Goggles
    Mask
    Safety Vest

Explicit violation classes:
    NO-Hardhat
    NO-Gloves
    NO-Goggles
    NO-Mask
    NO-Safety Vest

The compliance engine uses bounding-box geometry to associate PPE
detections with individual people instead of treating PPE detected
anywhere in an image as belonging to every person.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from app import config
from app.model_service import Detection


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PersonCompliance:
    """Compliance result for one detected person."""

    person_index: int
    person_confidence: float
    compliant: bool

    present_ppe: List[str] = field(default_factory=list)
    missing_ppe: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "person_index": self.person_index,
            "person_confidence": round(
                self.person_confidence,
                4,
            ),
            "compliant": self.compliant,
            "present_ppe": self.present_ppe,
            "missing_ppe": self.missing_ppe,
            "violations": self.violations,
        }


@dataclass
class ComplianceResult:
    """Overall PPE compliance result."""

    persons_detected: int
    required_ppe_classes: List[str]
    overall_status: str

    people: List[PersonCompliance] = field(
        default_factory=list
    )

    violations: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict:
        return {
            "persons_detected": self.persons_detected,
            "required_ppe_classes": self.required_ppe_classes,
            "overall_status": self.overall_status,
            "people": [
                person.to_dict()
                for person in self.people
            ],
            "violations": self.violations,
        }


# ---------------------------------------------------------------------------
# Bounding-box utilities
# ---------------------------------------------------------------------------

def _box_iou(
    box_a: List[float],
    box_b: List[float],
) -> float:
    """
    Calculate IoU between two bounding boxes.

    Bounding-box format:

        [x1, y1, x2, y2]
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)

    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union_area = (
        area_a
        + area_b
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def _box_center(
    box: List[float],
) -> tuple[float, float]:
    """Return the center point of a bounding box."""

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def _point_inside_box(
    point: tuple[float, float],
    box: List[float],
) -> bool:
    """Return True when a point lies inside a bounding box."""

    x, y = point

    x1, y1, x2, y2 = box

    return (
        x1 <= x <= x2
        and y1 <= y <= y2
    )


# ---------------------------------------------------------------------------
# Person/PPE association
# ---------------------------------------------------------------------------

def _is_associated_with_person(
    person: Detection,
    ppe: Detection,
) -> bool:
    """
    Determine whether a PPE detection belongs to a person.

    Association occurs when:

        1. PPE center lies inside the person's box

    OR:

        2. PPE and person have meaningful bounding-box overlap.

    This prevents PPE detected on the opposite side of an image
    from automatically being assigned to the person.
    """

    if not person.box_xyxy:
        return False

    if not ppe.box_xyxy:
        return False

    ppe_center = _box_center(
        ppe.box_xyxy
    )

    # Strong spatial relationship.
    if _point_inside_box(
        ppe_center,
        person.box_xyxy,
    ):
        return True

    # Secondary spatial relationship.
    return (
        _box_iou(
            person.box_xyxy,
            ppe.box_xyxy,
        )
        >= 0.05
    )


# ---------------------------------------------------------------------------
# Main compliance engine
# ---------------------------------------------------------------------------

def evaluate_compliance(
    detections: List[Detection],
    person_class_names: List[str] | None = None,
    required_ppe_classes: List[str] | None = None,
) -> ComplianceResult:
    """
    Evaluate PPE compliance independently for every detected person.

    A person is compliant only when:

        - every required PPE item is associated with that person
        - no explicit NO-* violation is associated with that person

    Explicit NO-* classes take priority over positive PPE detections.
    """

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    person_class_names = (
        person_class_names
        if person_class_names is not None
        else config.PERSON_CLASS_NAMES
    )

    required_ppe_classes = (
        required_ppe_classes
        if required_ppe_classes is not None
        else config.REQUIRED_PPE_CLASSES
    )

    # -----------------------------------------------------------------------
    # Find people
    # -----------------------------------------------------------------------

    persons = [
        detection
        for detection in detections
        if detection.class_name
        in person_class_names
    ]

    # -----------------------------------------------------------------------
    # No PPE configuration
    # -----------------------------------------------------------------------

    if not required_ppe_classes:

        return ComplianceResult(
            persons_detected=len(persons),
            required_ppe_classes=[],
            overall_status="not_configured",
            people=[],
            violations=[],
        )

    # -----------------------------------------------------------------------
    # No people detected
    # -----------------------------------------------------------------------

    if not persons:

        return ComplianceResult(
            persons_detected=0,
            required_ppe_classes=required_ppe_classes,
            overall_status="no_person_detected",
            people=[],
            violations=[],
        )

    # -----------------------------------------------------------------------
    # PPE classes
    # -----------------------------------------------------------------------

    positive_detections = [
        detection
        for detection in detections
        if detection.class_name
        in required_ppe_classes
    ]

    negative_to_positive = {
        "NO-Gloves": "Gloves",
        "NO-Goggles": "Goggles",
        "NO-Hardhat": "Hardhat",
        "NO-Mask": "Mask",
        "NO-Safety Vest": "Safety Vest",
    }

    negative_detections = [
        detection
        for detection in detections
        if detection.class_name
        in negative_to_positive
    ]

    # -----------------------------------------------------------------------
    # Evaluate each person
    # -----------------------------------------------------------------------

    people_results: List[PersonCompliance] = []

    all_violations: List[str] = []

    for person_index, person in enumerate(persons):

        # ---------------------------------------------------------------
        # Positive PPE associated with this person
        # ---------------------------------------------------------------

        associated_positive = [
            detection
            for detection in positive_detections
            if _is_associated_with_person(
                person,
                detection,
            )
        ]

        # ---------------------------------------------------------------
        # Negative PPE associated with this person
        # ---------------------------------------------------------------

        associated_negative = [
            detection
            for detection in negative_detections
            if _is_associated_with_person(
                person,
                detection,
            )
        ]

        # ---------------------------------------------------------------
        # Present PPE
        # ---------------------------------------------------------------

        present_ppe = sorted(
            {
                detection.class_name
                for detection in associated_positive
            }
        )

        # ---------------------------------------------------------------
        # Explicit violations
        # ---------------------------------------------------------------

        explicit_violations = sorted(
            {
                negative_to_positive[
                    detection.class_name
                ]
                for detection in associated_negative
            }
        )

        # ---------------------------------------------------------------
        # Missing PPE
        # ---------------------------------------------------------------

        missing_ppe = [
            ppe
            for ppe in required_ppe_classes
            if (
                ppe not in present_ppe
                or ppe in explicit_violations
            )
        ]

        # ---------------------------------------------------------------
        # Violation messages
        # ---------------------------------------------------------------

        violations = [
            f"missing_{ppe}"
            for ppe in missing_ppe
        ]

        violations.extend(
            f"violation_{ppe}"
            for ppe in explicit_violations
            if f"violation_{ppe}" not in violations
        )

        # ---------------------------------------------------------------
        # Compliance
        # ---------------------------------------------------------------

        compliant = (
            len(missing_ppe) == 0
            and len(explicit_violations) == 0
        )

        # ---------------------------------------------------------------
        # Store result
        # ---------------------------------------------------------------

        person_result = PersonCompliance(
            person_index=person_index,
            person_confidence=person.confidence,
            compliant=compliant,
            present_ppe=present_ppe,
            missing_ppe=missing_ppe,
            violations=violations,
        )

        people_results.append(
            person_result
        )

        # ---------------------------------------------------------------
        # Overall violations
        # ---------------------------------------------------------------

        for violation in violations:

            all_violations.append(
                f"person_{person_index}_{violation}"
            )

    # -----------------------------------------------------------------------
    # Overall compliance
    # -----------------------------------------------------------------------

    overall_status = (
        "compliant"
        if all(
            person.compliant
            for person in people_results
        )
        else "non_compliant"
    )

    # -----------------------------------------------------------------------
    # Return
    # -----------------------------------------------------------------------

    return ComplianceResult(
        persons_detected=len(persons),
        required_ppe_classes=required_ppe_classes,
        overall_status=overall_status,
        people=people_results,
        violations=all_violations,
    )   