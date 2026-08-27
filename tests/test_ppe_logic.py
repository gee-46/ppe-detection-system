"""
Unit tests for app/ppe_logic.py using synthetic Detection objects.

These prove the compliance decision logic itself is correct, independent
of which model produced the detections — important because the real PPE
model doesn't exist yet, so this is the only way to verify this layer
right now.
"""

from app.model_service import Detection
from app.ppe_logic import evaluate_compliance


def _det(name, conf=0.9):
    return Detection(class_id=0, class_name=name, confidence=conf, box_xyxy=[0, 0, 10, 10])


def test_not_configured_when_no_required_classes_set():
    result = evaluate_compliance([_det("person")], person_class_names=["person"], required_ppe_classes=[])
    assert result.overall_status == "not_configured"


def test_no_person_detected():
    result = evaluate_compliance(
        [_det("helmet")], person_class_names=["person"], required_ppe_classes=["helmet", "vest"]
    )
    assert result.overall_status == "no_person_detected"
    assert result.persons_detected == 0


def test_fully_compliant_person():
    dets = [_det("person"), _det("helmet"), _det("vest")]
    result = evaluate_compliance(dets, person_class_names=["person"], required_ppe_classes=["helmet", "vest"])
    assert result.overall_status == "compliant"
    assert result.violations == []
    assert result.people[0].compliant is True


def test_violation_missing_one_item():
    dets = [_det("person"), _det("vest")]
    result = evaluate_compliance(dets, person_class_names=["person"], required_ppe_classes=["helmet", "vest"])
    assert result.overall_status == "non_compliant"
    assert "person_0_missing_helmet" in result.violations
    assert result.people[0].missing_ppe == ["helmet"]
    assert result.people[0].present_ppe == ["vest"]


def test_violation_missing_all_items():
    dets = [_det("person")]
    result = evaluate_compliance(dets, person_class_names=["person"], required_ppe_classes=["helmet", "vest"])
    assert result.overall_status == "non_compliant"
    assert set(result.people[0].missing_ppe) == {"helmet", "vest"}


def test_multiple_people_mixed_compliance():
    # Two people detected, PPE items present in frame satisfy both (frame-level
    # association, documented limitation vs. per-person IoU association).
    dets = [_det("person"), _det("person"), _det("helmet"), _det("vest")]
    result = evaluate_compliance(dets, person_class_names=["person"], required_ppe_classes=["helmet", "vest"])
    assert result.persons_detected == 2
    assert result.overall_status == "compliant"


def test_custom_person_class_name():
    dets = [_det("worker"), _det("helmet")]
    result = evaluate_compliance(dets, person_class_names=["worker"], required_ppe_classes=["helmet"])
    assert result.persons_detected == 1
    assert result.overall_status == "compliant"
