"""Pydantic models for API request/response validation and serialization."""

from typing import List, Optional

from pydantic import BaseModel, Field


class DetectionOut(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box: List[float] = Field(..., description="[x1, y1, x2, y2] in pixels")


class PersonComplianceOut(BaseModel):
    person_index: int
    person_confidence: float
    compliant: bool
    present_ppe: List[str]
    missing_ppe: List[str]


class ComplianceOut(BaseModel):
    persons_detected: int
    required_ppe_classes: List[str]
    overall_status: str
    people: List[PersonComplianceOut]
    violations: List[str]


class ImagePredictionResponse(BaseModel):
    success: bool
    is_custom_ppe_model: bool
    filename: str
    detections: List[DetectionOut]
    compliance: ComplianceOut
    annotated_image_url: Optional[str] = None
    inference_time_ms: float


class VideoPredictionResponse(BaseModel):
    success: bool
    is_custom_ppe_model: bool
    filename: str
    frames_total: int
    frames_analyzed: int
    detections_by_class: dict
    overall_compliance_status: str
    violation_frame_count: int
    annotated_video_url: Optional[str] = None
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    is_custom_ppe_model: bool
    classes: List[str]
    person_class_names: List[str]
    required_ppe_classes: List[str]


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
