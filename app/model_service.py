"""
Reusable model-loading / inference service.

This replaces the old pattern where every script (inference.py,
person_detection.py, webcam_detection.py, video_detection.py,
detection_analysis.py) independently did:

    model = YOLO("yolov8n.pt")

which reloaded the weights from disk every time a script ran, and would
have meant reloading per-request inside the FastAPI app too.

Here the model is loaded exactly once (module-level singleton) and shared
by every caller: the CLI scripts, the API, and the training/eval code.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from app import config

logger = logging.getLogger("ppe.model_service")


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: List[float] = field(default_factory=list)  # [x1, y1, x2, y2]

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "box": [round(v, 2) for v in self.box_xyxy],
        }


class ModelService:
    """Thread-safe singleton wrapper around a single loaded YOLO model."""

    _instance: Optional["ModelService"] = None
    _lock = threading.Lock()

    def __init__(self, model_path: str, device: str = "auto"):
        logger.info("Loading YOLO model from %s (device=%s)", model_path, device)
        self.model_path = model_path
        self.device = None if device == "auto" else device
        self._model = YOLO(model_path)
        self.class_names = self._model.names  # dict[int, str]
        logger.info("Model loaded. Classes: %s", self.class_names)

    @classmethod
    def get_instance(cls) -> "ModelService":
        """Return the shared ModelService, creating it on first use."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config.MODEL_PATH, config.DEVICE)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Force a reload on next get_instance() call (used by tests / hot-reload)."""
        with cls._lock:
            cls._instance = None

    def predict(
        self,
        image: np.ndarray,
        confidence_threshold: float = None,
        iou_threshold: float = None,
    ) -> List[Detection]:
        """Run inference on a single BGR image (numpy array) and return Detections."""
        conf = confidence_threshold if confidence_threshold is not None else config.CONFIDENCE_THRESHOLD
        iou = iou_threshold if iou_threshold is not None else config.IOU_THRESHOLD

        results = self._model.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=self.class_names.get(class_id, str(class_id)),
                        confidence=confidence,
                        box_xyxy=xyxy,
                    )
                )
        return detections

    def predict_and_annotate(self, image: np.ndarray, **kwargs):
        """Run inference and return (detections, annotated_bgr_image)."""
        conf = kwargs.get("confidence_threshold", config.CONFIDENCE_THRESHOLD)
        iou = kwargs.get("iou_threshold", config.IOU_THRESHOLD)

        results = self._model.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        annotated = result.plot()  # BGR numpy array with boxes drawn

        detections: List[Detection] = []
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=self.class_names.get(class_id, str(class_id)),
                        confidence=confidence,
                        box_xyxy=xyxy,
                    )
                )
        return detections, annotated


def get_model_service() -> ModelService:
    """Convenience accessor used throughout the project."""
    return ModelService.get_instance()
