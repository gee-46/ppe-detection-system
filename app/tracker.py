"""
Multi-person tracking service.

Pipeline:

    Frame
      ↓
    YOLO detection of all PPE classes
      ↓
    ByteTrack
      ↓
    Filter tracked Person detections
      ↓
    Track-aware PPE compliance

The model performs one YOLO + ByteTrack inference per frame.

The returned TrackingResult contains:

    tracked_people
        Person detections with persistent ByteTrack IDs.

    detections
        ALL detections from the same YOLO inference.

This allows the video pipeline to associate PPE detections with
tracked people without running a second YOLO inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.model_service import Detection


@dataclass
class TrackedPerson:
    """Represents a person tracked across video frames."""

    track_id: int
    confidence: float
    box_xyxy: List[float]

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "confidence": round(
                self.confidence,
                4,
            ),
            "box": [
                round(value, 2)
                for value in self.box_xyxy
            ],
        }


@dataclass
class TrackingResult:
    """
    Result of one YOLO + ByteTrack inference.

    tracked_people:
        Person detections that received ByteTrack IDs.

    detections:
        All YOLO detections from this frame.
    """

    tracked_people: List[TrackedPerson]

    detections: List[Detection]


class PersonTracker:
    """
    Multi-person tracker using Ultralytics ByteTrack.

    YOLO detects all configured PPE classes.

    ByteTrack assigns tracking IDs to the detections.

    We then keep only detections whose class is Person.

    This means:

        YOLO
          ↓
        Person + Hardhat + Gloves + ...
          ↓
        ByteTrack
          ↓
        Person → tracked_people
        PPE    → detections
    """

    PERSON_CLASS_NAMES = {
        "Person",
        "person",
    }

    def __init__(
        self,
        model,
        tracker_config: str = "bytetrack.yaml",
        confidence_threshold: float = 0.10,
        iou_threshold: float = 0.7,
    ):
        self.model = model

        self.tracker_config = (
            tracker_config
        )

        self.confidence_threshold = (
            confidence_threshold
        )

        self.iou_threshold = (
            iou_threshold
        )

    def update(
        self,
        frame,
    ) -> TrackingResult:
        """
        Run one YOLO + ByteTrack inference.

        Returns:

            TrackingResult(
                tracked_people=[...],
                detections=[...],
            )

        The detections list contains every detection returned
        by YOLO.

        The tracked_people list contains only Person detections
        that received a ByteTrack ID.
        """

        # ---------------------------------------------------------------
        # Access the underlying Ultralytics model.
        # ---------------------------------------------------------------

        yolo_model = self.model._model

        # ---------------------------------------------------------------
        # Run YOLO + ByteTrack.
        #
        # IMPORTANT:
        #
        # There is intentionally NO `classes=[11]` here.
        #
        # We need all PPE detections from this same inference.
        # ---------------------------------------------------------------

        results = yolo_model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        # ---------------------------------------------------------------
        # No result.
        # ---------------------------------------------------------------

        if not results:

            return TrackingResult(
                tracked_people=[],
                detections=[],
            )

        result = results[0]

        # ---------------------------------------------------------------
        # No detections.
        # ---------------------------------------------------------------

        if result.boxes is None:

            return TrackingResult(
                tracked_people=[],
                detections=[],
            )

        boxes = result.boxes

        class_names = (
            self.model.class_names
        )

        detections: List[Detection] = []

        tracked_people: List[
            TrackedPerson
        ] = []

        # ---------------------------------------------------------------
        # Convert every YOLO detection.
        # ---------------------------------------------------------------

        for index in range(len(boxes)):

            # -----------------------------------------------------------
            # Class ID
            # -----------------------------------------------------------

            class_id = int(
                boxes.cls[index].item()
            )

            # -----------------------------------------------------------
            # Confidence
            # -----------------------------------------------------------

            confidence = float(
                boxes.conf[index].item()
            )

            # -----------------------------------------------------------
            # Bounding box
            # -----------------------------------------------------------

            box = [
                float(value)
                for value in (
                    boxes.xyxy[index]
                    .tolist()
                )
            ]

            # -----------------------------------------------------------
            # Class name
            # -----------------------------------------------------------

            if isinstance(
                class_names,
                dict,
            ):

                class_name = (
                    class_names.get(
                        class_id,
                        str(class_id),
                    )
                )

            else:

                class_name = (
                    class_names[class_id]
                )

            # -----------------------------------------------------------
            # Store ALL detections.
            # -----------------------------------------------------------

            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                box_xyxy=box,
            )

            detections.append(
                detection
            )

            # -----------------------------------------------------------
            # Only Person detections become tracked people.
            # -----------------------------------------------------------

            if (
                class_name
                not in self.PERSON_CLASS_NAMES
            ):
                continue

            # -----------------------------------------------------------
            # ByteTrack may not assign an ID.
            # -----------------------------------------------------------

            if boxes.id is None:
                continue

            track_id_value = (
                boxes.id[index]
            )

            if track_id_value is None:
                continue

            track_id = int(
                track_id_value.item()
            )

            # -----------------------------------------------------------
            # Store tracked person.
            # -----------------------------------------------------------

            tracked_people.append(
                TrackedPerson(
                    track_id=track_id,
                    confidence=confidence,
                    box_xyxy=box,
                )
            )

        return TrackingResult(
            tracked_people=tracked_people,
            detections=detections,
        )

    def reset(self) -> None:
        """
        Reset the internal Ultralytics tracking predictor.

        Call this when switching to another video or camera stream.
        """

        self.model._model.predictor = None