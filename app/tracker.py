"""
Multi-person tracking + person-crop PPE detection.

Architecture:

    Frame
      |
      +--------------------+
      |                    |
      v                    v
YOLOv8n COCO          Custom PPE YOLO
Person detection      on each person crop
      |                    |
      v                    |
   ByteTrack               |
      |                    |
      v                    v
Persistent IDs      PPE detections
      |                    |
      +---------+----------+
                |
                v
       Track-specific PPE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from app.model_service import Detection


@dataclass
class TrackedPerson:
    """A person tracked across video frames."""

    track_id: int
    confidence: float
    box_xyxy: List[float]

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "confidence": round(self.confidence, 4),
            "box": [
                round(value, 2)
                for value in self.box_xyxy
            ],
        }


@dataclass
class TrackingResult:
    """
    Result of:

        YOLOv8n Person + ByteTrack
        +
        person-crop PPE detection
    """

    tracked_people: List[TrackedPerson]
    detections: List[Detection]


class PersonTracker:
    """
    Person tracking is performed exclusively by YOLOv8n COCO.

    PPE detection is performed separately on each tracked person's
    crop using the custom PPE model.

    This is intentionally different from running the PPE model once
    on the entire frame because the people in the target video are
    relatively small.
    """

    PERSON_CLASS_ID = 0

    # Classes that are useful for compliance.
    PPE_CLASSES = {
        "Hardhat",
        "NO-Hardhat",
        "Gloves",
        "NO-Gloves",
        "Goggles",
        "NO-Goggles",
        "Mask",
        "NO-Mask",
        "Safety Vest",
        "NO-Safety Vest",
    }

    def __init__(
        self,
        model,
        person_model,
        tracker_config: str = "bytetrack.yaml",
        confidence_threshold: float = 0.20,
        iou_threshold: float = 0.7,
        ppe_confidence_threshold: float = 0.01,
        crop_padding: float = 0.15,
    ):
        self.model = model
        self.person_model = person_model

        self.tracker_config = tracker_config
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # The custom model needs a lower threshold because several
        # PPE classes are weakly detected on this dataset.
        self.ppe_confidence_threshold = (
            ppe_confidence_threshold
        )

        # Expand person crops slightly so objects such as hardhats
        # that extend above the person box are not clipped.
        self.crop_padding = crop_padding

    # ------------------------------------------------------------------
    # PERSON TRACKING
    # ------------------------------------------------------------------

    def _track_people(self, frame) -> List[TrackedPerson]:
        """
        Detect people with COCO YOLOv8n and assign ByteTrack IDs.
        """

        results = self.person_model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
        )

        tracked_people: List[TrackedPerson] = []

        if not results:
            return tracked_people

        result = results[0]

        if result.boxes is None:
            return tracked_people

        boxes = result.boxes

        if boxes.id is None:
            return tracked_people

        for index in range(len(boxes)):

            track_id_value = boxes.id[index]

            if track_id_value is None:
                continue

            track_id = int(
                track_id_value.item()
            )

            confidence = float(
                boxes.conf[index].item()
            )

            box = [
                float(value)
                for value in boxes.xyxy[index].tolist()
            ]

            tracked_people.append(
                TrackedPerson(
                    track_id=track_id,
                    confidence=confidence,
                    box_xyxy=box,
                )
            )

        return tracked_people

    # ------------------------------------------------------------------
    # CROP HELPERS
    # ------------------------------------------------------------------

    def _get_padded_crop(
        self,
        frame,
        box: List[float],
    ):
        """
        Extract a padded crop around one person.

        Returns:

            crop,
            offset_x,
            offset_y
        """

        frame_height, frame_width = frame.shape[:2]

        x1, y1, x2, y2 = box

        person_width = max(
            1.0,
            x2 - x1,
        )

        person_height = max(
            1.0,
            y2 - y1,
        )

        pad_x = person_width * self.crop_padding
        pad_y = person_height * self.crop_padding

        crop_x1 = max(
            0,
            int(x1 - pad_x),
        )

        crop_y1 = max(
            0,
            int(y1 - pad_y),
        )

        crop_x2 = min(
            frame_width,
            int(x2 + pad_x),
        )

        crop_y2 = min(
            frame_height,
            int(y2 + pad_y),
        )

        crop = frame[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]

        return (
            crop,
            crop_x1,
            crop_y1,
        )

    # ------------------------------------------------------------------
    # PPE DETECTION
    # ------------------------------------------------------------------

    def _detect_ppe_for_person(
        self,
        frame,
        person: TrackedPerson,
    ) -> List[Detection]:
        """
        Run the custom PPE model on one person's crop.

        PPE boxes are converted back to full-frame coordinates.
        """

        crop, offset_x, offset_y = (
            self._get_padded_crop(
                frame,
                person.box_xyxy,
            )
        )

        if crop is None:
            return []

        if crop.size == 0:
            return []

        ppe_model = getattr(self.model, "_model", None)
        if ppe_model is not None:
            raw_results = ppe_model.predict(
                source=crop,
                imgsz=640,
                conf=self.ppe_confidence_threshold,
                iou=0.5,
                device=0,
                verbose=False,
            )
            detections = []
            if raw_results and raw_results[0].boxes is not None:
                result = raw_results[0]
                names = self.model.class_names
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    class_name = (
                        names.get(class_id, str(class_id))
                        if isinstance(names, dict)
                        else names[class_id]
                    )
                    xyxy = box.xyxy[0].tolist()
                    detections.append(
                        Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            box_xyxy=[float(v) for v in xyxy],
                        )
                    )
        else:
            detections = self.model.predict(
                crop,
                confidence_threshold=self.ppe_confidence_threshold,
                iou_threshold=0.5,
            )

        results: List[Detection] = []

        for detection in detections:

            class_name = detection.class_name

            # Ignore unrelated classes such as Ladder, Fall-Detected,
            # etc. They are not PPE compliance classes.
            if class_name not in self.PPE_CLASSES:
                continue

            x1, y1, x2, y2 = (
                detection.box_xyxy
            )

            # Convert crop coordinates back into
            # original frame coordinates.
            full_box = [
                x1 + offset_x,
                y1 + offset_y,
                x2 + offset_x,
                y2 + offset_y,
            ]

            results.append(
                Detection(
                    class_id=detection.class_id,
                    class_name=class_name,
                    confidence=detection.confidence,
                    box_xyxy=full_box,
                )
            )

        return results

    # ------------------------------------------------------------------
    # DUPLICATE SUPPRESSION
    # ------------------------------------------------------------------

    @staticmethod
    def _box_iou(
        a: List[float],
        b: List[float],
    ) -> float:
        """
        Calculate IoU between two boxes.
        """

        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(
            0.0,
            ix2 - ix1,
        )

        ih = max(
            0.0,
            iy2 - iy1,
        )

        intersection = iw * ih

        area_a = max(
            0.0,
            ax2 - ax1,
        ) * max(
            0.0,
            ay2 - ay1,
        )

        area_b = max(
            0.0,
            bx2 - bx1,
        ) * max(
            0.0,
            by2 - by1,
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    def _deduplicate(
        self,
        detections: List[Detection],
    ) -> List[Detection]:
        """
        Keep the strongest detection for highly overlapping
        detections of the same class.
        """

        output: List[Detection] = []

        # Process strongest detections first.
        ordered = sorted(
            detections,
            key=lambda d: d.confidence,
            reverse=True,
        )

        for detection in ordered:

            duplicate = False

            for existing in output:

                if (
                    existing.class_name
                    != detection.class_name
                ):
                    continue

                if self._box_iou(
                    existing.box_xyxy,
                    detection.box_xyxy,
                ) >= 0.50:
                    duplicate = True
                    break

            if not duplicate:
                output.append(detection)

        return output

    # ------------------------------------------------------------------
    # PUBLIC UPDATE
    # ------------------------------------------------------------------

    def update(
        self,
        frame,
    ) -> TrackingResult:
        """
        Run:

            1. YOLOv8n Person detection
            2. ByteTrack
            3. PPE detection on each person crop

        """

        tracked_people = self._track_people(
            frame
        )

        all_ppe_detections: List[
            Detection
        ] = []

        # --------------------------------------------------------------
        # Detect PPE independently for every tracked person.
        # --------------------------------------------------------------

        for person in tracked_people:

            person_detections = (
                self._detect_ppe_for_person(
                    frame,
                    person,
                )
            )

            person_detections = (
                self._deduplicate(
                    person_detections
                )
            )

            all_ppe_detections.extend(
                person_detections
            )

        return TrackingResult(
            tracked_people=tracked_people,
            detections=all_ppe_detections,
        )

    def reset(self) -> None:
        """
        Reset ByteTrack state.
        """

        self.person_model.predictor = None