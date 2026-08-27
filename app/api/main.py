"""
FastAPI backend for the PPE Detection System.

The model is loaded exactly once, at application startup (see the
`lifespan` handler below) via ModelService.get_instance() — never inside a
request handler. All endpoints reuse that single loaded model.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import config
from app.model_service import ModelService, get_model_service
from app.ppe_logic import evaluate_compliance
from app.schemas import (
    ComplianceOut,
    DetectionOut,
    ErrorResponse,
    HealthResponse,
    ImagePredictionResponse,
    PersonComplianceOut,
    VideoPredictionResponse,
)
from app.video_detection import process_video_file_headless

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ppe.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup so the first request isn't slow and so
    # we fail fast (at boot) if the weights file is missing/corrupt.
    logger.info("Starting up: loading model...")
    try:
        ModelService.get_instance()
        logger.info("Model loaded successfully.")
    except Exception:
        logger.exception("Failed to load model at startup.")
        raise
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="PPE Detection System API",
    description="REST API for image/video PPE compliance detection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(config.OUTPUT_DIR)), name="outputs")


def _validate_upload(file: UploadFile, allowed_extensions: set):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed_extensions)}",
        )


def _compliance_to_schema(compliance) -> ComplianceOut:
    return ComplianceOut(
        persons_detected=compliance.persons_detected,
        required_ppe_classes=compliance.required_ppe_classes,
        overall_status=compliance.overall_status,
        people=[
            PersonComplianceOut(
                person_index=p.person_index,
                person_confidence=p.person_confidence,
                compliant=p.compliant,
                present_ppe=p.present_ppe,
                missing_ppe=p.missing_ppe,
            )
            for p in compliance.people
        ],
        violations=compliance.violations,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        service = get_model_service()
        model_loaded = True
        classes = list(service.class_names.values())
    except Exception as exc:
        logger.exception("Health check: model not available")
        model_loaded = False
        classes = []

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_path=config.MODEL_PATH,
        is_custom_ppe_model=config.IS_CUSTOM_PPE_MODEL,
        classes=classes,
        person_class_names=config.PERSON_CLASS_NAMES,
        required_ppe_classes=config.REQUIRED_PPE_CLASSES,
    )


@app.post(
    "/predict/image",
    response_model=ImagePredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict_image(file: UploadFile = File(...)):
    _validate_upload(file, config.ALLOWED_IMAGE_EXTENSIONS)

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(raw_bytes) > config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400, detail=f"File exceeds {config.MAX_UPLOAD_SIZE_MB}MB limit."
        )

    np_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image. File may be corrupt.")

    try:
        service = get_model_service()
        start = time.time()
        detections, annotated = service.predict_and_annotate(image)
        elapsed_ms = (time.time() - start) * 1000
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed.")

    compliance = evaluate_compliance(detections)

    out_name = f"{uuid.uuid4().hex}_{Path(file.filename).stem}.jpg"
    out_path = config.OUTPUT_DIR / out_name
    cv2.imwrite(str(out_path), annotated)

    return ImagePredictionResponse(
        success=True,
        is_custom_ppe_model=config.IS_CUSTOM_PPE_MODEL,
        filename=file.filename,
        detections=[
            DetectionOut(
                class_id=d.class_id, class_name=d.class_name, confidence=d.confidence, box=d.box_xyxy
            )
            for d in detections
        ],
        compliance=_compliance_to_schema(compliance),
        annotated_image_url=f"/outputs/{out_name}",
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.post(
    "/predict/video",
    response_model=VideoPredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict_video(file: UploadFile = File(...)):
    _validate_upload(file, config.ALLOWED_VIDEO_EXTENSIONS)

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(raw_bytes) > config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400, detail=f"File exceeds {config.MAX_UPLOAD_SIZE_MB}MB limit."
        )

    tmp_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    tmp_path = config.OUTPUT_DIR / tmp_name
    with open(tmp_path, "wb") as f:
        f.write(raw_bytes)

    try:
        start = time.time()
        stats = process_video_file_headless(str(tmp_path), sample_rate=config.VIDEO_FRAME_SAMPLE_RATE)
        elapsed_ms = (time.time() - start) * 1000
    except Exception:
        logger.exception("Video inference failed")
        raise HTTPException(status_code=500, detail="Video inference failed. File may be corrupt or unsupported.")
    finally:
        tmp_path.unlink(missing_ok=True)

    return VideoPredictionResponse(
        success=True,
        is_custom_ppe_model=config.IS_CUSTOM_PPE_MODEL,
        filename=file.filename,
        frames_total=stats["frames_total"],
        frames_analyzed=stats["frames_analyzed"],
        detections_by_class=stats["detections_by_class"],
        overall_compliance_status=stats["overall_compliance_status"],
        violation_frame_count=stats["violation_frame_count"],
        annotated_video_url=None,  # annotated video re-encoding not yet implemented, see README limitations
        processing_time_ms=round(elapsed_ms, 2),
    )
