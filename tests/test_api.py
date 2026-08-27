"""
API tests for the trained PPE Detection System.

These tests validate:

    - API startup and health
    - custom PPE model loading
    - image inference
    - PPE compliance response
    - upload validation
    - corrupt/empty file handling
    - video validation

The inference test uses a real image from the PPE validation dataset
because the production model is trained on 14 PPE-related classes.
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = Path(
    r"C:\Users\Dell\.cache\kagglehub\datasets"
    r"\shlokraval\ppe-dataset-yolov8\versions\1"
)

VALIDATION_IMAGES = DATASET_ROOT / "valid" / "images"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    Create a FastAPI TestClient for the test module.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_ppe_image():
    """
    Select a real image from the PPE validation dataset.

    The trained model was trained on this dataset, so a validation
    image provides a representative end-to-end inference test.
    """

    assert VALIDATION_IMAGES.exists(), (
        f"PPE validation image directory does not exist: "
        f"{VALIDATION_IMAGES}"
    )

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    images = [
        path
        for path in VALIDATION_IMAGES.iterdir()
        if path.is_file()
        and path.suffix.lower() in image_extensions
    ]

    assert images, (
        f"No validation images found in "
        f"{VALIDATION_IMAGES}"
    )

    return images[0]


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    """
    Verify that the API is running and the trained PPE model is loaded.
    """

    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"

    assert body["model_loaded"] is True

    assert body["is_custom_ppe_model"] is True

    # The trained PPE model contains 14 classes.
    assert len(body["classes"]) == 14

    # Person class.
    assert "Person" in body["classes"]

    # Required PPE classes.
    assert "Hardhat" in body["classes"]
    assert "Gloves" in body["classes"]
    assert "Goggles" in body["classes"]
    assert "Mask" in body["classes"]
    assert "Safety Vest" in body["classes"]

    # Explicit violation classes.
    assert "NO-Hardhat" in body["classes"]
    assert "NO-Gloves" in body["classes"]
    assert "NO-Goggles" in body["classes"]
    assert "NO-Mask" in body["classes"]
    assert "NO-Safety Vest" in body["classes"]


# ---------------------------------------------------------------------------
# Image inference
# ---------------------------------------------------------------------------

def test_predict_image_valid_file(
    client,
    sample_ppe_image,
):
    """
    Verify the complete image inference pipeline:

        upload
            ↓
        validation
            ↓
        YOLO inference
            ↓
        PPE compliance
            ↓
        annotated image
    """

    with open(sample_ppe_image, "rb") as image_file:

        response = client.post(
            "/predict/image",
            files={
                "file": (
                    sample_ppe_image.name,
                    image_file,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    body = response.json()

    # API request succeeded.
    assert body["success"] is True

    # Detection response exists.
    assert "detections" in body
    assert isinstance(body["detections"], list)

    # Compliance response exists.
    assert "compliance" in body
    assert isinstance(body["compliance"], dict)

    # Compliance must return one of the supported states.
    assert body["compliance"]["overall_status"] in {
        "compliant",
        "non_compliant",
        "no_person_detected",
        "not_configured",
    }

    # Annotated image should be generated.
    assert body["annotated_image_url"] is not None


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

def test_predict_image_rejects_wrong_extension(client):
    """
    Unsupported file extensions should be rejected.
    """

    response = client.post(
        "/predict/image",
        files={
            "file": (
                "notes.txt",
                io.BytesIO(b"hello"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_predict_image_rejects_corrupt_file(client):
    """
    A file with an image extension but invalid image contents
    should be rejected.
    """

    response = client.post(
        "/predict/image",
        files={
            "file": (
                "fake.jpg",
                io.BytesIO(
                    b"this is definitely not a jpeg"
                ),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400


def test_predict_image_rejects_empty_file(client):
    """
    Empty image uploads should be rejected.
    """

    response = client.post(
        "/predict/image",
        files={
            "file": (
                "empty.jpg",
                io.BytesIO(b""),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400


def test_predict_image_missing_file_field(client):
    """
    Missing the required upload field should produce
    FastAPI's validation error.
    """

    response = client.post("/predict/image")

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Video validation
# ---------------------------------------------------------------------------

def test_predict_video_rejects_wrong_extension(client):
    """
    Unsupported video extensions should be rejected.
    """

    response = client.post(
        "/predict/video",
        files={
            "file": (
                "notes.txt",
                io.BytesIO(b"hello"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_predict_video_rejects_corrupt_file(client):
    """
    A corrupted video should not be processed successfully.

    The current API implementation returns HTTP 500 when OpenCV
    cannot process the uploaded video.
    """

    response = client.post(
        "/predict/video",
        files={
            "file": (
                "fake.mp4",
                io.BytesIO(
                    b"not a real video file"
                ),
                "video/mp4",
            )
        },
    )

    assert response.status_code == 500