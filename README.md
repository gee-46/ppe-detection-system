# PPE Detection System

An end-to-end computer vision application for automated workplace PPE
(Personal Protective Equipment) compliance monitoring, built on YOLOv8,
OpenCV, and FastAPI.

> **Project status: infrastructure complete, model training blocked on dataset access.**
> Read the [Current Status](#current-status) section before anything else —
> it explains exactly what is real vs. temporary in this repo.

---

## Table of Contents

- [Current Status](#current-status)
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Training](#training)
- [Inference](#inference)
- [API](#api)
- [Frontend](#frontend)
- [Docker](#docker)
- [Deployment](#deployment)
- [Testing](#testing)
- [Getting Started](#getting-started)
- [Results](#results)
- [Known Limitations](#known-limitations)

---

## Current Status

Being direct about what is and isn't true right now, per component:

| Component | Status |
|---|---|
| Reusable model-loading service (`app/model_service.py`) | ✅ Implemented & tested |
| Image inference (`app/inference.py`) | ✅ Implemented & tested (real detections on `bus.jpg`) |
| Video inference (`app/video_detection.py`) | ✅ Implemented & tested (real detections on a test video) |
| PPE compliance logic (`app/ppe_logic.py`) | ✅ Implemented & unit-tested against synthetic detections. **Not yet meaningful against real PPE classes** — see below. |
| FastAPI backend (`/health`, `/predict/image`, `/predict/video`) | ✅ Implemented & tested (success + error paths, 15 automated tests passing) |
| Frontend (`frontend/index.html`) | ✅ Implemented, calls the real API contract, dynamically renders whatever classes the backend returns |
| Training pipeline (`app/train.py`) | ✅ Implemented, verified to correctly refuse to run against an empty/placeholder dataset |
| **Custom PPE model training** | ❌ **Blocked.** The sandbox this was built in cannot reach kaggle.com (network policy), so the real dataset (`shlokraval/ppe-dataset-yolov8`) has not been downloaded, inspected, or trained on. |
| Dockerfile | ⚠️ Written, follows standard practice, **not build-tested** — no Docker daemon available in the build sandbox |

**What this means concretely:** the system currently runs on the generic
COCO-pretrained `yolov8n.pt` (person, car, bus, etc. — 80 general object
classes, no PPE classes at all). Every part of the pipeline — detection,
the API, the frontend, the compliance-decision logic — is real, tested,
working code. It is not, however, currently capable of detecting helmets
or vests, because no PPE-specific model has been trained yet.

`app/config.py` documents exactly how to point the whole system at the
real trained model once it exists: set `MODEL_PATH`, `IS_CUSTOM_PPE_MODEL=true`,
and `REQUIRED_PPE_CLASSES` to the real class names. No other code changes
are required anywhere in the project.

---

## Overview

The system takes an image, video, or webcam frame, runs it through a YOLO
object-detection model, and evaluates whether each detected person is
wearing the required PPE. Detection and compliance decision-making are
deliberately separated into two layers (`model_service.py` vs.
`ppe_logic.py`) so the compliance rules can be reconfigured — different
required PPE, different person-class names — without touching detection
code.

---

## Features

| Feature | Status |
|---|---|
| YOLOv8 object detection | ✅ Available (temporary generic model) |
| Custom PPE-specific model | ❌ Blocked on dataset access |
| Image inference (CLI + API) | ✅ Available |
| Video file inference (CLI + API) | ✅ Available |
| Webcam inference (local demo) | ✅ Available (`app/video_detection.py --source webcam`) |
| PPE compliance decision logic | ✅ Available, dynamically configured, awaiting real classes |
| REST API (FastAPI) | ✅ Available |
| Web frontend | ✅ Available |
| Docker | ⚠️ Written, untested |

---

## Architecture

```
Image / Video / Webcam
        │
        ▼
     OpenCV (frame capture / decode)
        │
        ▼
  ModelService (YOLO, loaded once at startup)
        │
        ▼
   Object Detections (class, confidence, box)
        │
        ▼
   Confidence / IoU filtering (built into YOLO predict call)
        │
        ▼
   ppe_logic.evaluate_compliance()
        │  groups detections into "person" vs "PPE item" using
        │  PERSON_CLASS_NAMES / REQUIRED_PPE_CLASSES (config-driven)
        ▼
   Compliance Result (compliant / non_compliant / no_person_detected /
                       not_configured)
        │
        ▼
     FastAPI (/health, /predict/image, /predict/video)
        │
        ▼
   Frontend (frontend/index.html) — renders whatever classes/compliance
   state the backend actually returns; nothing about PPE classes is
   hardcoded in the UI.
```

---

## Tech Stack

| Technology | Role |
|---|---|
| Python 3.12 | Core language |
| Ultralytics YOLOv8 | Object detection model |
| PyTorch | Deep learning backend |
| OpenCV (headless) | Image/video decode, annotation |
| FastAPI + Uvicorn | REST API |
| Pydantic | Request/response validation |
| Vanilla HTML/CSS/JS | Frontend (no build step required) |
| pytest | Automated testing |
| Docker | Containerized deployment (untested, see status table) |

---

## Project Structure

```
ppe-detection-system/
│
├── app/
│   ├── config.py              # all env-driven settings — the single place
│   │                            to point the app at the real trained model
│   ├── model_service.py       # singleton YOLO loader, shared by everything
│   ├── inference.py           # image inference CLI
│   ├── detection_analysis.py  # detection summary/analysis CLI
│   ├── video_detection.py     # video file + webcam (consolidated, see below)
│   ├── ppe_logic.py           # compliance decision layer
│   ├── schemas.py             # Pydantic response models
│   ├── train.py               # training pipeline (blocked on dataset)
│   └── api/
│       └── main.py            # FastAPI app
│
├── data/
│   ├── ppe.yaml                # PLACEHOLDER dataset config — unverified classes
│   └── ppe_dataset/             # empty; real data goes here once obtained
│
├── models/
│   ├── yolov8n.pt               # temporary generic base model
│   └── trained/                 # trained best.pt/last.pt will go here
│
├── outputs/                     # annotated images/results written at runtime
├── frontend/
│   └── index.html                # single-file dashboard, no build step
├── tests/
│   ├── test_api.py
│   └── test_ppe_logic.py
│
├── requirements.txt
├── requirements-lock.txt        # exact tested versions
├── .gitignore
├── Dockerfile
├── .dockerignore
└── README.md
```

### Consolidation note

The original repo had three separate scripts that each reimplemented the
same "open a video capture, loop frames, run YOLO, draw boxes, show
window" logic: `video_detection.py`, `webcam_detection.py`, and
`person_detection.py` (a person-only-filtered webcam variant). These are
merged into a single `app/video_detection.py` with one shared capture
loop; `class_filter` reproduces the old person-only behavior for any
class name(s), and video-file-vs-webcam is just a different `source` to
`cv2.VideoCapture`, which was the only real difference between the
originals. A leftover unused `import mediapipe` from the old
`video_detection.py` was also removed.

---

## Dataset

**Intended source:** [Kaggle: shlokraval/ppe-dataset-yolov8](https://www.kaggle.com/datasets/shlokraval/ppe-dataset-yolov8)

**Status: not yet obtained.** The environment this repo was assembled in
cannot reach `kaggle.com` (confirmed directly — `curl -I https://www.kaggle.com`
returns `403 host_not_allowed`). `data/ppe.yaml` currently contains a
placeholder class list (`helmet, vest, person`) carried over from the
original repo — **this is an unverified guess, not confirmed against the
real dataset**, and is explicitly labeled as such in the file itself.

### What happens once the dataset is available

1. Extract it into `data/ppe_dataset/images/{train,val}` and
   `data/ppe_dataset/labels/{train,val}` (YOLO format: one `.txt` per
   image, `class_id x_center y_center width height`, normalized 0–1).
2. Read the dataset's own `data.yaml` to get the real class names/IDs —
   `data/ppe.yaml` gets regenerated from that, not assumed.
3. Run a dataset integrity check (image↔label counts, malformed/missing
   annotations, class balance) before training.
4. `app/train.py` already validates this automatically and will refuse to
   start a training run if `images/train` is empty — verified behavior,
   not aspirational.

Large dataset files are intentionally excluded from git via `.gitignore`
(`data/ppe_dataset/**` images/labels, `*.zip`) — only the folder structure
(`.gitkeep` files) and the eventual `data.yaml` should be committed.

---

## Training

```bash
# Once the real dataset is in place:
python -m app.train --epochs 50 --imgsz 640 --batch 16 --model yolov8n.pt

# Verify the pipeline wires together correctly with a tiny run first:
python -m app.train --smoke-test
```

All hyperparameters (`--epochs`, `--imgsz`, `--batch`, `--device`,
`--model`) are CLI flags with modest defaults — nothing here defaults to
an unreasonably large run. Device is auto-detected (CUDA if available,
else CPU); CPU-only environments get a logged warning recommending a
GPU machine for anything beyond a smoke test.

Trained checkpoints are copied to `models/trained/best.pt` and
`models/trained/last.pt` — a stable path independent of the ultralytics
`runs/` folder — so `MODEL_PATH` can point there without guessing a run
name.

**This has not been run against real data yet** — see [Dataset](#dataset).

### What the metrics mean (for once training does run)

- **Precision** — of everything the model labeled as (e.g.) "helmet", what
  fraction actually was a helmet. High precision = few false alarms.
- **Recall** — of every real helmet in the images, what fraction the model
  actually found. High recall = few missed detections.
- **mAP@50** — mean Average Precision at an IoU threshold of 0.5 (a
  detection counts as correct if its box overlaps the true box by ≥50%).
  The standard single-number summary of detection quality.
- **mAP@50-95** — mAP averaged over IoU thresholds 0.5 to 0.95 — a
  stricter, more comprehensive measure of localization quality.

No metrics are reported in this README because no training run has been
performed — reporting numbers here would be fabricating them, which was
an explicit constraint on this project.

---

## Inference

```bash
# Image
python -m app.inference path/to/image.jpg --out outputs/result.jpg

# Detection summary/analysis
python -m app.detection_analysis path/to/image.jpg

# Video file (interactive GUI window, press 'q' to quit)
python -m app.video_detection --source path/to/video.mp4

# Webcam (interactive GUI window)
python -m app.video_detection --source webcam
python -m app.video_detection --source webcam --filter person   # person-only, like the old person_detection.py
```

The webcam/GUI scripts require a display and `opencv-python` (not
`-headless`) — see the note in `requirements.txt`. The API's
`/predict/video` endpoint is the non-interactive, server-side equivalent
and does not require a display.

---

## API

Base URL: `http://localhost:8000` (default).

### `GET /health`

Returns model load state and current configuration.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "/app/models/yolov8n.pt",
  "is_custom_ppe_model": false,
  "classes": ["person", "bicycle", "car", "...80 total..."],
  "person_class_names": ["person"],
  "required_ppe_classes": []
}
```

### `POST /predict/image`

Multipart form upload, field name `file`. Accepts `.jpg .jpeg .png .bmp .webp`.

```json
{
  "success": true,
  "is_custom_ppe_model": false,
  "filename": "site_photo.jpg",
  "detections": [
    {"class_id": 0, "class_name": "person", "confidence": 0.87, "box": [48.6, 398.6, 245.3, 902.7]}
  ],
  "compliance": {
    "persons_detected": 3,
    "required_ppe_classes": [],
    "overall_status": "not_configured",
    "people": [],
    "violations": []
  },
  "annotated_image_url": "/outputs/22eb10ae10fc_site_photo.jpg",
  "inference_time_ms": 187.4
}
```

Once `REQUIRED_PPE_CLASSES` is configured (post-training),
`overall_status` becomes `compliant` / `non_compliant`, and `people[]` /
`violations[]` populate with real per-person results, e.g.:

```json
"compliance": {
  "persons_detected": 1,
  "required_ppe_classes": ["helmet", "vest"],
  "overall_status": "non_compliant",
  "people": [
    {"person_index": 0, "person_confidence": 0.91, "compliant": false,
     "present_ppe": ["vest"], "missing_ppe": ["helmet"]}
  ],
  "violations": ["person_0_missing_helmet"]
}
```

Error responses: `400` for wrong extension, empty file, oversized file,
or undecodable/corrupt image; `500` for an inference-time failure.

### `POST /predict/video`

Multipart form upload, field name `file`. Accepts `.mp4 .avi .mov .mkv`.
Samples every `VIDEO_FRAME_SAMPLE_RATE`-th frame (default 5) rather than
every frame, for reasonable response times on CPU.

```json
{
  "success": true,
  "is_custom_ppe_model": false,
  "filename": "site_walkthrough.mp4",
  "frames_total": 300,
  "frames_analyzed": 60,
  "detections_by_class": {"person": 210, "bus": 12},
  "overall_compliance_status": "not_configured",
  "violation_frame_count": 0,
  "annotated_video_url": null,
  "processing_time_ms": 8421.3
}
```

`annotated_video_url` is currently always `null` — see
[Known Limitations](#known-limitations).

---

## Frontend

`frontend/index.html` is a single self-contained file (no build step, no
npm install) — open it directly in a browser or serve it statically.

- Toggle between image/video mode
- Drag-and-drop or click-to-browse upload
- Live backend health/config display (shows whether the temporary or
  real PPE model is currently loaded)
- Renders detections, confidence bars, and compliance status **entirely
  from whatever the API returns** — no PPE class names are hardcoded in
  the UI, so it will correctly display real PPE results once the trained
  model is deployed, with zero frontend changes required

Run it:

```bash
# Terminal 1 — backend
uvicorn app.api.main:app --reload

# Terminal 2 — frontend (any static file server)
python -m http.server 5500 --directory frontend
# open http://localhost:5500
```

---

## Docker

```bash
docker build -t ppe-detection-system .
docker run -p 8000:8000 ppe-detection-system
```

To run the real trained model instead of the temporary one:

```bash
docker run -p 8000:8000 \
  -v $(pwd)/models/trained:/app/models/trained \
  -e MODEL_PATH=/app/models/trained/best.pt \
  -e IS_CUSTOM_PPE_MODEL=true \
  -e REQUIRED_PPE_CLASSES=helmet,vest \
  ppe-detection-system
```

**Honesty note:** the build sandbox used to assemble this project has no
Docker daemon, so this `Dockerfile` has not actually been built or run.
It follows standard, well-tested patterns (slim base image, cached
dependency layer, non-root-friendly, `HEALTHCHECK` against `/health`),
but please run `docker build` yourself before relying on it, and file/fix
anything that doesn't work — don't take "it should work" as "it works."

---

## Deployment

Recommended architecture:

```
Frontend (static file, any CDN/host)
        │  HTTP
        ▼
FastAPI backend (container, needs ~1-2GB RAM minimum for yolov8n-scale
                  models; more for larger YOLO variants)
        │
        ▼
YOLO model (loaded once at container startup)
```

**Hardware reality check:** this repo was built and tested on a
single-vCPU, ~4GB RAM sandbox with no GPU — the same class of environment
as most free-tier hosting (Render/Fly.io free tiers, HF Spaces free CPU,
etc.). Image inference on that hardware took ~150-250ms per image with
`yolov8n` — workable for a demo, not for high-throughput real-time video.
Do not choose a platform based on popularity alone; confirm it gives you
at least 1-2 CPU cores and 2GB+ RAM before deploying, or the container
may fail to start under load. GPU hosting (a paid tier, or Colab/Kaggle
notebooks for training only) is realistically required for real-time
video/webcam throughput, not for the image endpoint.

---

## Testing

```bash
pytest tests/ -v
```

15 tests, all passing as of this writing:
- API startup, `/health`, model-loaded state
- `/predict/image`: valid file → real detections; wrong extension → 400;
  corrupt file → 400; empty file → 400; missing field → 422
- `/predict/video`: wrong extension → 400; corrupt file → 500
- PPE compliance logic: not-configured state, no-person state, fully
  compliant, single violation, all-items-missing, multi-person, custom
  person-class-name — all against synthetic `Detection` objects, since
  the real model doesn't exist yet to generate real PPE detections

---

## Getting Started

```bash
git clone https://github.com/gee-46/ppe-detection-system.git
cd ppe-detection-system

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Run the API
uvicorn app.api.main:app --reload
# → http://localhost:8000/health

# Run the frontend (separate terminal)
python -m http.server 5500 --directory frontend
# → http://localhost:5500

# Run tests
pytest tests/ -v

# Try image inference directly
python -m app.inference bus.jpg --out outputs/result.jpg
```

---

## Results

No trained-model metrics are reported — training has not been performed
against the real PPE dataset yet (see [Dataset](#dataset) and
[Training](#training)). This section will be filled in with actual
Precision/Recall/mAP50/mAP50-95 numbers once that training run happens —
never invented ones.

---

## Known Limitations

- **No custom PPE model yet** — blocked on Kaggle dataset access from the
  build environment; the real dataset needs to be obtained and integrated
  by whoever has network access to Kaggle (see Dataset section).
- **Docker is untested** — no Docker daemon was available to verify the
  build in the environment this was built in.
- **Compliance association is frame-level, not per-person IoU-based** —
  `ppe_logic.py` currently checks "is this PPE class present anywhere in
  the frame" per detected person, not "is this specific helmet
  overlapping this specific person's bounding box." This is fine for
  single-worker frames; a busy multi-person frame could misattribute PPE
  between people. Documented as a known simplification, not hidden.
- **`/predict/video` does not return an annotated video file** — it
  returns aggregate detection/compliance statistics from sampled frames.
  Re-encoding and serving an annotated video is a reasonable follow-up
  but wasn't built to avoid over-scoping this pass.
- **Real-time webcam is a local/interactive script, not an API
  endpoint** — streaming inference over HTTP (e.g. via WebSocket) would
  be needed for a browser-based live camera feed; the current webcam
  support opens a local OpenCV window instead.
