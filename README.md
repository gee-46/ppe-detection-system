

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFAB?style=for-the-badge&logo=pytorch&logoColor=white)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-FF6B35?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-00C9A7?style=for-the-badge)]()

<br/>

> **"Safety isn't expensive, it's priceless."**  
> Automating PPE compliance so every worker goes home safe.

<br/>

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [The Problem](#-the-real-world-problem)
- [How It Works](#️-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Features](#-features)
- [Detection Workflow](#-detection-workflow)
- [Roadmap](#️-roadmap)
- [Learning Outcomes](#-learning-outcomes)
- [Author](#-author)

---

## 🔍 Overview

**Real-Time PPE Detection System** is an end-to-end Computer Vision application that automatically identifies whether workers are wearing required Personal Protective Equipment — helmets, safety vests, gloves, and more — using state-of-the-art object detection.

Built on **YOLOv8** (You Only Look Once v8) and **OpenCV**, this system delivers fast, accurate inference on both static images and live video streams, making it production-ready for integration into industrial safety pipelines.

> ⚡ Part of an **Applied AI Engineering** roadmap focused on building real-world, deployable AI systems.

---

## 🌍 The Real-World Problem

Every year, thousands of workplace injuries occur due to missing or improper use of Personal Protective Equipment. In environments like:

| Environment | Common PPE Required |
|---|---|
| 🏗️ Construction Sites | Helmets, safety vests, boots |
| 🏭 Factories & Warehouses | Gloves, goggles, hard hats |
| ⚗️ Chemical Plants | Full-body suits, face shields |
| 🔌 Electrical Facilities | Insulated gloves, arc flash gear |

**Manual monitoring** of PPE compliance across large facilities is:
- ❌ Time-consuming and error-prone
- ❌ Impossible to scale with limited supervisors
- ❌ Reactive rather than preventive
- ❌ Subject to human fatigue and oversight

This system makes compliance monitoring **proactive, automated, and scalable** — detecting violations in real time before accidents happen.

---

## ⚙️ How It Works

The system is built on a simple, modular inference pipeline — any input source (image, video file, or live webcam stream) is passed through YOLOv8 for detection, then annotated with bounding boxes and confidence scores before being written to an output file or displayed on screen.

Each component is decoupled, making it easy to swap in a custom-trained PPE model, add a new input source, or plug in an alerting layer without touching the rest of the pipeline.

See the [Detection Workflow](#-detection-workflow) section for the full pipeline diagram.

---

## 🧠 Tech Stack

| Technology | Role | Version |
|---|---|---|
| **Python** | Core language | 3.8+ |
| **YOLOv8 (Ultralytics)** | Object detection model | Latest |
| **OpenCV** | Image/video processing | 4.x |
| **PyTorch** | Deep learning backend | 2.x |
| **NumPy** | Numerical operations | 1.24+ |

---

## 📂 Project Structure

```bash
real-time-ppe-detection-system/
│
├── app/
│   └── inference.py          # 🔍 Core detection script
│
├── data/                     # 📁 Input images & videos
│   ├── images/
│   └── videos/
│
├── models/                   # 🤖 YOLOv8 model weights (.pt files)
│   └── yolov8n.pt
│
├── outputs/                  # 📤 Annotated detection results
│   ├── images/
│   └── videos/
│
├── requirements.txt          # 📦 Python dependencies
└── README.md                 # 📖 Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

---

### 1. Clone the Repository

```bash
git clone https://github.com/gee-46/real-time-ppe-detection-system.git
cd real-time-ppe-detection-system
```

---

### 2. Set Up a Virtual Environment

**Create the environment:**
```bash
python -m venv venv
```

**Activate — Windows:**
```bash
venv\Scripts\activate
```

**Activate — macOS / Linux:**
```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install ultralytics opencv-python numpy
```

Or using `requirements.txt` (recommended):
```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
```

---

### 4. Run Inference

```bash
python app/inference.py
```

> 📌 By default, the script runs detection on images inside the `data/` directory.  
> Modify `inference.py` to point to your own images, video files, or webcam feed.

---

## 🧩 Detection Workflow

The detection pipeline follows a clean, modular flow from any input source to annotated output:

```
📷 Input Source (Image / Video / Webcam)
          │
          ▼
┌─────────────────────┐
│   Frame Capture     │  ← OpenCV VideoCapture / imread
│   & Preprocessing   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   YOLOv8 Inference  │  ← Ultralytics model.predict()
│   (Object Detection)│
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Post-Processing    │  ← Bounding boxes, class labels,
│  & Visualization    │     confidence scores overlay
└─────────┬───────────┘
          │
          ▼
📤 Output (Annotated Image / Video / Alert)
```

---

## ✨ Features

| Feature | Status |
|---|---|
| 🎯 YOLOv8 object detection | ✅ Available |
| 🖼️ Image-based detection | ✅ Available |
| 📦 Bounding box visualization | ✅ Available |
| 🏷️ Class label + confidence overlay | ✅ Available |
| 🎥 Video file inference | ✅ Available |
| 📹 Live webcam detection | 🔜 Coming Soon |
| 🦺 PPE-specific custom model | 🔜 Coming Soon |
| 🚨 Real-time alert system | 🔜 Coming Soon |
| 🌐 FastAPI backend | 🔜 Coming Soon |

---

## 🗺️ Roadmap

```
Phase 1 — Foundation (Current)         ✅
├── YOLOv8 integration
├── Image-based detection pipeline
└── Bounding box + label visualization

Phase 2 — Live Detection               🔜
├── Webcam / RTSP stream support
├── Multi-frame processing
└── Real-time FPS optimization

Phase 3 — PPE-Specific Model           🔜
├── Custom dataset (helmets, vests, gloves)
├── Fine-tune YOLOv8 on PPE classes
└── Confidence threshold tuning

Phase 4 — Alerts & Backend            🔜
├── Violation alert system (sound/email)
├── FastAPI REST API integration
└── Detection logs & reporting

Phase 5 — Deployment                  🔜
├── Docker containerization
├── Edge deployment (Raspberry Pi / Jetson)
└── Cloud API (AWS / GCP)
```

---

## 🎓 Learning Outcomes

This project covers the following concepts hands-on:

- ✅ **Computer Vision Fundamentals** — image processing, frame handling, color spaces
- ✅ **Object Detection** — understanding YOLO architecture, bounding boxes, IoU, NMS
- ✅ **Real-Time Inference** — optimizing detection pipelines for speed
- ✅ **OpenCV Workflows** — reading images/videos, drawing annotations, displaying results
- ✅ **Deep Learning Integration** — using pretrained models from Ultralytics
- ✅ **AI for Safety Systems** — applying AI to solve real industrial problems

---

## 👤 Author

<div align="center">

<br/>

**Built by [gee-46](https://github.com/gee-46)**

Part of an **Applied AI Engineering** roadmap — building production-ready AI systems for real-world deployment.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-gee--46-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/gee-46)

<br/>

*If this project helped you, consider giving it a ⭐ on GitHub!*

<br/>

</div>

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:00C9A7,50:F7C948,100:FF6B35&height=6&section=footer"/>

</div>
