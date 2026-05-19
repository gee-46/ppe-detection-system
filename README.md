# Real-Time PPE Detection System

A real-time Computer Vision project built using YOLOv8 and OpenCV to detect Personal Protective Equipment (PPE) such as helmets and safety gear from images and video streams.

---

## 🚀 Project Overview

This project is designed to improve workplace safety by automatically detecting whether workers are wearing proper safety equipment.

The system uses a pretrained YOLOv8 object detection model for real-time inference and OpenCV for image/video processing.

---

## 🌍 Real-World Problem

In factories, warehouses, and construction sites, monitoring PPE compliance manually is difficult and error-prone.

Missing safety equipment can lead to:
- workplace accidents
- injuries
- compliance violations

This project aims to automate PPE monitoring using AI-powered computer vision.

---

## 🧠 Technologies Used

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- Deep Learning
- Computer Vision

---

## 📂 Project Structure

```bash
ppe-detection-system/
│
├── app/
│   └── inference.py
│
├── data/
├── models/
├── outputs/
│
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/real-time-ppe-detection-system.git
```

### 2. Navigate to Project

```bash
cd real-time-ppe-detection-system
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

#### Windows
```bash
venv\Scripts\activate
```

#### Mac/Linux
```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install ultralytics opencv-python
```

---

## ▶️ Running the Project

```bash
python app/inference.py
```

---

## 📸 Current Features

- Object detection using YOLOv8
- Real-time inference pipeline
- Bounding box visualization
- Image-based detection

---

## 🚀 Future Improvements

- Live webcam detection
- PPE-specific custom training
- Helmet and safety vest detection
- FastAPI backend integration
- Deployment support
- Real-time alert system

---

## 🧩 Sample Detection Workflow

```text
Input Image
    ↓
YOLOv8 Model
    ↓
Object Detection
    ↓
Bounding Boxes + Labels
    ↓
Output Image
```

---

## 🎯 Learning Outcomes

This project helped in understanding:

- Computer Vision pipelines
- Object Detection
- Real-time inference
- OpenCV workflows
- Deep Learning model integration

---

## 📌 Author

Built as part of an Applied AI Engineering roadmap focusing on real-world AI systems and deployment.
