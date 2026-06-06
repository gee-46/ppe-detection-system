from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Run object detection
results = model(
    "https://ultralytics.com/images/bus.jpg",
    save=True
)

print("Detection completed successfully!")
