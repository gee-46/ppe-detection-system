from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Run detection
results = model("bus.jpg")

for result in results:

    boxes = result.boxes

    for box in boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        
        print(
            f"Object: {class_name} | "
            f"Confidence: {confidence:.2f}"
        )

