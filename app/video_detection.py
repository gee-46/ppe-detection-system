import cv2
import os
import mediapipe
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Input video
video_path = "data/sample_video.mp4"

cap = cv2.VideoCapture(video_path)

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    start_time = time.time()

    # Run detection
    results = model(frame)

    annotated_frame = results[0].plot()

    end_time = time.time()

    fps = 1 / (end_time - start_time)

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Video Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
     break
   
cap.release()
cv2.destroyAllWindows()
