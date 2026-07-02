import kagglehub

path = kagglehub.dataset_download(
    "shlokraval/ppe-dataset-yolov8"
)

print("Path to dataset files:", path)