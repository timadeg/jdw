import subprocess
import os

HOME = os.getcwd()
dataset_location = "/home/sdp360/dl_task2/Food-Items-detection-7"

# Path to the model's weights file
weights_path = "/home/sdp360/dl_task2/runs/detect/train5/weights/best.pt"

# Validate Custom Model
subprocess.run(["yolo", "task=detect", "mode=val", f"model={weights_path}", f"data={dataset_location}/data.yaml"])

# Inference with Custom Model
subprocess.run(["yolo", "task=detect", "mode=predict", f"model={weights_path}", "conf=0.25", f"source={dataset_location}/test/images", "save=True"])

