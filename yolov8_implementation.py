import os
import subprocess

subprocess.run(["nvidia-smi"])
HOME = os.getcwd()
print(HOME)


import ultralytics
ultralytics.checks()

from ultralytics import YOLO

datasets_dir = os.path.join(HOME, 'datasets')
os.makedirs(datasets_dir, exist_ok=True)



from roboflow import Roboflow
#rf = Roboflow(api_key="rj9X7b8W62ij7cjtiOO6")
#project = rf.workspace("deep-learning-coursework").project("food-items-detection-sm84b")

#dataset = project.version(7).download("yolov8") ## unaugmented version


dataset_location = "home/sdp360/dl_task2/datasets/Food-Items-detection-7"  


subprocess.run(f"yolo task=detect mode=train model=yolov8s.pt data=/home/sdp360/dl_task2/Food-Items-detection-7/data.yaml epochs=50 imgsz=800 plots=True", shell=True)
