# Converting pt to engine model

from ultralytics import YOLO
model = YOLO("yolo11n.pt", task="detect")  # your model path
model.export(format="engine", opset=19, simplify=True, dynamic=False, device=0, verbose = True, half=True)
