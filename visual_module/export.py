# Converting pt to engine model

from ultralytics import YOLOE
model = YOLOE("yolo11n.pt")  # your model path
model.export(format="engine", opset=14, simplify=True, dynamic=False, device=0)
