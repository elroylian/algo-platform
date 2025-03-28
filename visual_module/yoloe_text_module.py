# yoloe_text_module.py

import cv2
import numpy as np
from ultralytics import YOLOE
import torch

device = 0 if torch.cuda.is_available() else "cpu"

def create_model_with_text_prompt(class_names):
    """
    Initialize YOLOE with text prompt classes.

    Args:
        class_names (list): List of class labels.

    Returns:
        model: YOLOE model instance with text-based prompt encoder.
    """
    model = YOLOE("yoloe-v8s-seg.pt")
    model.set_classes(class_names, model.get_text_pe(class_names))
    return model

def predict_with_model(model, image_np, class_names):
    """
    Run prediction with a text-prompt-initialized model.

    Args:
        model: YOLOE model
        image_np (np.ndarray): Image to predict
        class_names (list): Class labels

    Returns:
        Annotated image
    """
    results = model(image_np, device=device)
    labeldict = {idx: name for idx, name in enumerate(class_names)}

    try:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        indices = results[0].boxes.cls.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        for idx, box in enumerate(boxes):
            label = int(indices[idx])
            labelname = labeldict.get(label, f"Unknown-{label}")
            score = scores[idx]

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, f"{labelname}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    except Exception as e:
        print(f"Text prompt prediction error: {e}")

    return image_np