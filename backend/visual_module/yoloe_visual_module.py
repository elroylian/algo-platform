# yoloe_visual_module.py

import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

def create_model_with_visual_prompt(source_image_np, visuals: dict, class_names: list):
    """
    Initialize YOLOE model and setup visual prompts.

    Args:
        source_image_np (np.ndarray): Source image as a NumPy array (BGR).
        visuals (dict): Dictionary with keys 'bboxes' and 'cls'.
        class_names (list): List of class names like ["chair", "orb"].

    Returns:
        model: YOLOE model with visual prompt encoder setup.
    """
    # Load model
    model = YOLOE("yoloe-v8s-seg.pt")

    # Write source image temporarily for visual prompt setup
    temp_filename = "temp_source_image.jpg"
    cv2.imwrite(temp_filename, source_image_np)

    # First stage: attach the visual prompt encoder
    model.predict(temp_filename, prompts=visuals, predictor=YOLOEVPSegPredictor)

    # Second stage: get the VPE returned
    model.predict(temp_filename, prompts=visuals, return_vpe=True)

    # Set the classes for the VPE
    model.set_classes(class_names, model.predictor.vpe)

    # Remove the visual predictor so model uses default predictor later
    model.predictor = None

    return model


def predict_with_model(model, target_image_np, class_names):
    """
    Predict on target image using the model with visual prompt encoder.

    Args:
        model: YOLOE model
        target_image_np (np.ndarray): Target image as a NumPy array (BGR)
        class_names (list): List of class names

    Returns:
        Annotated image with bounding boxes and labels
    """
    results = model.predict(target_image_np, device=0)
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
            cv2.rectangle(target_image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(target_image_np, f"{labelname}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    except Exception as e:
        print(f"Error during prediction: {e}")

    return target_image_np
