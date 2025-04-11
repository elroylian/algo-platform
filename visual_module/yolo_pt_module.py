from ultralytics import YOLO
import cv2

# Load the model once
model = YOLO("yolo11n.pt")

def predict_with_pt_model(frame_np):
    # Run inference using the model (handles resizing and preprocessing)
    results = model(frame_np, device=0)

    # Plot the boxes directly onto the frame
    result_img = results[0].plot()  # returns a BGR numpy array with boxes drawn

    return result_img
