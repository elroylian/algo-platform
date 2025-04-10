from ultralytics import YOLO

try:
    model = YOLO("yolo11n.engine")  # make sure this path is correct
except Exception as e:
    print(f"[ERROR] Failed to load YOLOv11 engine: {e}")
    model = None

def predict_with_engine_model(frame_np):
    # Run inference using the model (handles resizing and preprocessing)
    results = model(frame_np)

    # Plot the boxes directly onto the frame
    result_img = results[0].plot()  # returns a BGR numpy array with boxes drawn

    return result_img
