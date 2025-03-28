# app.py

from flask import Flask, request, jsonify
import numpy as np
import cv2
import json
from PIL import Image
from io import BytesIO
import base64
from yoloe_visual_module import create_model_with_visual_prompt, predict_with_model

app = Flask(__name__)
model = None  # Store the model globally
class_names = []  # Store classes for the label dictionary

def read_image_from_file(file):
    """Convert file upload to BGR NumPy array."""
    img = Image.open(file.stream).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def read_image_from_base64(b64string):
    """Convert base64 image string to BGR NumPy array."""
    img_data = base64.b64decode(b64string)
    img = Image.open(BytesIO(img_data)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def encode_image_to_base64(img_np):
    """Encode BGR NumPy array to base64 string."""
    _, buffer = cv2.imencode('.jpg', img_np)
    return base64.b64encode(buffer).decode("utf-8")

@app.route('/set-prompt', methods=['POST'])
def set_prompt():
    global model, class_names

    source_file = request.files.get("source_image")
    bboxes_str = request.form.get("bboxes")
    classes_str = request.form.get("classes")

    if not source_file or not bboxes_str or not classes_str:
        return jsonify({"error": "Missing required fields."}), 400

    try:
        bboxes = np.array(json.loads(bboxes_str), dtype=np.float32)
        class_names = json.loads(classes_str)
        cls = np.arange(len(class_names), dtype=np.int32)
        visuals = {"bboxes": bboxes, "cls": cls}

        source_image_np = read_image_from_file(source_file)
        model = create_model_with_visual_prompt(source_image_np, visuals, class_names)

        return jsonify({"message": "Visual prompt model set successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stream-frame', methods=['POST'])
def stream_frame():
    global model, class_names

    if model is None:
        return jsonify({"error": "Visual prompt model not set. Call /set-prompt first."}), 400

    data = request.get_json()
    if "image_base64" not in data:
        return jsonify({"error": "Missing 'image_base64' field."}), 400

    try:
        frame_np = read_image_from_base64(data["image_base64"])
        result = predict_with_model(model, frame_np, class_names)
        result_base64 = encode_image_to_base64(result)

        return jsonify({"image_base64": result_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
