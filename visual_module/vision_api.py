from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import json
from PIL import Image
from io import BytesIO
import base64

from yoloe_visual_module import create_model_with_visual_prompt, predict_with_model
from yoloe_text_module import create_model_with_text_prompt, predict_with_model as predict_text

from yolo_inference import preprocess, draw_boxes, COCO_CLASSES, load_engine, allocate_buffers, do_inference, postprocess
from yolo_pt_module import predict_with_pt_model
from yolo_engine_module import predict_with_engine_model


app = Flask(__name__)
CORS(app)

visual_model = None
text_model = None
text_class_names = []
class_names = []

def read_image_from_file(file):
    img = Image.open(file.stream).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def read_image_from_base64(b64string):
    img_data = base64.b64decode(b64string)
    img = Image.open(BytesIO(img_data)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def encode_image_to_base64(img_np):
    _, buffer = cv2.imencode('.jpg', img_np)
    return base64.b64encode(buffer).decode("utf-8")

@app.route('/set-prompt', methods=['POST'])
def set_prompt():
    global visual_model, class_names

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
        visual_model = create_model_with_visual_prompt(source_image_np, visuals, class_names)

        return jsonify({"message": "Visual prompt model set successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stream-frame', methods=['POST'])
def stream_frame():
    global visual_model, class_names

    if visual_model is None:
        return jsonify({"error": "Visual prompt model not set. Call /set-prompt first."}), 400

    data = request.get_json()
    if "image_base64" not in data:
        return jsonify({"error": "Missing 'image_base64' field."}), 400

    try:
        frame_np = read_image_from_base64(data["image_base64"])
        result = predict_with_model(visual_model, frame_np, class_names)
        result_base64 = encode_image_to_base64(result)

        return jsonify({"image_base64": result_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/set-text-prompt', methods=['POST'])
def set_text_prompt():
    global text_model, text_class_names
    try:
        data = request.get_json()
        if "classes" not in data:
            return jsonify({"error": "Missing 'classes' field."}), 400

        text_class_names = data["classes"]
        text_model = create_model_with_text_prompt(text_class_names)

        return jsonify({"message": "Text prompt model set successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stream-frame-text', methods=['POST'])
def stream_frame_text():
    global text_model, text_class_names

    if text_model is None:
        return jsonify({"error": "Text prompt model not set. Call /set-text-prompt first."}), 400

    data = request.get_json()
    if "image_base64" not in data:
        return jsonify({"error": "Missing 'image_base64' field."}), 400

    try:
        frame_np = read_image_from_base64(data["image_base64"])
        result = predict_text(text_model, frame_np, text_class_names)
        result_base64 = encode_image_to_base64(result)

        return jsonify({"image_base64": result_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stream-frame-engine', methods=['POST'])
def stream_frame_engine():

    data = request.get_json()
    if "image_base64" not in data:
        return jsonify({"error": "Missing 'image_base64' field."}), 400

    try:
        frame_np = read_image_from_base64(data["image_base64"])

        results = predict_with_engine_model(frame_np)
        result_base64 = encode_image_to_base64(results)

        return jsonify({"image_base64": result_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stream-frame-pytorch', methods=['POST'])
def stream_frame_pytorch():
    try:
        data = request.get_json()
        if "image_base64" not in data:
            return jsonify({"error": "Missing 'image_base64' field."}), 400

        frame_np = read_image_from_base64(data["image_base64"])
        result = predict_with_pt_model(frame_np)
        result_base64 = encode_image_to_base64(result)

        return jsonify({"image_base64": result_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
