# 🖼️ Visual Prompt Detection API (YOLOE)

This module uses **Ultralytics YOLOE** with **visual and text prompts** to perform real-time object detection via webcam or uploaded frames. It supports both prompt-based detection modes via a Flask API.

---

## 🧠 Features

- Visual Prompt: Upload source image + bounding boxes
- Text Prompt: Provide class labels only
- Real-time webcam detection (~30 FPS)
- Base64-based image frame streaming

---

## 📦 Installation

```bash
# YOLOE core setup
git clone https://github.com/THU-MIG/yoloe
cd yoloe
pip install -r requirements.txt
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt

# Visual API dependencies
pip install flask flask-cors opencv-python ultralytics numpy
```

> ⚠️ PyTorch is also required.  
> Visit https://pytorch.org/get-started/locally/ and install based on your OS and environment.

---

## 🚀 Usage

```bash
python3 -m http.server 8080
python3 vision_api.py
```

Then open your browser at:  
👉 http://localhost:8080

Models tested:
- [YOLOe-v8-S](https://huggingface.co/jameslahm/yoloe/blob/main/yoloe-v8s-seg.pt)
- [YOLO-v11-S](https://huggingface.co/jameslahm/yoloe/blob/main/yoloe-11s-seg.pt)

---

## 📡 API Endpoints

### `POST /set-prompt`
Set visual prompt with image and bounding boxes.

**Form Data:**
- `source_image`: Image file (used for visual prompting)
- `bboxes`: JSON array of bounding boxes (e.g. `[[x1, y1, x2, y2], ...]`)
- `classes`: JSON array of class labels (e.g. `["chair", "orb"]`)

---

### `POST /set-text-prompt`
Set prompt with only class names.

**Request JSON:**
```json
{
  "classes": ["person", "chair"]
}
```

---

### `POST /stream-frame`
Stream webcam frame (base64 JPEG) and get visual-prompt-based detection result.

**Request JSON:**
```json
{
  "image_base64": "<base64-encoded image>"
}
```

---

### `POST /stream-frame-text`
Same as `/stream-frame`, but uses the text-prompt model instead.

---

## 🖼️ Frontend Interface

Hosted at `http://localhost:8080` via `index.html`:

- Select mode: `Visual Prompt` or `Text Prompt`
- Upload image + bounding boxes or class list
- Starts webcam + streams annotated results
