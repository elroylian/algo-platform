from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transcriber import (
    speech_to_text_eng,
    speech_to_text_chi,
    trans_eng_to_chi,
    trans_chi_to_eng,
    translate_text_zh_to_en,
    generate_tts_bytes
)
import os
import uuid
import io

app = Flask(__name__)
CORS(app)

def get_temp_path(extension=".m4a"):
    return f"temp_{uuid.uuid4().hex}{extension}"

# 🎙️ English Speech → English Text
@app.route("/speech-to-text-en", methods=["POST"])
def speech_to_text_en():
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "Missing audio file"}), 400

    temp_path = get_temp_path()
    audio.save(temp_path)

    try:
        text = speech_to_text_eng(temp_path)
        return jsonify({"text": text})
    finally:
        os.remove(temp_path)

# 🎤 Chinese Speech → English Text
@app.route("/speech-to-text-zh", methods=["POST"])
def speech_to_text_zh():
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "Missing audio file"}), 400

    temp_path = get_temp_path()
    audio.save(temp_path)

    try:
        text = speech_to_text_chi(temp_path)
        return jsonify({"text": text})
    finally:
        os.remove(temp_path)

# 🧠 English Speech → Chinese Text
@app.route("/speech-en-to-zh", methods=["POST"])
def speech_en_to_zh():
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "Missing audio file"}), 400

    temp_path = get_temp_path()
    audio.save(temp_path)

    try:
        translated = trans_eng_to_chi(temp_path)
        return jsonify({"text": translated})
    finally:
        os.remove(temp_path)

# 🧠 Chinese Speech → English Text
@app.route("/speech-zh-to-en", methods=["POST"])
def speech_zh_to_en():
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "Missing audio file"}), 400

    temp_path = get_temp_path()
    audio.save(temp_path)

    try:
        translated = trans_chi_to_eng(temp_path)
        return jsonify({"text": translated})
    finally:
        os.remove(temp_path)

# 🔊 Text (EN or ZH) → TTS
@app.route("/text-to-speech", methods=["POST"])
def text_to_speech():
    data = request.json
    text = data.get("text")
    lang = data.get("lang", "en")

    if not text:
        return jsonify({"error": "Missing text"}), 400

    audio_io = generate_tts_bytes(text, lang=lang)
    return send_file(audio_io, mimetype="audio/mpeg", as_attachment=False)

# 🌐 Chinese Text → English Text
@app.route("/translate-zh-to-en", methods=["POST"])
def translate_zh_to_en():
    data = request.get_json()
    zh_text = data.get("text")

    if not zh_text:
        return jsonify({"error": "Missing text"}), 400

    translated = translate_text_zh_to_en(zh_text)
    return jsonify({"text": translated})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
