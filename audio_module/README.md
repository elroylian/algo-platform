# 🎙️ Speech Translation & TTS API (Whisper + gTTS + Helsinki-NLP)

This module provides an audio-to-text and text-to-audio interface powered by **Whisper**, **gTTS**, and **Helsinki-NLP**. It supports English and Chinese speech translation using Flask API.

---

## 🧠 Features

- English and Chinese speech-to-text (ASR)
- English ↔ Chinese speech translation
- English/Chinese text-to-speech (TTS)
- Chinese text → English text translation

---

## 📦 Installation

```bash
pip install flask flask-cors transformers gTTS
```

> ⚠️ PyTorch is also required.  
> Visit https://pytorch.org/get-started/locally/ and install based on your OS and environment.

---

## 🚀 Usage

```bash
python3 -m http.server 8080
python speech_api.py
```

Then open your browser at:  
👉 http://localhost:8080

---

## 📡 API Endpoints

### `POST /speech-to-text-en`
- **Input:** English audio (file)
- **Output:** Transcribed English text

---

### `POST /speech-to-text-zh`
- **Input:** Chinese audio (file)
- **Output:** Translated English text

---

### `POST /speech-en-to-zh`
- **Input:** English audio (file)
- **Output:** Translated Chinese text

---

### `POST /speech-zh-to-en`
- **Input:** Chinese audio (file)
- **Output:** Translated English text

---

### `POST /translate-zh-to-en`
- **Input:** JSON with Chinese text
```json
{
  "text": "你好世界"
}
```
- **Output:** Translated English text

---

### `POST /text-to-speech`
- **Input:** JSON with text and language
```json
{
  "text": "Hello, world!",
  "lang": "en"
}
```
- **Output:** MP3 audio stream
