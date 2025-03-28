import torch
import torchaudio
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline
)
from gtts import gTTS
import io

# Detect device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Constants
MODEL_NAME = "openai/whisper-large-v3"

# Load Whisper model + processor
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Load ASR pipeline
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    chunk_length_s=30,
    device=device,
)

# Load translation models (cached once)
en2zh_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
en2zh_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh").to(device)

zh2en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
zh2en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en").to(device)

# 🎙 English Speech → English Text
def speech_to_text_eng(audio_file_path, batch_size=8):
    return pipe(
        audio_file_path,
        batch_size=batch_size,
        generate_kwargs={"task": "transcribe", "language": "english"},
        return_timestamps=True
    )["text"]

# 🎙 Chinese Speech → English Text
def speech_to_text_chi(audio_file_path, batch_size=8):
    return pipe(
        audio_file_path,
        batch_size=batch_size,
        generate_kwargs={"task": "translate", "language": "chinese"},
        return_timestamps=True
    )["text"]

# 🧠 English Speech → Chinese Text
def trans_eng_to_chi(audio_file_path):
    transcription = speech_to_text_eng(audio_file_path)
    return translate_text_en_to_zh(transcription)

# 🧠 Chinese Speech → English Text
def trans_chi_to_eng(audio_file_path, batch_size=8):
    return pipe(
        audio_file_path,
        batch_size=batch_size,
        generate_kwargs={"task": "translate", "language": "chinese"},
        return_timestamps=True
    )["text"]

# 🌐 English Text → Chinese Text
def translate_text_en_to_zh(text):
    chunks = [text[i:i+512] for i in range(0, len(text), 512)]
    translated = []
    for chunk in chunks:
        inputs = en2zh_tokenizer(chunk, return_tensors="pt").to(device)
        outputs = en2zh_model.generate(**inputs, max_length=512, num_beams=4)
        translated.append(en2zh_tokenizer.decode(outputs[0], skip_special_tokens=True))
    return "".join(translated)

# 🌐 Chinese Text → English Text
def translate_text_zh_to_en(text):
    inputs = zh2en_tokenizer(text, return_tensors="pt").to(device)
    output = zh2en_model.generate(**inputs)
    return zh2en_tokenizer.decode(output[0], skip_special_tokens=True)

# 🔊 Text (EN/ZH) → Speech
def generate_tts_bytes(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes
