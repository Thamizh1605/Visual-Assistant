"""
Customized Offline Indoor VQA Navigation System
 - N = Describe scene in front of camera
 - W = Start recording a voice question (also captures camera frame)
 - E = Stop recording, process audio + image, generate answer, speak
"""

import cv2
import sounddevice as sd
import numpy as np
from pathlib import Path
import tempfile
import queue
import torch
from PIL import Image

# HuggingFace Transformers
from transformers import (
    ViltProcessor, ViltForQuestionAnswering,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer,
    WhisperProcessor, WhisperForConditionalGeneration
)

import pyttsx3
from scipy.io.wavfile import write

# ------------------ Config ------------------
CAMERA_INDEX = 0
AUDIO_SAMPLE_RATE = 16000
OUTPUT_VOICE_FILE = "answer.mp3"

# ------------------ Audio recording (async) ------------------
recording = False
audio_q = queue.Queue()
recorded_audio = []

def audio_callback(indata, frames, time, status):
    if recording:
        audio_q.put(indata.copy())

def start_recording():
    global recording, recorded_audio
    recorded_audio = []
    recording = True
    stream = sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype="int16", callback=audio_callback)
    stream.start()
    return stream

def stop_recording(stream):
    global recording, recorded_audio
    recording = False
    stream.stop()
    stream.close()

    # collect all chunks
    while not audio_q.empty():
        recorded_audio.append(audio_q.get())

    if len(recorded_audio) == 0:
        return None

    audio_data = np.concatenate(recorded_audio, axis=0)

    tmp_wav = Path(tempfile.mktemp(suffix=".wav"))
    write(tmp_wav, AUDIO_SAMPLE_RATE, audio_data)
    return tmp_wav

# ------------------ Load Pretrained Models ------------------
# VQA Model
vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
vqa_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Image Captioning Model (scene description)
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Whisper (speech-to-text)
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# TTS (offline)
engine = pyttsx3.init()
engine.setProperty("rate", 160)

# ------------------ STT ------------------
def transcribe_audio(file_path: Path):
    print("[STT] Transcribing with Whisper...")
    import librosa
    speech_array, sr = librosa.load(file_path, sr=AUDIO_SAMPLE_RATE)
    inputs = whisper_processor(speech_array, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = whisper_model.generate(**inputs)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# ------------------ VQA ------------------
def ask_vqa(image_bgr, question: str):
    print("[VQA] Answering with ViLT...")
    image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    inputs = vqa_processor(image, question, return_tensors="pt")
    with torch.no_grad():
        outputs = vqa_model(**inputs)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
    return vqa_model.config.id2label[idx]

# ------------------ Scene Description ------------------
def describe_scene(image_bgr):
    print("[Describe] Generating caption...")
    image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    pixel_values = caption_feature_extractor(images=image, return_tensors="pt").pixel_values
    output_ids = caption_model.generate(pixel_values, max_length=50, num_beams=4)
    caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption

# ------------------ TTS ------------------
def speak_answer(answer: str):
    print("[TTS] Speaking:", answer)
    engine = pyttsx3.init()   # reinitialize every time
    engine.setProperty("rate", 160)
    engine.say(answer)
    engine.runAndWait()
    engine.stop()


# ------------------ Main Loop ------------------
def main_loop():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    print("[MAIN] Press 'q' to quit | 'N' describe scene | 'W' start recording | 'E' stop & process")

    stream = None
    saved_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('n'):
            desc = describe_scene(frame)
            print("[SCENE]", desc)
            speak_answer(desc)

        elif key == ord('w'):
            print("[REC] Starting recording...")
            saved_frame = frame.copy()
            stream = start_recording()

        elif key == ord('e'):
            print("[REC] Stopping recording...")
            if stream:
                wav_file = stop_recording(stream)
                if wav_file:
                    question = transcribe_audio(wav_file)
                    print("[QUESTION]", question)
                    answer = ask_vqa(saved_frame, question)
                    print("[ANSWER]", answer)
                    speak_answer(answer)
                else:
                    print("[ERROR] No audio captured.")
                stream = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
