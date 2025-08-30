import cv2
import os
import sys
import queue
import time
import json
import threading
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ---- If you already have answer_question in ass3_image_api, keep this import ----
# It should implement: answer_question(image_path: str, question: str) -> str
from image_qa import answer_question
# -----------------------------------------------------------------------------

# -------------------- CONFIG --------------------
# Use absolute path so launching from any folder works:
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path("/Users/thamizharasan/Desktop/Final Project/ViLT/vosk-model-en-us-0.22")
WAKE_WORDS = {"hello"}                      # say "hello" to snap a picture
SHUTDOWN_WORDS = {"shutdown", "shut down"}  # end program
OUTPUT_DIR = SCRIPT_DIR / "captures"        # where to save images
SHOW_PREVIEW = True                         # show a preview window (press 'q' to quit)
SAMPLE_RATE = 16000                         # 16000 usually works; adjust if your mic needs it
PREFERRED_CAM_INDICES = (0, 1, 2, 3)        # try these indices in order (mac external cams often 1/2)
# ------------------------------------------------

# --------------- Utilities ---------------

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def speak(text: str):
    """Speak text. macOS uses 'say'; others try pyttsx3 if available."""
    try:
        if platform.system() == "Darwin":
            # macOS built-in TTS
            subprocess.run(["say", text], check=False)
        else:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception:
                # As a last resort, just print
                print(f"[TTS unavailable] {text}")
    except Exception:
        print(f"[TTS error] {text}")

def is_question(text: str) -> bool:
    """
    Heuristic: Vosk output has no punctuation.
    Treat as a question if starts with typical question forms
    or contains common polite question patterns.
    """
    t = text.strip().lower()
    if not t:
        return False

    # Quick exits
    if t in WAKE_WORDS or t in SHUTDOWN_WORDS:
        return False

    # Common question starters (expand as needed)
    starters = (
        "what", "who", "where", "when", "why", "how", "which",
        "whom", "whose", "can", "could", "will", "would",
        "is", "are", "am", "do", "does", "did", "should", "shall", "may", "might"
    )
    polite_patterns = ("can you", "could you", "would you", "please tell me", "i want to know")

    return t.startswith(starters) or any(p in t for p in polite_patterns)

def capture_frame_to_file(frame) -> str:
    fname = f"capture.jpg"
    fpath = OUTPUT_DIR / fname
    ok = cv2.imwrite(str(fpath), frame)
    if ok:
        print(f"✅ Captured")
        return str(fpath)
    else:
        print("❌ Failed to save image.")
        return ""

# --------------- Camera open (mac friendly) ---------------

def open_camera_mac(preferred_indices=PREFERRED_CAM_INDICES):
    for i in preferred_indices:
        cam = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cam.isOpened():
            ok, _ = cam.read()
            if ok:
                print(f"[OK] Opened camera index {i} with AVFoundation")
                return cam
        if cam is not None:
            cam.release()
    for i in preferred_indices:
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            ok, _ = cam.read()
            if ok:
                print(f"[OK] Opened camera index {i} with default backend")
                return cam
        if cam is not None:
            cam.release()
    return None

def open_camera_cross_platform(preferred_indices=PREFERRED_CAM_INDICES):
    if platform.system() == "Darwin":
        return open_camera_mac(preferred_indices)
    for i in preferred_indices:
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            ok, _ = cam.read()
            if ok:
                print(f"[OK] Opened camera index {i}")
                return cam
        if cam is not None:
            cam.release()
    return None

# --------------- Speech thread ---------------

def load_stt_model():
    if not MODEL_PATH.is_dir():
        print(f"Vosk model folder not found:\n  {MODEL_PATH}")
        sys.exit(1)
    print("Loading Vosk model (first time can take a few seconds)...")
    model = Model(str(MODEL_PATH))
    print("Vosk model loaded.")
    return model

def audio_listener(commands_q: queue.Queue, stop_event: threading.Event):
    """
    Listens continuously. Pushes events into commands_q:
      {"type": "hello"}
      {"type": "question", "text": "..."}
      {"type": "shutdown"}
    """
    q_audio = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            # print(status, file=sys.stderr)
            pass
        q_audio.put(bytes(indata))

    model = load_stt_model()
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(False)

    # Audio stream
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        print("🎤 Listening… Say 'hello' to snap, ask a question, or say 'shutdown' to exit.")
        while not stop_event.is_set():
            try:
                data = q_audio.get(timeout=0.1)
            except queue.Empty:
                continue

            got_final = recognizer.AcceptWaveform(data)
            # We’ll act only on final results to avoid duplicate triggers
            if got_final:
                try:
                    result = json.loads(recognizer.Result())
                except json.JSONDecodeError:
                    continue
                text = (result.get("text") or "").strip().lower()
                if not text:
                    continue
                # print(f"[heard] {text}")  # uncomment for debugging

                # 1) Shutdown
                if any(w in text.split() for w in SHUTDOWN_WORDS) or "shut down" in text:
                    commands_q.put({"type": "shutdown"})
                    # Don’t return immediately; let main handle cleanup
                    continue

                # 2) Hello (image capture)
                if any(w in text.split() for w in WAKE_WORDS):
                    commands_q.put({"type": "hello"})
                    # If the utterance also looks like a question, let it fall through

                # 3) Question
                if is_question(text):
                    commands_q.put({"type": "question", "text": text})

            else:
                # If you want faster hello reaction, you could parse partials:
                # partial = json.loads(recognizer.PartialResult()).get("partial", "").lower()
                # if any(w in partial.split() for w in WAKE_WORDS):
                #     commands_q.put({"type": "hello"})
                pass

# --------------- Main ---------------

def main():
    ensure_dirs()

    # Inter-thread communication
    stop_event = threading.Event()
    commands_q: queue.Queue = queue.Queue()

    # Start listener thread
    listener_thread = threading.Thread(
        target=audio_listener, args=(commands_q, stop_event), daemon=True
    )
    listener_thread.start()

    # Open camera
    cap = open_camera_cross_platform()
    if not cap or not cap.isOpened():
        print("❌ Could not open any camera. On macOS, check System Settings → Privacy & Security → Camera,")
        print("   and ensure Terminal/your IDE is allowed. Close other apps using the camera and try again.")
        stop_event.set()
        listener_thread.join(timeout=2)
        return

    print("📷 Camera started. Say 'HELLO' to take a picture, ask a question anytime, or say 'SHUTDOWN' to exit.")
    print("Press 'q' in the preview window (if enabled) or Ctrl+C in the terminal to quit.")

    try:
        last_frame = None
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️ Failed to read from camera.")
                time.sleep(0.05)
                continue
            last_frame = frame

            # Handle any pending voice commands
            try:
                while True:
                    cmd = commands_q.get_nowait()
                    ctype = cmd.get("type")

                    if ctype == "shutdown":
                        speak("Shutting down. Goodbye.")
                        raise KeyboardInterrupt  # break to finally-block

                    elif ctype == "hello":
                        if last_frame is not None:
                            capture_frame_to_file(last_frame)

                    elif ctype == "question":
                        q_text = cmd.get("text", "")
                        if not q_text:
                            continue
                        # Always capture a fresh image for the question
                        img_path = capture_frame_to_file(last_frame) if last_frame is not None else ""
                        if not img_path:
                            speak("I could not capture an image.")
                            continue
                        # Call your VQA function and speak the answer
                        try:
                            answer = answer_question(img_path, q_text)  # <- your function
                        except Exception as e:
                            print(f"[answer_question error] {e}")
                            answer = "I ran into a problem answering that."
                        print(f"Q: {q_text}\nA: {answer}")
                        speak(answer)

                    # Continue draining the queue this frame
            except queue.Empty:
                pass

            # Preview window
            if SHOW_PREVIEW:
                cv2.putText(
                    frame,
                    "Say 'HELLO' to capture; ask a question; say 'SHUTDOWN' to quit. Press 'q' to quit.",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                cv2.imshow("Always-On Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stop_event.set()
        listener_thread.join(timeout=2)
        if cap:
            cap.release()
        if SHOW_PREVIEW:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        print("👋 Exited cleanly.")

if __name__ == "__main__":
    main()
