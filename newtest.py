import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import webrtcvad
import collections
import threading

def record_audio(filename="input.wav", fs=16000):
    vad = webrtcvad.Vad(2)

    input("👉 Press ENTER to start recording...")
    print("🎤 Recording... Press ENTER again to stop.")

    frame_duration = 30  # ms
    frame_size = int(fs * frame_duration / 1000)

    audio_frames = []
    running = True

    def stop_recording():
        nonlocal running
        input()
        running = False

    def callback(indata, frames, time, status):
        nonlocal running

        if not running:
            raise sd.CallbackStop()

        audio_int16 = (indata.flatten() * 32768).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # VAD check
        if vad.is_speech(audio_bytes, fs):
            audio_frames.append(audio_bytes)

    # thread to listen for ENTER
    stopper = threading.Thread(target=stop_recording)
    stopper.start()

    try:
        with sd.InputStream(
            samplerate=fs,
            channels=1,
            dtype='float32',
            blocksize=frame_size,
            callback=callback
        ):
            while running:
                pass
    except sd.CallbackStop:
        pass

    print("⏹️ Recording stopped.")

    # convert to numpy
    if not audio_frames:
        print("⚠️ No speech detected")
        return filename

    audio_np = np.frombuffer(b"".join(audio_frames), dtype=np.int16).astype(np.float32) / 32768.0

    # normalize
    audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-6)

    write(filename, fs, audio_np)
    return filename

       


# ==============================
# LOAD MODELS
# ==============================
model = whisper.load_model("small")

# ==============================
# MAIN PIPELINE
# ==============================
audio_file = record_audio()

print("🔍 Transcribing...")
result = model.transcribe(audio_file, fp16=False, language="en")
text = result["text"].strip()

print("📝 English:", text)

# ==============================
# TRANSLATION (stable)
# ==============================
translated = GoogleTranslator(
    source="en",
    target="te"
).translate(text)

print("🌍 Telugu:", translated)

# ==============================
# SAVE TEXT
# ==============================
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(translated)

# ==============================
# TTS
# ==============================
print("🔊 Speaking...")

try:
    tts = gTTS(text=translated, lang='te')
    tts.save("output.mp3")
    os.system("ffplay -autoexit -loglevel quiet output.mp3")
except Exception as e:
    print("TTS failed:", e)
