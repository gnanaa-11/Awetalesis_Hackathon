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
    vad = webrtcvad.Vad(1)

    input("Press ENTER to start recording...")
    print("Recording... Press ENTER again to stop.")

    frame_duration = 30  # ms
    frame_size = int(fs * frame_duration / 1000)

    audio_frames = []
    running = True

    def stop_recording():
        nonlocal running
        input()
        running = False

    speech_detected = False

    def callback(indata, frames, time, status):
        nonlocal running, speech_detected

        if not running:
            raise sd.CallbackStop()

        audio_int16 = (indata.flatten() * 32768).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        if vad.is_speech(audio_bytes, fs):
            speech_detected = True
            audio_frames.append(audio_bytes)
        elif speech_detected:

            audio_frames.append(audio_bytes)


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

    print("Recording stopped.")


    if not audio_frames:
        print("No speech detected, recording raw audio")

        audio = sd.rec(int(5 * fs), samplerate=fs, channels=1)
        sd.wait()
        write(filename, fs, audio)
        return filename
    audio_np = np.frombuffer(b"".join(audio_frames), dtype=np.int16).astype(np.float32) / 32768.0


    audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-6)

    write(filename, fs, audio_np)
    return filename

       



model = whisper.load_model("small")


audio_file = record_audio()


result = model.transcribe(audio_file, fp16=False, language="en")
text = result["text"].strip()

print("English:", text)



translated = GoogleTranslator(
    source="en",
    target="te"
).translate(text)

print("Telugu:", translated)


with open("output.txt", "w", encoding="utf-8") as f:
    f.write(translated)


print("Audio Speech...")

try:
    tts = gTTS(text=translated, lang='te')
    tts.save("output.mp3")
    os.system("ffplay -autoexit -loglevel quiet output.mp3")
except Exception as e:
    print("TTS failed:", e)
