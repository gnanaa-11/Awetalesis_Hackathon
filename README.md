#  Speech-to-Speech Translation System

##  Overview

This project implements an end-to-end **speech-to-speech translation pipeline** that captures spoken English input, converts it into text, translates it into Telugu, and generates corresponding speech output.

The system focuses on **reliability and clarity** by incorporating speech filtering and structured processing instead of attempting unstable real-time streaming.

---

## Features

*  Live audio recording from microphone
*  Voice Activity Detection (VAD) using WebRTC
*  Speech-to-Text using Whisper
*  English → Telugu translation
*  Text-to-Speech output generation
*  Output saved as text and audio



## System Pipeline

**Pipeline:**

Microphone Input → Audio Frames (30 ms chunks) → Voice Activity Detection (WebRTC VAD) → Speech Buffering → Audio Normalization → Speech-to-Text (Whisper) → Text Translation (English → Telugu) → Text-to-Speech (TTS) → Audio Playback + File Output


---

## Key Design 

* Used **WebRTC VAD** to filter non-speech noise and improve transcription quality
* Chose **batch processing** instead of real-time streaming for stability
* Normalized audio before transcription to improve Whisper accuracy
* Structured the pipeline into clear stages for maintainability

---

##  Challenges Faced

* Handling noise and silence in raw audio input
* Ensuring Whisper receives clean and meaningful audio segments
* Managing VAD sensitivity to avoid cutting important speech
* Balancing simplicity vs real-time complexity

---

##  Limitations

* Not real-time (processes audio after recording ends)
* Translation and TTS depend on internet connectivity
* VAD may occasionally miss low-volume speech
* No UI (command-line based interaction)



## Tech Stack

* Python
* Whisper (ASR)
* WebRTC VAD
* SoundDevice (audio capture)
* Deep Translator
* gTTS


## How to Run

1. Install dependencies:

   pip install whisper sounddevice numpy scipy deep-translator gTTS webrtcvad

2. Run the script:

   python main.py

3. Press ENTER to start recording and again to stop.

---

##  Conclusion

This project demonstrates a structured approach to building a **end-to-end speech processing pipeline**, focusing on correctness, modularity, and practical implementation over premature optimization.
