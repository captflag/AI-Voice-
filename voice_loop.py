"""Phase 1: the naive voice loop.

    mic -> endpointing (VAD) -> speech-to-text -> LLM -> text-to-speech -> speaker

Turn-based and blocking on purpose. Every stage is timed, so you can see where
the delay comes from before phase 2 starts removing it.

    python voice_loop.py           talk to it (Ctrl+C to quit)
    python voice_loop.py --check   test the pieces without a mic or an API key
"""
import os
import sys
import tempfile
import time
import wave
from collections import deque
from contextlib import closing

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # quiet a harmless Windows notice on model download

import anthropic
import numpy as np
import pyttsx3
import sounddevice as sd
from faster_whisper import WhisperModel

RATE = 16000          # Hz, mono. What Whisper expects.
FRAME = 480           # samples per frame = 30 ms
SILENCE_FRAMES = 27   # ~0.8 s of quiet means "the user finished". This wait is pure latency.
PRE_ROLL = 10         # frames kept from before speech started, so word onsets aren't clipped
MIN_LOUD = 5          # fewer loud frames than this is a click or a cough, not speech
MAX_FRAMES = 1000     # 30 s cap on one utterance
MIN_THRESHOLD = 0.01  # loudness floor for "speech". Raise it if noise triggers recordings.
MODEL = "claude-opus-5-5"
SYSTEM = (
    "You are the receptionist at Bright Smile Dental, answering a phone call. "
    "Your words are spoken aloud, so reply in one or two short sentences of plain "
    "speech: no lists, no markdown, no emoji. Never answer with silence: the "
    "caller is still on the line, so always say at least a few words. "
    "You cannot book anything yet; take "
    "the caller's details and say someone will confirm."
)


def rms(frame):
    return float(np.sqrt(np.mean(frame**2)))


def capture(frames, threshold):
    """Endpointing: read 30 ms frames until one utterance is complete, then return it.

    ponytail: a loudness threshold, so loud background noise counts as speech.
    Phase 3 replaces it with a trained VAD (Silero).
    """
    pre, kept, quiet, loud_n = deque(maxlen=PRE_ROLL), [], 0, 0
    for frame in frames:
        loud = rms(frame) > threshold
        if not kept:  # still waiting for speech to start
            pre.append(frame)
            if loud:
                kept, quiet, loud_n = list(pre), 0, 1
            continue
        kept.append(frame)
        loud_n += loud
        quiet = 0 if loud else quiet + 1
        if len(kept) >= MAX_FRAMES or (quiet >= SILENCE_FRAMES and loud_n >= MIN_LOUD):
            return np.concatenate(kept)
        if quiet >= SILENCE_FRAMES:  # too short to be speech: reset and keep waiting
            pre.clear()
            kept = []
    return None


def mic_frames():
    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=FRAME) as mic:
        while True:
            yield mic.read(FRAME)[0][:, 0]


def load_stt():
    # ponytail: small model on CPU. Phase 2 measures GPU and larger models against it.
    return WhisperModel("base.en", device="cpu", compute_type="int8")


def transcribe(stt, audio):
    segments, _ = stt.transcribe(audio, language="en", beam_size=1)
    return " ".join(s.text.strip() for s in segments).strip()


def ask(llm, history, text):
    history.append({"role": "user", "content": text})
    resp = llm.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "low"},  # least thinking, fastest reply
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # if the model declines, the API retries on a fallback model
        messages=history,
    )
    if resp.stop_reason == "refusal":
        history.pop()
        return "Sorry, I can't help with that."
    history.append({"role": "assistant", "content": resp.content})  # whole content, not just the text
    return " ".join(b.text for b in resp.content if b.type == "text")


def speak(text):
    # ponytail: the built-in Windows voice. Robotic, but instant and offline.
    # Phase 2 replaces it with a streaming neural voice.
    tts = pyttsx3.init()  # keep this local: an engine held between calls goes silent after its first one
    tts.say(text)
    tts.runAndWait()


def main():
    llm = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    stt = load_stt()
    history = []

    print("Measuring room noise, stay quiet for a second...")
    with closing(mic_frames()) as frames:
        ambient = float(np.median([rms(next(frames)) for _ in range(30)]))
    threshold = max(MIN_THRESHOLD, 4 * ambient)
    print(f"Room noise {ambient:.4f}, speech threshold {threshold:.4f}. Say something.\n")

    while True:
        with closing(mic_frames()) as frames:
            audio = capture(frames, threshold)
        t0 = time.perf_counter()
        text = transcribe(stt, audio)
        t1 = time.perf_counter()
        if not text:
            continue
        print("you:", text)
        reply = ask(llm, history, text)
        t2 = time.perf_counter()
        print("bot:", reply)
        wait, stt_ms, llm_ms = SILENCE_FRAMES * 30, (t1 - t0) * 1000, (t2 - t1) * 1000
        print(f"     [silence wait {wait} ms | stt {stt_ms:.0f} ms | llm {llm_ms:.0f} ms"
              f" | reply starts {wait + stt_ms + llm_ms:.0f} ms after you stopped]\n")
        speak(reply)


def check():
    # Endpointing on synthetic audio: a lone click is ignored, speech comes back with its pre-roll.
    quiet, loud = np.zeros(FRAME, np.float32), np.full(FRAME, 0.1, np.float32)
    frames = [quiet] * 20 + [loud] + [quiet] * 40 + [loud] * 20 + [quiet] * 40
    out = capture(iter(frames), MIN_THRESHOLD)
    assert len(out) == (PRE_ROLL + 19 + SILENCE_FRAMES) * FRAME, len(out) / FRAME
    print("endpointing ok")

    # Text-to-speech into a file, then speech-to-text back out of it.
    path = os.path.join(tempfile.gettempdir(), "voice_loop_check.wav")
    tts = pyttsx3.init()
    tts.save_to_file("I would like to book an appointment for Tuesday.", path)
    tts.runAndWait()
    with wave.open(path) as w:  # 16-bit mono at the voice's own rate; resample to 16 kHz
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        audio = np.interp(np.arange(0, len(pcm), w.getframerate() / RATE), np.arange(len(pcm)), pcm)
    heard = transcribe(load_stt(), audio.astype(np.float32))
    assert "appointment" in heard.lower(), heard
    print("tts -> stt ok:", heard)


if __name__ == "__main__":
    try:
        check() if "--check" in sys.argv else main()
    except KeyboardInterrupt:
        print("\nbye")
