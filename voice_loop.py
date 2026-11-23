"""Voice loop, phase 2: streaming.

    mic -> endpointing (VAD) -> speech-to-text -> LLM -> text-to-speech -> speaker

Still turn-based, but the LLM reply is streamed and spoken sentence by sentence,
so the caller hears the first sentence while the rest is being written.
Every stage is timed.

    python voice_loop.py           talk to it (Ctrl+C to quit)
    python voice_loop.py --check   test the pieces without a mic or an API key
"""
import glob
import os
import re
import sys
import tempfile
import time
import wave
from collections import deque
from contextlib import closing

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # quiet a harmless Windows notice on model download
# pip puts the NVIDIA DLLs inside the venv, where Windows only finds them through PATH.
for _dll_dir in glob.glob(os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin")):
    os.environ["PATH"] = _dll_dir + os.pathsep + os.environ["PATH"]

import anthropic
import ctranslate2
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
MODEL = "claude-haiku-5-5"  # measured: first sentence in ~750 ms, vs ~1,190 ms on claude-opus-5-5
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
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
    # Measured on a GTX 1660, 1.7 s clip: GPU 110 ms, CPU 641 ms. small.en on GPU: 338 ms.
    if ctranslate2.get_cuda_device_count():
        return WhisperModel("base.en", device="cuda", compute_type="float32")
    return WhisperModel("base.en", device="cpu", compute_type="int8")


def transcribe(stt, audio):
    segments, _ = stt.transcribe(audio, language="en", beam_size=1)
    return " ".join(s.text.strip() for s in segments).strip()


def sentences(chunks):
    """Regroup a stream of text chunks into whole sentences, yielding each as soon as it ends.

    ponytail: splits on . ! ? followed by a space, so "Dr. Lee" splits early.
    The pieces are spoken back to back, so the cost is a slightly odd pause.
    """
    buf = ""
    for chunk in chunks:
        *done, buf = SENTENCE_END.split(buf + chunk)
        yield from done
    if buf.strip():
        yield buf.strip()


def ask(llm, history, text):
    """Yield the reply one sentence at a time, while the model is still writing the rest."""
    history.append({"role": "user", "content": text})
    with llm.messages.stream(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "low"},  # least thinking, fastest reply
        messages=history,
    ) as stream:
        yield from sentences(stream.text_stream)
        resp = stream.get_final_message()
    if resp.stop_reason == "refusal":
        history.pop()
        yield "Sorry, I can't help with that."
    else:
        history.append({"role": "assistant", "content": resp.content})  # whole content, not just the text


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
        print("bot:", end=" ", flush=True)
        first = None
        for sentence in ask(llm, history, text):  # speak each sentence as soon as it is written
            first = first or time.perf_counter()
            print(sentence, end=" ", flush=True)
            speak(sentence)
        wait, stt_ms, llm_ms = SILENCE_FRAMES * 30, (t1 - t0) * 1000, ((first or time.perf_counter()) - t1) * 1000
        print(f"\n     [silence wait {wait} ms | stt {stt_ms:.0f} ms | llm first sentence {llm_ms:.0f} ms"
              f" | reply starts {wait + stt_ms + llm_ms:.0f} ms after you stopped]\n")


def check():
    # Endpointing on synthetic audio: a lone click is ignored, speech comes back with its pre-roll.
    quiet, loud = np.zeros(FRAME, np.float32), np.full(FRAME, 0.1, np.float32)
    frames = [quiet] * 20 + [loud] + [quiet] * 40 + [loud] * 20 + [quiet] * 40
    out = capture(iter(frames), MIN_THRESHOLD)
    assert len(out) == (PRE_ROLL + 19 + SILENCE_FRAMES) * FRAME, len(out) / FRAME
    print("endpointing ok")

    # Sentence regrouping: chunk boundaries fall anywhere, sentences come out whole.
    assert list(sentences(["Hi the", "re. How are", " you? Bye"])) == ["Hi there.", "How are you?", "Bye"]
    print("sentence streaming ok")

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
