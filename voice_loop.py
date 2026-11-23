"""Voice loop, phase 2: streaming.

    mic -> endpointing (VAD) -> speech-to-text -> LLM -> text-to-speech -> speaker

Still turn-based, but the LLM reply is streamed sentence by sentence, each
sentence is synthesised as soon as it is written, and a background thread plays
them back to back. The caller hears the first sentence while the rest is still
being written and synthesised. Every stage is timed.

    python voice_loop.py           talk to it (Ctrl+C to quit)
    python voice_loop.py --check   test the pieces without a mic or an API key
"""
import glob
import os
import queue
import re
import sys
import threading
import time
import urllib.request
from collections import deque
from contextlib import closing

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # quiet a harmless Windows notice on model download
# pip puts the NVIDIA DLLs inside the venv, where Windows only finds them through PATH.
for _dll_dir in glob.glob(os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin")):
    os.environ["PATH"] = _dll_dir + os.pathsep + os.environ["PATH"]

import anthropic
import ctranslate2
import numpy as np
import onnxruntime as ort
import sounddevice as sd
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

RATE = 16000          # Hz, mono. What Whisper expects.
FRAME = 480           # samples per frame = 30 ms
SILENCE_FRAMES = 27   # ~0.8 s of quiet means "the user finished". This wait is pure latency.
PRE_ROLL = 10         # frames kept from before speech started, so word onsets aren't clipped
MIN_LOUD = 5          # fewer loud frames than this is a click or a cough, not speech
MAX_FRAMES = 1000     # 30 s cap on one utterance
MIN_THRESHOLD = 0.01  # loudness floor for "speech". Raise it if noise triggers recordings.
MODEL = "claude-haiku-5-5"  # measured: first sentence in ~750 ms, vs ~1,190 ms on claude-opus-5-5
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
VOICE = "af_sarah"    # tts.get_voices() lists the others
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
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


def load_tts():
    """Kokoro, a small neural voice. Measured on a GTX 1660: ~250 ms for a 3 s sentence, ~1,270 ms on CPU."""
    paths = [os.path.join(MODELS_DIR, name) for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin")]
    for path in paths:
        if not os.path.exists(path):  # first run only, about 350 MB in total
            print(f"Downloading {os.path.basename(path)}...")
            os.makedirs(MODELS_DIR, exist_ok=True)
            urllib.request.urlretrieve(f"{KOKORO_URL}/{os.path.basename(path)}", path + ".part")
            os.replace(path + ".part", path)  # so a half-finished download never looks complete
    ort.set_default_logger_severity(3)  # errors only
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        ort.preload_dlls()  # load the CUDA libraries pip put in the venv
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    tts = Kokoro.from_session(ort.InferenceSession(paths[0], providers=providers), paths[1])
    tts.create("Warming up.", voice=VOICE)  # the first synthesis is several times slower than the rest
    return tts


def play_queue(speech):
    """Play queued clips back to back, so the next sentence can be synthesised while this one plays."""
    while True:
        samples, rate = speech.get()
        try:
            sd.play(samples, rate, blocking=True)
        except Exception as e:  # keep the thread alive, or main() would wait on the queue forever
            print("\nplayback failed:", e)
        finally:
            speech.task_done()


def main():
    llm = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    stt, tts = load_stt(), load_tts()
    history, speech = [], queue.Queue()
    threading.Thread(target=play_queue, args=(speech,), daemon=True).start()

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
        t_text = t_audio = None
        for sentence in ask(llm, history, text):
            t_text = t_text or time.perf_counter()
            print(sentence, end=" ", flush=True)
            speech.put(tts.create(sentence, voice=VOICE))  # playback of earlier sentences continues meanwhile
            t_audio = t_audio or time.perf_counter()
        speech.join()  # finish talking before listening again, or the mic hears the bot
        if not t_audio:
            print("(no reply)\n")
            continue
        wait, stt_ms, llm_ms, tts_ms = SILENCE_FRAMES * 30, (t1 - t0) * 1000, (t_text - t1) * 1000, (t_audio - t_text) * 1000
        print(f"\n     [silence wait {wait} ms | stt {stt_ms:.0f} ms | llm first sentence {llm_ms:.0f} ms | tts {tts_ms:.0f} ms"
              f" | reply starts {wait + stt_ms + llm_ms + tts_ms:.0f} ms after you stopped]\n")


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

    # The neural voice says a sentence, speech-to-text reads it back.
    samples, rate = load_tts().create("I would like to book an appointment for Tuesday.", voice=VOICE)
    audio = np.interp(np.arange(0, len(samples), rate / RATE), np.arange(len(samples)), samples)  # 24 kHz -> 16 kHz
    heard = transcribe(load_stt(), audio.astype(np.float32))
    assert "appointment" in heard.lower(), heard
    print("tts -> stt ok:", heard)


if __name__ == "__main__":
    try:
        check() if "--check" in sys.argv else main()
    except KeyboardInterrupt:
        print("\nbye")
