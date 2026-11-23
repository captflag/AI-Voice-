"""Voice loop, phase 3: turn-taking.

    mic -> Silero VAD -> speech-to-text -> LLM -> text-to-speech -> speaker

The mic never closes, and replies are produced on a worker thread and can be
cancelled. Two behaviours follow from that:

  * Interruptions. If the caller starts talking while the bot is thinking or
    speaking, the reply is dropped and the speaker stops at once.
  * Eager end of turn. The bot starts answering after a short pause (~0.4 s).
    If the caller was only pausing and carries on, the reply nobody heard is
    thrown away and the whole utterance is answered when they really finish.

    python voice_loop.py           talk to it (Ctrl+C to quit)
    python voice_loop.py --check   test the pieces without a mic or an API key

Use a headset. Through open speakers the mic hears the bot and it interrupts
itself; set BARGE_IN = False in that case.
"""
import glob
import os
import queue
import re
import socket
import sys
import threading
import time
import urllib.request
from collections import deque

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
from faster_whisper.vad import get_assets_path
from kokoro_onnx import Kokoro

RATE = 16000          # Hz, mono. What Whisper and Silero expect.
FRAME = 512           # samples per frame = 32 ms, the frame size Silero VAD is trained on
VAD_THRESHOLD = 0.5   # speech probability above this counts as speech
START_FRAMES = 4      # ~130 ms of unbroken speech before it counts. Ignores clicks; also the barge-in delay.
SILENCE_FRAMES = 13   # ~420 ms of quiet starts a reply. Short is safe now: a wrong guess is cancelled, not heard.
PRE_ROLL = 10         # frames kept from before speech started, so word onsets aren't clipped
BARGE_IN = True       # False = ignore the mic while the bot is speaking (open speakers, no echo cancellation)
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

play_lock = threading.Lock()  # makes "check for cancel, then start playing" one step


class Vad:
    """Silero VAD, one 32 ms frame at a time: the probability that the frame contains speech.

    ponytail: uses the model file bundled with faster-whisper, so its name and
    inputs are tied to that package's version.
    """

    def __init__(self):
        path = os.path.join(get_assets_path(), "silero_vad_v6.onnx")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1  # one core is plenty; the default grabs them all
        self.session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        self.h = self.c = np.zeros((1, 1, 128), np.float32)  # the model's memory of earlier frames
        self.context = np.zeros(64, np.float32)              # tail of the previous frame

    def __call__(self, frame):
        x = np.concatenate([self.context, frame])[None]
        probs, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.context = frame[-64:]
        return float(probs.ravel()[0])


class Endpointer:
    """Turns a stream of speech / not-speech frames into "start" and "end" events."""

    def __init__(self):
        self.speaking, self.run, self.quiet = False, 0, 0

    def update(self, is_speech):
        self.run = self.run + 1 if is_speech else 0
        self.quiet = 0 if is_speech else self.quiet + 1
        if not self.speaking and self.run >= START_FRAMES:
            self.speaking = True
            return "start"
        if self.speaking and self.quiet >= SILENCE_FRAMES:
            self.speaking = False
            return "end"
        return None


class Turn:
    """One attempt at answering the caller."""

    def __init__(self, stopped_at):
        self.stopped_at = stopped_at     # when the caller stopped talking
        self.cancel = threading.Event()  # set when the caller talks over this attempt
        self.spoken = []                 # sentences that started playing, i.e. what the caller heard
        self.first_audio = None          # when the first of them started
        self.consumed = False            # True once the caller's audio has been answered; until then it is kept


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
    """Play queued clips back to back, skipping any whose turn has been cancelled."""
    while True:
        samples, rate, sentence, turn = speech.get()
        try:
            with play_lock:
                if turn.cancel.is_set():
                    continue
                turn.first_audio = turn.first_audio or time.perf_counter()
                turn.consumed = True
                turn.spoken.append(sentence)
                sd.play(samples, rate)
            sd.wait()  # returns early if interrupt() stops the speaker
        except Exception as e:  # keep the thread alive, or respond() would wait on the queue forever
            print("\nplayback failed:", e)
        finally:
            speech.task_done()


def interrupt(turn):
    """The caller is talking: drop the reply in progress and silence the speaker."""
    with play_lock:
        turn.cancel.set()
        sd.stop()


def respond(turn, audio, stt, tts, llm, history, speech):
    """Answer one utterance, giving up quietly if the turn is cancelled."""
    if turn.cancel.is_set():
        return
    t0 = time.perf_counter()
    text = transcribe(stt, audio)
    t1 = time.perf_counter()
    if not text:
        turn.consumed = True  # noise, not words: nothing to keep
        return
    if turn.cancel.is_set():
        return
    print("you:", text)
    mark, t_text, tts_ms = len(history), None, None
    reply = ask(llm, history, text)
    for sentence in reply:
        t_text = t_text or time.perf_counter()
        if turn.cancel.is_set():
            break
        t = time.perf_counter()
        samples, rate = tts.create(sentence, voice=VOICE)
        tts_ms = tts_ms or (time.perf_counter() - t) * 1000
        speech.put((samples, rate, sentence, turn))
    reply.close()  # if we left early, this ends the LLM stream
    speech.join()  # until everything queued has been played, or skipped after an interruption

    heard = " ".join(turn.spoken)
    if turn.cancel.is_set():
        del history[mark:]  # forget this attempt...
        if heard:           # ...except the part the caller actually heard
            history += [{"role": "user", "content": text}, {"role": "assistant", "content": heard}]
        print("bot:", f"{heard} [interrupted]" if heard else "[dropped: the caller kept talking]")
    else:
        turn.consumed = True
        print("bot:", heard)
    if turn.first_audio:
        print(f"     [stt {(t1 - t0) * 1000:.0f} ms | llm first sentence {(t_text - t1) * 1000:.0f} ms | tts {tts_ms:.0f} ms"
              f" | reply started {(turn.first_audio - turn.stopped_at) * 1000:.0f} ms after you stopped]")
    print()


def work(jobs, stt, tts, llm, history, speech):
    """The one thread that runs the models, so replies are produced strictly one after another.

    Measured on a GTX 1660: a fresh thread per reply made synthesis ~130 ms slower,
    and after ~4 s with no work the GPU slows down and speech-to-text takes 400 ms
    instead of 150. A tiny transcription every half second while idle prevents that.
    """
    idle_clip = np.zeros(RATE // 10, np.float32)
    while True:
        try:
            turn, audio = jobs.get(timeout=0.5)
        except queue.Empty:
            transcribe(stt, idle_clip)
            continue
        try:
            respond(turn, audio, stt, tts, llm, history, speech)
        except Exception as e:  # e.g. a network error: say so and keep the call alive, don't die silently
            print("reply failed:", e)


def run(frames, stt, tts, llm):
    """The listening loop. Never blocks on anything but the next mic frame."""
    vad, endpointer, pre, utterance = Vad(), Endpointer(), deque(maxlen=PRE_ROLL), []
    history, speech, jobs, turn = [], queue.Queue(), queue.Queue(), None
    threading.Thread(target=play_queue, args=(speech,), daemon=True).start()
    threading.Thread(target=work, args=(jobs, stt, tts, llm, history, speech), daemon=True).start()
    for frame in frames:
        if not BARGE_IN and speech.unfinished_tasks:
            continue  # half-duplex: deaf while the bot is speaking
        event = endpointer.update(vad(frame) > VAD_THRESHOLD)
        if event == "start":
            if turn:
                interrupt(turn)
                if turn.consumed:
                    utterance = []  # the last turn was answered, so this is a new one
            utterance += pre        # otherwise the caller is carrying on: keep what they said so far
            pre.clear()
        (utterance if endpointer.speaking else pre).append(frame)
        if event == "end":
            turn = Turn(time.perf_counter() - SILENCE_FRAMES * FRAME / RATE)
            jobs.put((turn, np.concatenate(utterance)))


def main():
    lock = socket.socket()  # one copy at a time: two would both answer on the same mic
    try:
        lock.bind(("127.0.0.1", 47631))  # any fixed local port; held until this process exits
    except OSError:
        sys.exit("voice_loop.py is already running in another terminal. Stop it there with Ctrl+C first.")
    llm = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    stt, tts = load_stt(), load_tts()
    print("Listening. Say something; talk over the bot to interrupt it.\n")
    run(mic_frames(), stt, tts, llm)


def check():
    # Turn events: a two-frame blip is ignored; speech starts after START_FRAMES and ends after SILENCE_FRAMES.
    flags = [False] * 5 + [True] * 2 + [False] * 5 + [True] * 10 + [False] * 20
    endpointer = Endpointer()
    events = [(i, e) for i, f in enumerate(flags) if (e := endpointer.update(f))]
    assert events == [(12 + START_FRAMES - 1, "start"), (22 + SILENCE_FRAMES - 1, "end")], events
    print("turn events ok")

    # Sentence regrouping: chunk boundaries fall anywhere, sentences come out whole.
    assert list(sentences(["Hi the", "re. How are", " you? Bye"])) == ["Hi there.", "How are you?", "Bye"]
    print("sentence streaming ok")

    # The neural voice says a sentence; the VAD hears speech in it and none in silence.
    samples, rate = load_tts().create("I would like to book an appointment for Tuesday.", voice=VOICE)
    audio = np.interp(np.arange(0, len(samples), rate / RATE), np.arange(len(samples)), samples).astype(np.float32)
    vad = Vad()
    in_speech = np.mean([vad(audio[i:i + FRAME]) > VAD_THRESHOLD for i in range(0, len(audio) - FRAME, FRAME)])
    in_silence = np.mean([vad(np.zeros(FRAME, np.float32)) > VAD_THRESHOLD for _ in range(60)][30:])
    assert in_speech > 0.6 and in_silence == 0, (in_speech, in_silence)
    print(f"vad ok: speech in {in_speech:.0%} of the spoken frames, {in_silence:.0%} of the silent ones")

    # Speech-to-text reads the same clip back.
    heard = transcribe(load_stt(), audio)
    assert "appointment" in heard.lower(), heard
    print("tts -> stt ok:", heard)


if __name__ == "__main__":
    try:
        check() if "--check" in sys.argv else main()
    except KeyboardInterrupt:
        print("\nbye")
