"""A scripted caller for the LiveKit transport: joins a room, speaks, and transcribes what the bot says back.

Start a LiveKit server and the bot first:

    bin/livekit/livekit-server.exe --dev
    LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret python bot.py -t livekit
    python evals/livekit_call.py

It asks the runner for a room, joins as the caller, waits for the greeting, then says each line in LINES and
prints the bot's reply and how long the reply took to start. Pipecat's eval harness only drives its own
transport, so this is how a call over LiveKit gets tested end to end.
"""
import asyncio
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from livekit import rtc

ROOT = Path(__file__).resolve().parent.parent
RUNNER = "http://localhost:7860/start"
LINES = ["How much is a routine cleaning?", "Do you take Cigna?", "No, that's everything. Thank you, bye."]
RATE = 16000              # what the caller sends and receives
FRAME = RATE // 100       # 10 ms
LOUD = 500                # int16 RMS above this counts as the bot speaking
QUIET_AFTER = 1.5         # this much silence ends a bot reply


def rms(samples):
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if len(samples) else 0.0


class Caller:
    def __init__(self):
        self.heard = []           # (time, int16 samples) of the bot's audio
        self.speech = asyncio.Queue()
        self.voice = Kokoro(str(ROOT / "models" / "kokoro-v1.0.onnx"), str(ROOT / "models" / "voices-v1.0.bin"))
        self.whisper = WhisperModel("base.en", device="cpu", compute_type="int8")

    def say(self, text):
        """Queue one line of speech; returns when it has been queued, with the time it will finish."""
        samples, rate = self.voice.create(text, voice="am_michael", speed=1.0, lang="en-us")
        samples = np.interp(np.arange(0, len(samples), rate / RATE), np.arange(len(samples)), samples)
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        for i in range(0, len(pcm) - FRAME + 1, FRAME):
            self.speech.put_nowait(pcm[i:i + FRAME])
        return len(pcm) / RATE

    async def send(self, source):
        """Keep the microphone open: queued speech when there is some, silence otherwise, in real time."""
        silence = np.zeros(FRAME, np.int16)
        start, sent = time.monotonic(), 0
        while True:
            chunk = silence if self.speech.empty() else self.speech.get_nowait()
            await source.capture_frame(rtc.AudioFrame(chunk.tobytes(), RATE, 1, FRAME))
            sent += 1
            await asyncio.sleep(max(0.0, start + sent / 100 - time.monotonic()))

    async def listen(self, track):
        async for event in rtc.AudioStream(track, sample_rate=RATE, num_channels=1):
            self.heard.append((time.monotonic(), np.frombuffer(event.frame.data, np.int16).copy()))

    async def reply(self, after, timeout=30.0):
        """Wait for the bot to start and finish speaking after `after`; return (start time, text)."""
        deadline, began, last_loud = time.monotonic() + timeout, None, None
        while time.monotonic() < deadline:
            await asyncio.sleep(0.05)
            for t, chunk in self.heard:
                if t > after and rms(chunk) > LOUD:
                    began = began or t
                    last_loud = t
            if began and time.monotonic() - last_loud > QUIET_AFTER:
                break
        if not began:
            return None, ""
        audio = np.concatenate([c for t, c in self.heard if began - 0.2 <= t <= last_loud + 0.2])
        segments, _ = self.whisper.transcribe(audio.astype(np.float32) / 32768, language="en")
        return began, " ".join(s.text.strip() for s in segments)


async def main():
    async with aiohttp.ClientSession() as http:
        async with http.post(RUNNER, json={"transport": "livekit"}) as response:
            session = await response.json()
    caller, room = Caller(), rtc.Room()
    left = asyncio.Event()

    @room.on("track_subscribed")
    def on_track(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.ensure_future(caller.listen(track))

    room.on("participant_disconnected")(lambda participant: left.set())

    await room.connect(session["url"], session["token"])
    source = rtc.AudioSource(RATE, 1)
    await room.local_participant.publish_track(rtc.LocalAudioTrack.create_audio_track("mic", source),
                                               rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    sender = asyncio.ensure_future(caller.send(source))
    print(f"joined room {room.name}")

    connected = time.monotonic()
    began, text = await caller.reply(connected)
    print(f"bot (greeting, {began - connected:.1f} s after joining): {text}" if began else "no greeting heard")
    failures = 0 if began else 1
    for line in LINES:
        ends = time.monotonic() + caller.say(line)
        print(f"caller: {line}")
        began, text = await caller.reply(ends)
        if began:
            print(f"bot ({(began - ends) * 1000:.0f} ms after the caller stopped): {text}")
        else:
            print("bot: (no reply)")
            failures += 1
    try:
        await asyncio.wait_for(left.wait(), 15)
        print("bot left the room")
    except TimeoutError:
        print("bot did not leave the room")
        failures += 1
    sender.cancel()
    await room.disconnect()
    sys.exit(1 if failures else 0)


asyncio.run(main())
