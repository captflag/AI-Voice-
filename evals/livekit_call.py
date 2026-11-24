"""A scripted caller for the LiveKit transport: joins a room, speaks, and transcribes what the bot says back.

Start a LiveKit server and the bot first:

    bin/livekit/livekit-server.exe --dev
    LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret python bot.py -t livekit
    python evals/livekit_call.py

It asks the runner for a room, joins as the caller, waits for the greeting, then says each line in LINES and
prints the bot's reply and how long the reply took to start. Pipecat's eval harness only drives its own
transport, so this is how a call over LiveKit gets tested end to end. evals/transfer_call.py reuses Line.
"""
import asyncio
import json
import sys
import time
from functools import cache
from pathlib import Path

import aiohttp
import numpy as np
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from livekit import rtc

ROOT = Path(__file__).resolve().parent.parent
RUNNER = "http://localhost:7860/start"
LINES = ["How much is a routine cleaning?", "Do you take Cigna?", "No, that's everything. Thank you, bye."]
RATE = 16000              # what a line sends and receives
FRAME = RATE // 100       # 10 ms
LOUD = 500                # int16 RMS above this counts as someone speaking
QUIET_AFTER = 1.5         # this much silence ends a reply


@cache
def voice():
    return Kokoro(str(ROOT / "models" / "kokoro-v1.0.onnx"), str(ROOT / "models" / "voices-v1.0.bin"))


@cache
def whisper():
    return WhisperModel("base.en", device="cpu", compute_type="int8")


def rms(samples):
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if len(samples) else 0.0


async def start_call():
    """Ask the bot's runner for a new room; returns its URL and a caller token."""
    async with aiohttp.ClientSession() as http:
        async with http.post(RUNNER, json={"transport": "livekit"}) as response:
            session = await response.json()
    return session["url"], session["token"]


class Line:
    """One participant in a LiveKit room with a microphone it speaks into and ears it transcribes."""

    def __init__(self, speaker="am_michael"):
        self.speaker = speaker
        self.heard = []            # (time, int16 samples) of everything heard
        self.data = []             # data messages received, as dicts
        self.speech = asyncio.Queue()
        self.room = None
        self.left = []             # identities that left the room
        self.tasks = []

    async def join(self, url, token):
        self.room = rtc.Room()

        @self.room.on("track_subscribed")
        def on_track(track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                self.tasks.append(asyncio.ensure_future(self._listen(track)))

        self.room.on("data_received")(lambda packet: self.data.append(json.loads(packet.data)))
        self.room.on("participant_disconnected")(lambda participant: self.left.append(participant.identity))
        self.room.on("participant_connected")(self._briefing)
        await self.room.connect(url, token)
        for participant in self.room.remote_participants.values():
            self._briefing(participant)
        source = rtc.AudioSource(RATE, 1)
        await self.room.local_participant.publish_track(rtc.LocalAudioTrack.create_audio_track("mic", source),
                                                        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        self.tasks.append(asyncio.ensure_future(self._send(source)))

    def _briefing(self, participant):
        """The bot's transfer briefing arrives as an attribute of its desk connection; file it with the data."""
        if participant.attributes.get("brief"):
            self.data.append({"type": "brief", "text": participant.attributes["brief"]})

    async def leave(self):
        for task in self.tasks:
            task.cancel()
        self.tasks = []
        if self.room:
            await self.room.disconnect()
            self.room = None

    async def send(self, message):
        await self.room.local_participant.publish_data(json.dumps(message).encode(), reliable=True)

    async def wait_for_data(self, kind, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for message in self.data:
                if message.get("type") == kind:
                    return message
            await asyncio.sleep(0.1)
        return None

    def say(self, text):
        """Queue one line of speech; returns the time it will have finished playing."""
        samples, rate = voice().create(text, voice=self.speaker, speed=1.0, lang="en-us")
        samples = np.interp(np.arange(0, len(samples), rate / RATE), np.arange(len(samples)), samples)
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        for i in range(0, len(pcm) - FRAME + 1, FRAME):
            self.speech.put_nowait(pcm[i:i + FRAME])
        return time.monotonic() + self.speech.qsize() / 100

    async def _send(self, source):
        """Keep the microphone open: queued speech when there is some, silence otherwise, in real time."""
        silence = np.zeros(FRAME, np.int16)
        start, sent = time.monotonic(), 0
        while True:
            chunk = silence if self.speech.empty() else self.speech.get_nowait()
            await source.capture_frame(rtc.AudioFrame(chunk.tobytes(), RATE, 1, FRAME))
            sent += 1
            await asyncio.sleep(max(0.0, start + sent / 100 - time.monotonic()))

    async def _listen(self, track):
        async for event in rtc.AudioStream(track, sample_rate=RATE, num_channels=1):
            self.heard.append((time.monotonic(), np.frombuffer(event.frame.data, np.int16).copy()))

    async def reply(self, after, timeout=30.0):
        """Wait for speech to start and finish after `after`; return (start time, text)."""
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
        return began, self.transcript(began - 0.2, last_loud + 0.2)

    def transcript(self, start, end):
        """Everything heard between two times, as text."""
        chunks = [c for t, c in self.heard if start <= t <= end]
        if not chunks:
            return ""
        audio = np.concatenate(chunks).astype(np.float32) / 32768
        return " ".join(s.text.strip() for s in whisper().transcribe(audio, language="en")[0])


async def main():
    caller = Line()
    await caller.join(*await start_call())
    print(f"joined room {caller.room.name}")

    connected = time.monotonic()
    began, text = await caller.reply(connected)
    print(f"bot (greeting, {began - connected:.1f} s after joining): {text}" if began else "no greeting heard")
    failures = 0 if began else 1
    for line in LINES:
        ends = caller.say(line)
        print(f"caller: {line}")
        began, text = await caller.reply(ends)
        if began:
            print(f"bot ({(began - ends) * 1000:.0f} ms after the caller stopped): {text}")
        else:
            print("bot: (no reply)")
            failures += 1
    for _ in range(150):
        if caller.left:
            break
        await asyncio.sleep(0.1)
    print("bot left the room" if caller.left else "bot did not leave the room")
    failures += 0 if caller.left else 1
    await caller.leave()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
