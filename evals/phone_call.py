"""Pretend to be Twilio: open the media-stream WebSocket to the bot and hold a short phone call.

Speaks Twilio's protocol (connected, start, media, stop) with 8 kHz mu-law audio in 20 ms chunks,
so the bot's phone path can be tested without a Twilio account. Start the bot in phone mode first:

    $env:TWILIO_ACCOUNT_SID = "ACtest"; $env:TWILIO_AUTH_TOKEN = "test"; $env:APPOINTMENTS_DB = "fresh"
    python bot.py -t twilio -x localhost:7870 --port 7870

then, in another terminal:

    python evals/phone_call.py 7870

It prints what the caller said and how long the bot took to start answering. What the bot heard and
did is in the bot's own output.
"""
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import websockets

import voice_loop as v

PORT = sys.argv[1] if len(sys.argv) > 1 else "7870"
CHUNK = 160  # 20 ms at 8 kHz


def ulaw(pcm):
    """16-bit PCM to G.711 mu-law bytes, the encoding phone lines use."""
    sign = pcm < 0
    mag = np.minimum(np.abs(pcm.astype(np.int32)), 32635) + 0x84
    exponent = (np.floor(np.log2(mag)).astype(np.int32) - 7).clip(0, 7)
    mantissa = (mag >> (exponent + 3)) & 0x0F
    return (~((sign.astype(np.int32) << 7) | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8)


def clip(tts, text):
    samples, rate = tts.create(text, voice="am_michael")
    audio = np.interp(np.arange(0, len(samples), rate / 8000), np.arange(len(samples)), samples)
    return ulaw((np.clip(audio, -1, 1) * 32767).astype(np.int16)).tobytes()


async def main():
    tts = v.load_tts()
    lines = [
        "Hi, I'd like to book a cleaning for next Wednesday at ten thirty in the morning. My name is Sam Lee.",
        "Yes, that's all correct. I mean the nearest Wednesday.",
    ]
    clips = [clip(tts, line) for line in lines]
    silence = bytes([0xFF]) * CHUNK
    state = {"last_bot_audio": 0.0, "bot_bytes": 0, "events": {}}

    async with websockets.connect(f"ws://localhost:{PORT}/ws") as ws:
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({"event": "start", "sequenceNumber": "1", "streamSid": "MZtest", "start": {
            "accountSid": "ACtest", "streamSid": "MZtest", "callSid": "CAtest", "tracks": ["inbound"],
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
            "customParameters": {"from_number": "+15550142237", "to_number": "+15550100000"}}}))

        async def receive():
            async for raw in ws:
                msg = json.loads(raw)
                state["events"][msg["event"]] = state["events"].get(msg["event"], 0) + 1
                if msg["event"] == "media":
                    state["last_bot_audio"] = time.perf_counter()
                    state["bot_bytes"] += len(base64.b64decode(msg["media"]["payload"]))

        receiver = asyncio.create_task(receive())
        t0, n = time.perf_counter(), 0

        async def send_chunk(payload):
            nonlocal n
            n += 1
            await ws.send(json.dumps({"event": "media", "streamSid": "MZtest", "media": {
                "track": "inbound", "chunk": str(n), "timestamp": str(n * 20), "payload": base64.b64encode(payload).decode()}}))
            await asyncio.sleep(max(0, t0 + n * 0.02 - time.perf_counter()))  # real-time pacing

        async def until_bot_quiet(label, timeout=40):
            """Send silence until the bot has spoken and then been quiet for 1.5 s."""
            start, before = time.perf_counter(), state["bot_bytes"]
            while time.perf_counter() - start < timeout:
                await send_chunk(silence)
                spoken = state["bot_bytes"] > before
                if spoken and time.perf_counter() - state["last_bot_audio"] > 1.5:
                    print(f"[call] {label}: bot spoke {(state['bot_bytes'] - before) / 8000:.1f} s of audio")
                    return True
            print(f"[call] {label}: TIMED OUT waiting for the bot")
            return False

        await until_bot_quiet("greeting")
        for line, audio in zip(lines, clips):
            print(f"[call] caller says: {line}")
            for i in range(0, len(audio), CHUNK):
                await send_chunk(audio[i:i + CHUNK].ljust(CHUNK, b"\xff"))
            spoke_at, before = time.perf_counter(), state["bot_bytes"]
            while state["bot_bytes"] == before and time.perf_counter() - spoke_at < 30:
                await send_chunk(silence)
            if state["bot_bytes"] > before:
                print(f"[call] bot audio started {(time.perf_counter() - spoke_at) * 1000:.0f} ms after the caller's last word")
            await until_bot_quiet("reply")
        await ws.send(json.dumps({"event": "stop", "streamSid": "MZtest", "stop": {"accountSid": "ACtest", "callSid": "CAtest"}}))
        await asyncio.sleep(1)
        receiver.cancel()
    print("[call] events received from bot:", state["events"])


asyncio.run(main())
