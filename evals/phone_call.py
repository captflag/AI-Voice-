"""Pretend to be a phone provider: open the media-stream WebSocket to the bot and hold a short booking call.

Speaks the stream protocol of Twilio, Plivo or Exotel with 8 kHz audio in 20 ms chunks, so the bot's phone
path can be tested without an account. Start the bot in phone mode first, with placeholder credentials (Twilio
and Plivo need some to hang up; the calls they make fail harmlessly):

    $env:TWILIO_ACCOUNT_SID = "ACtest"; $env:TWILIO_AUTH_TOKEN = "test"
    $env:PLIVO_AUTH_ID = "MAtest"; $env:PLIVO_AUTH_TOKEN = "test"; $env:APPOINTMENTS_DB = "fresh"
    python bot.py -t twilio -x localhost:7870 --port 7870

then, in another terminal:

    python evals/phone_call.py 7870 plivo         twilio (the default), plivo or exotel

The bot tells the providers apart by their first message, so one bot takes all three. The caller books a
cleaning. Exotel sends the caller's number with the call, in India's 0-prefixed form, so the bot should use it
and read it back with the booking; Twilio and Plivo calls here carry none, so the bot asks and the caller types it on the keypad.
Whether the booking was made, and under which number, is in the bot's output: look for book_appointment. It prints what
the caller did and how long the bot took to start answering; what the bot heard and did is in its own output.
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
PROVIDER = sys.argv[2] if len(sys.argv) > 2 else "twilio"
CALLER_ID = "09876501234"   # an Indian mobile number as Exotel's caller ID gives it
KEYPAD = "5550142237#"


def ulaw(pcm):
    """16-bit PCM to G.711 mu-law bytes, the encoding Twilio and Plivo use."""
    sign = pcm < 0
    mag = np.minimum(np.abs(pcm.astype(np.int32)), 32635) + 0x84
    exponent = (np.floor(np.log2(mag)).astype(np.int32) - 7).clip(0, 7)
    mantissa = (mag >> (exponent + 3)) & 0x0F
    return (~((sign.astype(np.int32) << 7) | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8)


# Each provider's wire format: how 16-bit PCM is encoded, bytes per sample, and its messages.
if PROVIDER == "exotel":     # raw 16-bit little-endian PCM
    encode, WIDTH, BOT_AUDIO = (lambda pcm: pcm.astype("<i2").tobytes()), 2, "media"
    opening = [{"event": "connected"},
               {"event": "start", "sequence_number": 1, "stream_sid": "STtest", "start": {
                   "stream_sid": "STtest", "call_sid": "CAtest", "account_sid": "ACtest", "from": CALLER_ID,
                   "to": "08000000000", "media_format": {"encoding": "base64", "sample_rate": "8000"}}}]
    media = lambda n, b64: {"event": "media", "sequence_number": n, "stream_sid": "STtest",  # noqa: E731
                            "media": {"chunk": n, "timestamp": str(n * 20), "payload": b64}}
    dtmf = lambda d: {"event": "dtmf", "stream_sid": "STtest", "dtmf": {"digit": d}}  # noqa: E731
    stop = {"event": "stop", "stream_sid": "STtest", "stop": {"call_sid": "CAtest", "reason": "callended"}}
elif PROVIDER == "plivo":    # mu-law
    encode, WIDTH, BOT_AUDIO = (lambda pcm: ulaw(pcm).tobytes()), 1, "playAudio"
    opening = [{"event": "start", "sequenceNumber": 0, "start": {
        "callId": "CAtest", "streamId": "STtest", "accountId": "MAtest", "tracks": ["inbound"],
        "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000}}}]
    media = lambda n, b64: {"event": "media", "sequenceNumber": n, "streamId": "STtest",  # noqa: E731
                            "media": {"track": "inbound", "timestamp": str(n * 20), "chunk": n, "payload": b64}}
    dtmf = lambda d: {"event": "dtmf", "streamId": "STtest", "dtmf": {"digit": d}}  # noqa: E731
    stop = {"event": "stop", "streamId": "STtest"}
else:                        # Twilio, mu-law
    encode, WIDTH, BOT_AUDIO = (lambda pcm: ulaw(pcm).tobytes()), 1, "media"
    opening = [{"event": "connected", "protocol": "Call", "version": "1.0.0"},
               {"event": "start", "sequenceNumber": "1", "streamSid": "MZtest", "start": {
                   "accountSid": "ACtest", "streamSid": "MZtest", "callSid": "CAtest", "tracks": ["inbound"],
                   "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}}}]
    media = lambda n, b64: {"event": "media", "streamSid": "MZtest", "media": {  # noqa: E731
        "track": "inbound", "chunk": str(n), "timestamp": str(n * 20), "payload": b64}}
    dtmf = lambda d: {"event": "dtmf", "streamSid": "MZtest", "dtmf": {"digit": d}}  # noqa: E731
    stop = {"event": "stop", "streamSid": "MZtest", "stop": {"accountSid": "ACtest", "callSid": "CAtest"}}

CHUNK = 160 * WIDTH  # 20 ms at 8 kHz


def clip(tts, text):
    samples, rate = tts.create(text, voice="am_michael")
    audio = np.interp(np.arange(0, len(samples), rate / 8000), np.arange(len(samples)), samples)
    return encode((np.clip(audio, -1, 1) * 32767).astype(np.int16))


async def main():
    tts = v.load_tts()
    # A line is something to say, or the number to type on the keypad when the bot asks for it.
    # Each line asks for its own time: one bot takes all three calls, and they share its appointment book.
    when = {"twilio": "ten thirty in the morning", "plivo": "eleven thirty in the morning", "exotel": "two in the afternoon"}
    lines = [f"Hi, I'd like to book a cleaning for next Wednesday at {when[PROVIDER]}. My name is Sam Lee."]
    if PROVIDER != "exotel":  # Exotel brings the number with the call; here it has to be typed
        lines.append(KEYPAD)
    lines.append("Yes, that's all correct.")
    clips = {line: clip(tts, line) for line in lines if line != KEYPAD}
    silence = encode(np.zeros(160, np.int16))
    state = {"last_bot_audio": 0.0, "bot_bytes": 0, "events": {}}
    replies = 0

    async with websockets.connect(f"ws://localhost:{PORT}/ws") as ws:
        for message in opening:
            await ws.send(json.dumps(message))

        async def receive():
            async for raw in ws:
                msg = json.loads(raw)
                state["events"][msg["event"]] = state["events"].get(msg["event"], 0) + 1
                if msg["event"] == BOT_AUDIO:
                    state["last_bot_audio"] = time.perf_counter()
                    state["bot_bytes"] += len(base64.b64decode(msg["media"]["payload"]))

        receiver = asyncio.create_task(receive())
        t0, n = time.perf_counter(), 0

        async def send_chunk(payload):
            nonlocal n
            n += 1
            await ws.send(json.dumps(media(n, base64.b64encode(payload).decode())))
            await asyncio.sleep(max(0, t0 + n * 0.02 - time.perf_counter()))  # real-time pacing

        async def until_bot_quiet(label, timeout=40):
            """Send silence until the bot has spoken and then been quiet for 1.5 s."""
            start, before = time.perf_counter(), state["bot_bytes"]
            while time.perf_counter() - start < timeout:
                await send_chunk(silence)
                if state["bot_bytes"] > before and time.perf_counter() - state["last_bot_audio"] > 1.5:
                    print(f"[call] {label}: bot spoke {(state['bot_bytes'] - before) / WIDTH / 8000:.1f} s of audio")
                    return True
            print(f"[call] {label}: TIMED OUT waiting for the bot")
            return False

        try:
            replies += await until_bot_quiet("greeting")
            for line in lines:
                if line == KEYPAD:
                    print(f"[call] caller types {line}")
                    for digit in line:
                        await ws.send(json.dumps(dtmf(digit)))
                        for _ in range(10):  # 200 ms between presses
                            await send_chunk(silence)
                else:
                    print(f"[call] caller says: {line}")
                    audio = clips[line]
                    for i in range(0, len(audio), CHUNK):
                        await send_chunk(audio[i:i + CHUNK].ljust(CHUNK, silence[:1]))
                spoke_at, before = time.perf_counter(), state["bot_bytes"]
                while state["bot_bytes"] == before and time.perf_counter() - spoke_at < 30:
                    await send_chunk(silence)
                if state["bot_bytes"] > before:
                    replies += 1
                    print(f"[call] bot audio started {(time.perf_counter() - spoke_at) * 1000:.0f} ms after the caller finished")
                await until_bot_quiet("reply")
            await ws.send(json.dumps(stop))
            await asyncio.sleep(1)
        except websockets.ConnectionClosed:
            # A caller who has nothing more to add can be hung up on: the bot says goodbye and ends the call.
            print("[call] the bot hung up")
        receiver.cancel()
    print(f"[call] {PROVIDER}: {replies} of {len(lines) + 1} turns answered; events from the bot: {state['events']}")
    sys.exit(0 if replies == len(lines) + 1 else 1)


asyncio.run(main())
