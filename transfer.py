"""Phase 8: warm transfer to a person at the front desk, over LiveKit.

Staff wait in a LiveKit room of their own, the desk (staff.py serves the page they use). A transfer:

    1. hold      the caller hears a hold line, then soft hold music, with a reminder every 20 s; the bot
                 stops listening to them
    2. brief     the bot joins the desk room and tells the staff, out loud and as text, who is calling and
                 why. The caller is in another room, so cannot hear it
    3. connect   the first member of staff to press Accept is sent a token for the caller's room; when they
                 join it, the bot says who they are and leaves
    4. fallback  nobody at the desk, a decline, or no answer in 30 s: the music stops and the bot is told to
                 offer to take a message instead

The tool the model calls is transfer_to_staff in tools.py; it finds this object as params.llm.desk.
"""
import asyncio
import json
import os
import tempfile
import wave

import numpy as np
from livekit import api, rtc
from loguru import logger
from pipecat.frames.frames import (EndWorkerFrame, Frame, InputAudioRawFrame, InterimTranscriptionFrame,
                                   LLMRunFrame, MixerEnableFrame, TranscriptionFrame, TTSSpeakFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

DESK = "front-desk"
ANSWER_WITHIN = 30     # seconds from the start of the briefing
REMIND_EVERY = 20      # seconds between "thanks for holding" lines
RATE = 24000           # the bot's output rate, which the hold music must match
BOT_ID = "receptionist"


def token(room, identity, name=None, attributes=None):
    """A token to join one room, signed with the same key the runner uses."""
    grant = api.VideoGrants(room_join=True, room=room)
    return (api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
            .with_identity(identity).with_name(name or identity).with_grants(grant)
            .with_attributes(attributes or {}).to_jwt())


def hold_music():
    """A soft two-note chime every four seconds, written once to a temporary WAV for Pipecat's mixer."""
    path = os.path.join(tempfile.gettempdir(), "bright-smile-hold.wav")
    if not os.path.exists(path):
        t = np.arange(int(RATE * 0.45)) / RATE
        fade = np.minimum(1, np.minimum(t, t[::-1]) * 20)  # 50 ms fade in and out, so no clicks
        notes = [np.sin(2 * np.pi * f * t) * fade * np.exp(-3 * t) for f in (523.25, 659.25)]
        loop = np.concatenate(notes + [np.zeros(RATE * 4 - 2 * len(t))])
        with wave.open(path, "wb") as out:
            out.setnchannels(1), out.setsampwidth(2), out.setframerate(RATE)
            out.writeframes((loop * 0.5 * 32767).astype(np.int16).tobytes())
    return path


class HoldGate(FrameProcessor):
    """Drops the caller's audio and words while they are on hold, so the bot neither answers nor is interrupted.

    It sits after speech-to-text so that it also drops words already being transcribed when the hold began: a
    request said in two parts once reached the model as a second turn, and the bot talked over the hold.
    """

    def __init__(self):
        super().__init__()
        self.closed = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not (self.closed and isinstance(frame, (InputAudioRawFrame, TranscriptionFrame,
                                                   InterimTranscriptionFrame))):
            await self.push_frame(frame, direction)


class Desk:
    """One call's link to the front desk. Created per call on the LiveKit transport only."""

    def __init__(self, url, caller_room, gate, worker, context, kokoro, voice):
        self.url, self.caller_room, self.worker, self.context = url, caller_room, worker, context
        self.gate, self.kokoro, self.voice = gate, kokoro, voice
        self.room = None          # the bot's connection to the desk room, while a transfer is on
        self.staff = None         # identity of whoever accepted
        self.task = None
        self.joined = asyncio.Event()
        self.outcome = None       # "connected", "no staff", "declined", "no answer", "caller hung up"

    async def begin(self, reason, summary):
        """Start a transfer. Returns None once the caller is on hold, or why it could not start."""
        if self.task and not self.task.done():
            return None  # already on hold: a repeated request changes nothing and says nothing
        self.joined.clear()
        brief = f"Transfer from the receptionist. Reason: {reason}. {summary}"
        self.room = rtc.Room()
        # The written briefing travels in the bot's join token, so staff see it the moment the bot appears. Sent
        # as a data message straight after joining, it was lost by a browser that was still setting up the
        # connection it receives on.
        await self.room.connect(self.url, token(DESK, f"{BOT_ID}-{self.caller_room}", "Receptionist",
                                                {"brief": brief}))
        staff = [p.identity for p in self.room.remote_participants.values() if not p.identity.startswith(BOT_ID)]
        if not staff:
            await self.room.disconnect()
            self.room, self.outcome = None, "no staff"
            logger.info("transfer: nobody at the desk")
            return "Nobody is at the front desk right now. Offer to take a message so someone can call back."
        logger.info(f"transfer: {len(staff)} at the desk, briefing them: {reason}")
        self.gate.closed = True
        await self.worker.queue_frames([TTSSpeakFrame("Of course. Please hold for a moment while I get someone "
                                                      "from the front desk."), MixerEnableFrame(True)])
        self.task = asyncio.create_task(self._run(brief))
        return None

    async def _run(self, brief):
        answered = asyncio.get_running_loop().create_future()

        @self.room.on("data_received")
        def on_data(packet: rtc.DataPacket):
            message = json.loads(packet.data)
            if message.get("type") in ("accept", "decline") and not answered.done():
                answered.set_result((message["type"], packet.participant.identity))

        reminders = asyncio.create_task(self._remind())
        try:
            await self._say(brief + " Press accept to take the call.")
            choice, who = await asyncio.wait_for(answered, ANSWER_WITHIN)
            if choice == "decline":
                return await self._fall_back("declined", "The front desk could not take the call.")
            self.staff = who
            await self._send({"type": "join", "url": self.url, "token": token(self.caller_room, who)}, who)
            await asyncio.wait_for(self.joined.wait(), 15)
            reminders.cancel()
            self.outcome = "connected"
            logger.info(f"transfer: {who} joined the caller, the bot leaves")
            await self.worker.queue_frames([MixerEnableFrame(False),
                                            TTSSpeakFrame("Thanks for holding. You're through to the front desk now. "
                                                          "Goodbye from me."),
                                            EndWorkerFrame()])
        except TimeoutError:
            await self._send({"type": "withdrawn"})
            await self._fall_back("no answer", "Nobody at the front desk answered within 30 seconds.")
        finally:
            reminders.cancel()
            await self._leave_desk()

    def staff_joined(self, identity):
        """Called when anyone joins the caller's room; the bot leaves once it is the member of staff."""
        if self.staff and identity == self.staff:
            self.joined.set()

    async def cancel(self, why):
        """The caller hung up: tell the desk and stop."""
        if self.task and not self.task.done():
            self.outcome = "caller hung up"
            await self._send({"type": "withdrawn", "text": why})
            self.task.cancel()

    async def _fall_back(self, outcome, why):
        self.outcome = outcome
        logger.info(f"transfer: {outcome}")
        self.gate.closed = False
        self.context.add_message({"role": "developer", "content": f"{why} The caller is back with you and has "
                                  "heard nothing since the hold line. Apologise briefly and offer to take a message "
                                  "so someone can call them back."})
        await self.worker.queue_frames([MixerEnableFrame(False), LLMRunFrame()])

    async def _remind(self):
        while True:
            await asyncio.sleep(REMIND_EVERY)
            await self.worker.queue_frames([TTSSpeakFrame("Thanks for holding, I'm still getting someone for you.")])

    async def _send(self, message, to=None):
        if self.room:
            await self.room.local_participant.publish_data(json.dumps(message).encode(), reliable=True,
                                                           destination_identities=[to] if to else [])

    async def _say(self, text):
        """Speak into the desk room only, with the bot's own voice model."""
        source = rtc.AudioSource(RATE, 1)
        track = rtc.LocalAudioTrack.create_audio_track("briefing", source)
        await self.room.local_participant.publish_track(track, rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE))
        # ponytail: reuses the TTS service's private Kokoro model rather than loading a second copy
        samples, rate = await asyncio.to_thread(self.kokoro.create, text, voice=self.voice, speed=1.0, lang="en-us")
        if rate != RATE:
            samples = np.interp(np.arange(0, len(samples), rate / RATE), np.arange(len(samples)), samples)
        pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
        frame = RATE // 100
        for i in range(0, len(pcm) - frame + 1, frame):
            await source.capture_frame(rtc.AudioFrame(pcm[i:i + frame].tobytes(), RATE, 1, frame))
        await source.wait_for_playout()

    async def _leave_desk(self):
        if self.room:
            await self.room.disconnect()
            self.room = None
