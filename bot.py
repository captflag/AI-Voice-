"""Phase 4: the same receptionist, rebuilt on Pipecat, talking through a browser.

    python bot.py          then open http://localhost:7860/client and click Connect

voice_loop.py is the hand-built version of this pipeline. Here the framework
supplies what that file had to implement:

    mic and speaker loop    -> a WebRTC transport (the browser also cancels echo,
                               so no headset is needed)
    Vad + Endpointer        -> SileroVADAnalyzer plus Smart Turn, a model that
                               judges whether a sentence sounds finished
    interrupt() and Turn    -> built-in interruption handling
    sentences() + play_queue-> the TTS service's own text aggregation and playout
    the timing line         -> UserBotLatencyObserver

Phase 5 adds tools (tools.py): the receptionist can check free times, book and reschedule
appointments, and hang up when the caller is done. Phase 8 adds LiveKit, transfers to the front desk
(transfer.py) and OpenTelemetry traces (telemetry.py).
"""
import functools
import os
import time
from datetime import date
from pathlib import Path

import aiohttp
import ctranslate2
import onnxruntime as ort
from loguru import logger
from opentelemetry import context as otel_context

# Importing voice_loop also puts the NVIDIA DLLs on PATH, which Whisper needs here too.
import rag
import telemetry
from tools import TOOLS, Refused, digits
from voice_loop import MODEL, MODELS_DIR, VOICE

KNOWLEDGE = rag.Index()  # the clinic's documents, embedded once when the server starts
TRACING = telemetry.setup()  # OpenTelemetry, when OTEL_EXPORTER_OTLP_ENDPOINT is set

SYSTEM = (
    "You are the receptionist at Bright Smile Dental, answering a phone call. Your words are spoken aloud, so "
    "reply in one or two short sentences of plain speech: no lists, no markdown, no emoji. Never answer with "
    "silence. "
    "You can check free times, book appointments and reschedule them with your tools. The clinic is open Monday "
    "to Friday, 9 AM to 5 PM, and appointments start on the hour or the half hour. "
    "Only offer times that check_availability returned, and offer at most three at once. "
    "To book you need the caller's name, phone number, the reason for the visit and a time. Speech recognition "
    "mishears names and numbers, so before booking read the name, the whole number (never just its last "
    "digits), day and time back in one sentence "
    "and ask once whether that is right. Book as soon as the caller says yes; do not ask a second time. "
    "Phone numbers here have 10 digits. If the number you heard does not have exactly 10, do not read it back: "
    "ask the caller to type it on their phone keypad and press the pound key. Keypad entries reach you as "
    "'DTMF:' followed by the digits, and they are exact. "
    "Say an appointment is booked, moved or cancelled only after the tool that does it has succeeded; if you "
    "have only checked a time, say that you checked it. "
    "If a tool reports a problem, tell the caller plainly and suggest an alternative. "
    "Callers also ask about prices, insurance, opening hours, parking, first visits, cancellations and "
    "emergencies. Facts about the clinic arrive inside <clinic_facts> tags after the caller's words. They are "
    "your own knowledge as the receptionist: answer from them, and never mention the tags, \"the facts\", notes, "
    "documents or where a fact came from. The facts only cover the topics they mention: a list of what the clinic offers or "
    "accepts settles questions about that kind of thing only, and anything the facts do not mention is unknown, "
    "not a no. When the facts do not cover a question, do not say whether the clinic does or does not do it, and "
    "do not explain why you cannot answer: say \"I'm not sure about that\" and offer to have someone call back. "
    "Never guess a price or a policy. "
    "Never give medical advice or say which symptoms matter: for pain or swelling, offer an emergency "
    "appointment, and repeat only the emergency guidance the facts give. "
    "If the caller asks to speak to a person, wants to complain or dispute a bill, or you have tried and still "
    "cannot help, call transfer_to_staff without saying anything first; it tells the caller to hold. If nobody "
    "can take the call, offer to take a message: get the caller's name, number and what it is about, read the "
    "name and number back, and call take_message once they agree. "
    "When the caller has nothing more to ask, call end_call with your goodbye; it says it and hangs up. "
    "Today is {today:%A} {today.day} {today:%B} {today.year}."
)

GPU = "CUDAExecutionProvider" in ort.get_available_providers()
if GPU:
    ort.preload_dlls()  # load the CUDA libraries pip put in the venv
    os.environ.setdefault("ONNX_PROVIDER", "CUDAExecutionProvider")  # tells kokoro-onnx to synthesise on the GPU
ort.set_default_logger_severity(3)  # errors only

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.evals.transport import EvalTransportParams
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.frames.frames import LLMContextFrame, LLMRunFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.service_switcher import ServiceSwitcher, ServiceSwitcherStrategyFailover
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.dtmf_aggregator import DTMFAggregator
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.runner.types import LiveKitRunnerArguments, RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
import pipecat.services.kokoro.tts as kokoro_tts
import pipecat.services.whisper.stt as whisper_stt
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.whisper.stt import WhisperSTTService

# One copy of each local model per process, shared by every call. Pipecat loads fresh copies for each call, and
# on the LiveKit runner, where one process takes every call, six calls filled the 6 GB GPU and the voice took
# up to 29 s to start. (The eval harness starts a new process per call, so it never showed.)
kokoro_tts.Kokoro = functools.cache(kokoro_tts.Kokoro)
whisper_stt.WhisperModel = functools.cache(whisper_stt.WhisperModel)
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.livekit.transport import LiveKitParams
from pipecat.audio.mixers.soundfile_mixer import SoundfileMixer

import transfer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

# Background sound. In a first live test the bot treated a noisy room as one long caller turn and was
# interrupted 28 times in 11 minutes.
#
# MIN_VOLUME: the loudness score (0 to 1) a frame needs before it can count as speech. Pipecat's default
# is 0.6. Measured with this VAD: a voice at normal level scores about 0.81, half as loud 0.75, a quarter
# 0.69, and ten times quieter still 0.61, so the default lets distant voices through. The speech-confidence
# score cannot do this job: it stays near 1.0 even for a voice fifty times quieter.
# In the next live test the caller's own speech peaked at 0.82 to 0.96 and nothing else was picked up.
# Raise it if far voices still get through; lower it if the bot ignores you. The log prints the loudness
# of every stretch of speech it accepts.
#
# Tried and removed: requiring three recognised words before a turn could start (MinWordsUserTurnStartStrategy).
# It made a turn start only once its transcript arrived, which added a fixed 0.8 s wait to every reply.
MIN_VOLUME = 0.72


# The bot's voice: "kokoro", local and free, or "elevenlabs", a cloud voice that needs ELEVENLABS_API_KEY.
# Speech-to-text stays local either way. Set as an environment variable, so the eval suite can switch it too.
TTS = os.getenv("TTS", "kokoro")
if TTS not in ("kokoro", "elevenlabs"):
    raise SystemExit(f"TTS must be kokoro or elevenlabs, not {TTS!r}")
if TTS == "elevenlabs" and not os.getenv("ELEVENLABS_API_KEY"):
    raise SystemExit("TTS=elevenlabs needs ELEVENLABS_API_KEY")
SARAH = "EXAVITQu4vr4xnSDxMaL"  # an ElevenLabs default voice, close to Kokoro's af_sarah; ELEVENLABS_VOICE_ID overrides it
# Paths written the way Pipecat writes them, so the shared Kokoro (see above) is one model for calls and briefings.
KOKORO_FILES = (str(Path(MODELS_DIR, "kokoro-v1.0.onnx")), str(Path(MODELS_DIR, "voices-v1.0.bin")))
# Once ElevenLabs has failed, new calls skip it for this many seconds, then the next call tries it again. Each
# failure costs a call about 3 s of silence before its first word, and while the key or the service stayed broken
# every call paid it. Kept per process: on LiveKit and the phone lines one process takes every call.
CLOUD_RETRY_AFTER = float(os.getenv("CLOUD_VOICE_RETRY_AFTER", "300"))
cloud_down_until = 0.0  # time.monotonic() before which calls go straight to Kokoro


HOTWORDS = "Bright Smile Dental, Cigna, Delta Dental, MetLife, Aetna, Medicare, Apple Pay, Riverside Orthodontics"


class LoudnessLoggingVAD(SileroVADAnalyzer):
    """Silero VAD that also logs how loud each accepted stretch of speech was, for tuning MIN_VOLUME."""

    _peak = None

    async def analyze_audio(self, buffer):
        state = await super().analyze_audio(buffer)
        if state in (VADState.SPEAKING, VADState.STOPPING):
            self._peak = max(self._peak or 0.0, self._prev_volume)  # ponytail: reads a private field of the analyzer
        elif self._peak is not None:
            logger.info(f"speech ended, loudness peak {self._peak:.2f} (MIN_VOLUME {MIN_VOLUME})")
            self._peak = None
        return state


class ClinicNotes(FrameProcessor):
    """Retrieval. When the caller's turn is on its way to the LLM, attach the clinic notes that match it.

    The lookup happens before the LLM is called, so answering from the notes costs one embedding (about
    10 ms) and no extra LLM round trip. Only the caller's current turn is ever changed: its notes block is
    kept after its last fragment. Everything before the bot's last reply stays exactly as it was.
    """

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # A speculative inference (eager end of turn, off by default) runs on a copy of the conversation that
        # Pipecat builds for it, so the facts are attached to that copy and the real history is untouched.
        if isinstance(frame, LLMContextFrame):
            query, found, messages = rag.notes_for_turn(KNOWLEDGE, frame.context.messages)
            if messages is not None:
                frame.context.set_messages(messages)  # changes only the caller's current turn
            if query:
                hits = ", ".join(f"{c['section']} ({s:.2f})" for s, c in found) or "nothing"
                logger.info(f"retrieved for {query[-60:]!r}: {hits}")
        await self.push_frame(frame, direction)


async def caller_number(call, provider):
    """The phone number a call came from, or None if it cannot be found.

    Exotel sends it with the call. Twilio sends it only when the TwiML passes it as a stream parameter, which
    the development runner's TwiML does not, so for Twilio the usual route is its REST API. Plivo's stream does
    not carry it, so on Plivo the bot asks.
    """
    if call.from_number:
        return call.from_number
    sid, token = os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN")
    if not (provider == "twilio" and sid and token and call.call_id):
        return None
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls/{call.call_id}.json"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as http:
            async with http.get(url, auth=aiohttp.BasicAuth(sid, token)) as response:
                return (await response.json()).get("from") if response.status == 200 else None
    except (aiohttp.ClientError, TimeoutError) as problem:
        logger.warning(f"caller ID lookup failed, the bot will ask for the number instead: {problem}")
        return None


async def bot(runner_args: RunnerArguments):
    """Called by the runner once per call: a browser session, a phone call or an eval run."""
    # FastAPI traces the runner's /start request, and this call would become a child of that 1 ms span. Each
    # call gets a trace of its own instead, with its conversation span as the root.
    otel_context.attach(otel_context.Context())
    transport = await create_transport(
        runner_args,
        {
            "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
            # A LiveKit room: the base for adding a second person (a staff member) to the call.
            "livekit": lambda: LiveKitParams(
                audio_in_enabled=True, audio_out_enabled=True, audio_out_sample_rate=transfer.RATE,
                # Hold music for transfers, silent until a transfer turns it on.
                audio_out_mixer=SoundfileMixer(sound_files={"hold": transfer.hold_music()}, default_sound="hold",
                                               volume=0.3, mixing=False)),
            # Phone calls through Twilio, Plivo or Exotel. The runner tells them apart by their first message and
            # adds each one's audio format (mu-law for Twilio and Plivo, raw 16-bit audio for Exotel) itself.
            "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
            "plivo": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
            "exotel": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
            # Lets Pipecat's eval harness connect as a caller (see evals/).
            "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
        },
    )

    system = SYSTEM.format(today=date.today())
    call = getattr(runner_args, "call_data", None)  # set for phone calls only
    number = await caller_number(call, runner_args.transport_type) if call else None
    try:
        number = digits(number) if number else None  # +1..., 0... and +91... caller IDs, as the 10 digits the tools want
    except Refused:
        number = None  # not a number the clinic can store, so the bot asks for one
    if number:
        system += (
            f" The caller is phoning from {number}. Use that as their phone number: do not ask them to say a "
            "number, and do not check it on its own; it is read back with the rest before booking, like any number."
        )

    stt = WhisperSTTService(
        device="cuda" if ctranslate2.get_cuda_device_count() else "cpu",
        compute_type="float32" if ctranslate2.get_cuda_device_count() else "int8",
        # Names from the clinic documents that Whisper otherwise mishears ("Cigna" came out as "signal").
        settings=WhisperSTTService.Settings(model="base.en", hotwords=HOTWORDS),
    )
    llm = AnthropicLLMService(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        settings=AnthropicLLMService.Settings(model=MODEL, system_instruction=system),
    )
    tts = KokoroTTSService(model_path=KOKORO_FILES[0], voices_path=KOKORO_FILES[1],
                           settings=KokoroTTSService.Settings(voice=VOICE))
    if TTS == "elevenlabs" and time.monotonic() < cloud_down_until:
        logger.info(f"the cloud voice is down, so this call uses Kokoro; ElevenLabs is tried again in "
                    f"{cloud_down_until - time.monotonic():.0f} s")
    elif TTS == "elevenlabs":
        # Flash v2.5 by default, ElevenLabs' lowest-latency model. It streams over one websocket per call.
        cloud = ElevenLabsTTSService(api_key=os.environ["ELEVENLABS_API_KEY"],
                                     settings=ElevenLabsTTSService.Settings(voice=os.getenv("ELEVENLABS_VOICE_ID", SARAH)))
        # Kokoro stands behind it. With a bad key, Pipecat gave up on ElevenLabs after three refused connections and
        # the call went on in silence; now whatever ElevenLabs cannot say (a bad key, an outage, no credit) is
        # said in the local voice instead.
        tts = ServiceSwitcher(services=[cloud, tts], strategy_type=ServiceSwitcherStrategyFailover)

    # The conversation history, and the two processors that write the caller's and the bot's turns into it.
    # Pipecat reads each tool's signature and docstring and registers it; each call is also a span.
    context = LLMContext(tools=[telemetry.traced(tool) for tool in TOOLS])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=LoudnessLoggingVAD(params=VADParams(min_volume=MIN_VOLUME))),
    )

    # The gate's own voice detector hears a caller on hold, so a "never mind" can still be understood.
    gate = transfer.HoldGate(vad=SileroVADAnalyzer(params=VADParams(min_volume=MIN_VOLUME)))
    pipeline = Pipeline([
        transport.input(),     # caller audio in
        stt,
        gate,                  # drops the caller's audio and words while they are on hold for a transfer
        DTMFAggregator(),      # keypad presses become a "DTMF: 7413333333#" transcription
        user_aggregator,       # decides when the caller's turn is over
        ClinicNotes(),         # attaches matching clinic notes to the caller's turn
        llm,
        tts,
        transport.output(),    # bot audio out
        assistant_aggregator,  # records what the bot actually said
    ])

    latency = UserBotLatencyObserver()
    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
                            observers=[latency], enable_tracing=TRACING,
                            conversation_id=getattr(runner_args, "room_name", None))  # LiveKit: the room's name
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    @latency.event_handler("on_latency_measured")
    async def on_latency_measured(observer, seconds):
        logger.info(f"reply started {seconds * 1000:.0f} ms after the caller stopped")

    @latency.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(observer, breakdown):
        for line in breakdown.turn_contribution_lines():
            logger.info(f"    {line}")

    if isinstance(tts, ServiceSwitcher):
        @tts.strategy.event_handler("on_service_switched")
        async def on_voice_switched(strategy, service):
            # The reply ElevenLabs had when it failed is lost: with a bad key that was the greeting, and the caller
            # heard nothing until they spoke. The context only keeps what was actually spoken, so if none of this
            # reply was, it is asked for again and said in the local voice.
            global cloud_down_until
            cloud_down_until = time.monotonic() + CLOUD_RETRY_AFTER
            logger.warning(f"the cloud voice failed, carrying on with {service.name}; new calls skip it for "
                           f"{CLOUD_RETRY_AFTER:.0f} s")
            last = context.messages[-1] if context.messages else {}
            if not (isinstance(last, dict) and last.get("role") == "assistant"):
                await worker.queue_frames([LLMRunFrame()])

    greeted = False

    async def greet(*_):
        # A phone agent speaks first. A browser reports "client ready" and a phone call "client connected";
        # a browser session fires both, so greet only once.
        nonlocal greeted
        if not greeted:
            greeted = True
            context.add_message({"role": "developer", "content": "The call has just connected. Greet the caller."})
            await worker.queue_frames([LLMRunFrame()])

    worker.rtvi.event_handler("on_client_ready")(greet)
    if call or isinstance(runner_args, LiveKitRunnerArguments):  # these callers may not send "client ready"
        transport.event_handler("on_client_connected")(greet)

    desk = None
    if isinstance(runner_args, LiveKitRunnerArguments):  # only a LiveKit room can take a second person
        # The private briefing is always read by Kokoro, loaded on the first transfer if the call uses ElevenLabs.
        desk = transfer.Desk(runner_args.url, runner_args.room_name, gate, worker, context,
                             functools.partial(kokoro_tts.Kokoro, *KOKORO_FILES), VOICE)
        llm.desk = desk  # transfer_to_staff in tools.py finds it here
        gate.on_words = desk.caller_said  # what the caller says on hold

        @transport.event_handler("on_participant_connected")
        async def on_participant_connected(transport, identity):
            desk.staff_joined(identity)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        if desk and desk.outcome == "connected":
            return  # the bot is handing over and leaving; the call goes on without it
        if desk:
            await desk.cancel("The caller hung up.")
        await runner.cancel()

    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
