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
appointments, and hang up when the caller is done.
"""
import os
from datetime import date

import aiohttp
import ctranslate2
import onnxruntime as ort
from loguru import logger

# Importing voice_loop also puts the NVIDIA DLLs on PATH, which Whisper needs here too.
import rag
from tools import TOOLS
from voice_loop import MODEL, MODELS_DIR, VOICE

KNOWLEDGE = rag.Index()  # the clinic's documents, embedded once when the server starts

SYSTEM = (
    "You are the receptionist at Bright Smile Dental, answering a phone call. Your words are spoken aloud, so "
    "reply in one or two short sentences of plain speech: no lists, no markdown, no emoji. Never answer with "
    "silence. "
    "You can check free times, book appointments and reschedule them with your tools. The clinic is open Monday "
    "to Friday, 9 AM to 5 PM, and appointments start on the hour or the half hour. "
    "Only offer times that check_availability returned, and offer at most three at once. "
    "To book you need the caller's name, phone number, the reason for the visit and a time. Speech recognition "
    "mishears names and numbers, so before booking read the name, number, day and time back in one sentence "
    "and ask once whether that is right. Book as soon as the caller says yes; do not ask a second time. "
    "Phone numbers here have 10 digits. If the number you heard does not have exactly 10, do not read it back: "
    "ask the caller to type it on their phone keypad and press the pound key. Keypad entries reach you as "
    "'DTMF:' followed by the digits, and they are exact. "
    "Say an appointment is booked, moved or cancelled only after the tool that does it has succeeded; if you "
    "have only checked a time, say that you checked it. "
    "If a tool reports a problem, tell the caller plainly and suggest an alternative. "
    "Callers also ask about prices, insurance, opening hours, parking, first visits, cancellations and "
    "emergencies. Facts about the clinic arrive inside <clinic_facts> tags after the caller's words. They are "
    "your own knowledge as the receptionist: answer from them, and never mention the tags, notes, documents or "
    "where a fact came from. The facts only cover the topics they mention: a list of what the clinic offers or "
    "accepts settles questions about that kind of thing only, and anything the facts do not mention is unknown, "
    "not a no. When the facts do not cover a question, do not say whether the clinic does or does not do it, and "
    "do not explain why you cannot answer: say \"I'm not sure about that\" and offer to have someone call back. "
    "Never guess a price or a policy. "
    "Never give medical advice or say which symptoms matter: for pain or swelling, offer an emergency "
    "appointment, and repeat only the emergency guidance the facts give. "
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
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.dtmf_aggregator import DTMFAggregator
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.runner.types import LiveKitRunnerArguments, RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.livekit.transport import LiveKitParams
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


async def caller_number(call):
    """The phone number a Twilio call came from, or None if it cannot be found.

    Twilio sends it with the call only when the TwiML passes it as a stream parameter. The development
    runner's TwiML does not, so the usual route is to look the call up through Twilio's REST API.
    """
    if call.from_number:
        return call.from_number
    sid, token = os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN")
    if not (sid and token and call.call_id):
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
    transport = await create_transport(
        runner_args,
        {
            "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
            # A LiveKit room: the base for adding a second person (a staff member) to the call.
            "livekit": lambda: LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
            # A phone call through Twilio. The runner adds the Twilio audio format itself.
            "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
            # Lets Pipecat's eval harness connect as a caller (see evals/).
            "eval": lambda: EvalTransportParams(audio_in_enabled=True, audio_out_enabled=True),
        },
    )

    system = SYSTEM.format(today=date.today())
    call = getattr(runner_args, "call_data", None)  # set for phone calls only
    number = await caller_number(call) if call else None
    if number:
        system += (
            f" The caller is phoning from {number}. Use that as their phone number: do not ask them to say a "
            "number, just check once that it is the right one to reach them on."
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
    tts = KokoroTTSService(
        model_path=os.path.join(MODELS_DIR, "kokoro-v1.0.onnx"),
        voices_path=os.path.join(MODELS_DIR, "voices-v1.0.bin"),
        settings=KokoroTTSService.Settings(voice=VOICE),
    )

    # The conversation history, and the two processors that write the caller's and the bot's turns into it.
    context = LLMContext(tools=TOOLS)  # Pipecat reads each tool's signature and docstring and registers it
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=LoudnessLoggingVAD(params=VADParams(min_volume=MIN_VOLUME))),
    )

    pipeline = Pipeline([
        transport.input(),     # caller audio in
        stt,
        DTMFAggregator(),      # keypad presses become a "DTMF: 7413333333#" transcription
        user_aggregator,       # decides when the caller's turn is over
        ClinicNotes(),         # attaches matching clinic notes to the caller's turn
        llm,
        tts,
        transport.output(),    # bot audio out
        assistant_aggregator,  # records what the bot actually said
    ])

    latency = UserBotLatencyObserver()
    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True), observers=[latency])
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    @latency.event_handler("on_latency_measured")
    async def on_latency_measured(observer, seconds):
        logger.info(f"reply started {seconds * 1000:.0f} ms after the caller stopped")

    @latency.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(observer, breakdown):
        for line in breakdown.turn_contribution_lines():
            logger.info(f"    {line}")

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

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel()

    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
