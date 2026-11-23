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
"""
import os

import ctranslate2
import onnxruntime as ort
from loguru import logger

# Importing voice_loop also puts the NVIDIA DLLs on PATH, which Whisper needs here too.
from voice_loop import MODEL, MODELS_DIR, SYSTEM, VOICE

GPU = "CUDAExecutionProvider" in ort.get_available_providers()
if GPU:
    ort.preload_dlls()  # load the CUDA libraries pip put in the venv
    os.environ.setdefault("ONNX_PROVIDER", "CUDAExecutionProvider")  # tells kokoro-onnx to synthesise on the GPU
ort.set_default_logger_severity(3)  # errors only

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import TransportParams
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


async def bot(runner_args: RunnerArguments):
    """Called by the runner once per browser session."""
    transport = await create_transport(
        runner_args, {"webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)}
    )

    stt = WhisperSTTService(
        device="cuda" if ctranslate2.get_cuda_device_count() else "cpu",
        compute_type="float32" if ctranslate2.get_cuda_device_count() else "int8",
        settings=WhisperSTTService.Settings(model="base.en"),
    )
    llm = AnthropicLLMService(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        settings=AnthropicLLMService.Settings(model=MODEL, system_instruction=SYSTEM),
    )
    tts = KokoroTTSService(
        model_path=os.path.join(MODELS_DIR, "kokoro-v1.0.onnx"),
        voices_path=os.path.join(MODELS_DIR, "voices-v1.0.bin"),
        settings=KokoroTTSService.Settings(voice=VOICE),
    )

    # The conversation history, and the two processors that write the caller's and the bot's turns into it.
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=LoudnessLoggingVAD(params=VADParams(min_volume=MIN_VOLUME))),
    )

    pipeline = Pipeline([
        transport.input(),     # caller audio in
        stt,
        user_aggregator,       # decides when the caller's turn is over
        llm,
        tts,
        transport.output(),    # bot audio out
        assistant_aggregator,  # records what the bot actually said
    ])

    latency = UserBotLatencyObserver()
    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=True), observers=[latency])
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    @latency.event_handler("on_latency_measured")
    async def on_latency_measured(observer, seconds):
        logger.info(f"reply started {seconds * 1000:.0f} ms after the caller stopped")

    @latency.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(observer, breakdown):
        for line in breakdown.turn_contribution_lines():
            logger.info(f"    {line}")

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        # A phone agent speaks first.
        context.add_message({"role": "developer", "content": "The call has just connected. Greet the caller."})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel()

    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
