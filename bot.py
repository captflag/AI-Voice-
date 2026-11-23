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
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
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
