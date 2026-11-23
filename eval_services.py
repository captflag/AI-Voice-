"""Services for the eval harness. Scenario files in evals/ name these functions by dotted path."""
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def caller_voice(config):
    """The simulated caller's voice: the Kokoro model files the bot already has, with a different speaker."""
    from pipecat.services.kokoro.tts import KokoroTTSService

    return KokoroTTSService(
        model_path=os.path.join(MODELS_DIR, "kokoro-v1.0.onnx"),
        voices_path=os.path.join(MODELS_DIR, "voices-v1.0.bin"),
        settings=KokoroTTSService.Settings(voice=config.get("voice", "am_michael")),
    )
