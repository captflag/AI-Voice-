"""Does Whisper hear the clinic's names? Each line is spoken in three Kokoro voices and transcribed twice,
without and with the bot's hotwords. Prints only the lines where a transcript differs from what was said.

    python evals/stt_names.py
"""
import sys
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bot import HOTWORDS

LINES = ["Do you take Cigna?", "Is MetLife accepted?", "I have Delta Dental insurance.", "Do you accept Aetna?",
         "Can I pay with Apple Pay?", "How much is a routine cleaning?", "No, that's everything. Thank you, bye.",
         "My number is 741 333 3333.", "I'd like to book a cleaning for Friday at ten."]
voice = Kokoro(str(ROOT / "models" / "kokoro-v1.0.onnx"), str(ROOT / "models" / "voices-v1.0.bin"))
whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
for speaker in ["am_michael", "bf_emma", "af_bella"]:
    for line in LINES:
        samples, rate = voice.create(line, voice=speaker, speed=1.0, lang="en-us")
        audio = np.interp(np.arange(0, len(samples), rate / 16000), np.arange(len(samples)), samples).astype(np.float32)
        plain, hot = (" ".join(s.text.strip() for s in whisper.transcribe(audio, language="en", hotwords=h)[0])
                      for h in (None, HOTWORDS))
        if plain != hot or plain.rstrip(".?!").lower() != line.rstrip(".?!").lower():
            print(f"{speaker:10s} said {line!r}\n{'':10s} plain {plain!r}\n{'':10s} hot   {hot!r}")
