"""The bot's ClinicNotes processor, run inside a real Pipecat pipeline with Pipecat's own test helper.

    python evals/test_clinic_notes.py

Covers what a voice eval cannot reach: a speculative inference (eager end of turn is off by default, so live
calls never produce one) must get the facts on its own copy of the conversation and leave the real one alone.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipecat.frames.frames import LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import run_test

import rag
from bot import ClinicNotes

GREETING = [{"role": "developer", "content": "Greet the caller."}, {"role": "assistant", "content": "Hello, how can I help?"}]


async def main():
    # A normal turn: the facts are attached to the conversation itself.
    real = LLMContext(messages=GREETING + [{"role": "user", "content": "How much is a teeth cleaning?"}])
    down, _ = await run_test(ClinicNotes(), frames_to_send=[LLMContextFrame(context=real)], expected_down_frames=[LLMContextFrame])
    assert rag.is_notes(real.messages[-1]) and "95 dollars" in real.messages[-1]["content"], real.messages[-1]
    print("normal turn: facts attached")

    # A speculative turn: Pipecat builds a copy holding the speculated words; the facts go on the copy only.
    real = LLMContext(messages=list(GREETING))
    copy = LLMContext(messages=[*real.messages, {"role": "user", "content": "Do you take Cigna?"}])
    down, _ = await run_test(ClinicNotes(), frames_to_send=[LLMContextFrame(context=copy, speculation=True)],
                             expected_down_frames=[LLMContextFrame])
    assert down[0].speculation and down[0].context is copy, "the speculative frame must pass through unchanged"
    assert rag.is_notes(copy.messages[-1]) and "Cigna" in copy.messages[-1]["content"], copy.messages[-1]
    assert real.messages == GREETING, "the real conversation must not change"
    print("speculative turn: facts attached to the copy, real conversation untouched")

    # A booking turn: no facts.
    real = LLMContext(messages=GREETING + [{"role": "user", "content": "I'd like to book a cleaning for Friday at ten."}])
    await run_test(ClinicNotes(), frames_to_send=[LLMContextFrame(context=real)], expected_down_frames=[LLMContextFrame])
    assert not any(rag.is_notes(m) for m in real.messages), real.messages
    print("booking turn: no facts")


asyncio.run(main())
