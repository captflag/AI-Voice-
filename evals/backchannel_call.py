"""Backchannels and interruptions, measured on the bot's audio over LiveKit.

While the bot gives a long answer, a scripted caller says something short. "Mm-hmm" and "uh-huh" only show the
caller is listening, so the bot must keep talking; "wait, stop" and "stop" are real interruptions, so it must go
quiet. Each case is a fresh call. The eval harness can wait for an event but cannot check that one never happens,
which is why this listens to the audio instead.

Times are measured at the caller's end, so they include LiveKit's path and buffering both ways: the bot before
phase 9, which stopped for any sound, went quiet 1.8 s after a caller began to speak. An interruption must land
within INTERRUPT_WITHIN, and the time it took is printed either way.

Start the LiveKit server and the bot as for evals/livekit_call.py, then:

    python evals/backchannel_call.py
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from livekit_call import LOUD, Line, rms, speech, start_call  # noqa: E402  (same folder)

ASK = "Can you tell me your opening hours, and everything I should bring to a first visit?"
CASES = [("Mm-hmm.", False), ("Uh-huh.", False), ("Wait, stop. I have a different question.", True), ("Stop.", True)]
INTERRUPT_WITHIN = 3.0
results = []


def check(ok, what, detail=""):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"  ({detail})" if detail and not ok else ""))


def first_silence(caller, since, until, length=0.8):
    """When the bot's audio first went quiet for `length` seconds after `since`, or None."""
    loud = sorted(t for t, chunk in caller.heard if since <= t <= until and rms(chunk) > LOUD)
    marks = [since, *loud, until]
    return next((a for a, b in zip(marks, marks[1:]) if b - a >= length), None)


async def case(line, interrupts):
    caller = Line()
    await caller.join(*await start_call())
    await caller.reply(time.monotonic())                       # the greeting
    asked = caller.say(ASK)
    began = None
    while not began and time.monotonic() < asked + 30:        # the answer starts
        await asyncio.sleep(0.05)
        began = next((t for t, chunk in caller.heard if t > asked and rms(chunk) > LOUD), None)
    if not began:
        check(False, f"{line!r}: the bot answered the question")
        return await caller.leave()
    await asyncio.sleep(max(0.0, began + 1.5 - time.monotonic()))
    spoke = time.monotonic()
    said = caller.say(line)
    # 2.5 s after the caller's line: long enough for an interruption to show, short enough that the answer has not
    # simply ended on its own (it runs about 8 s).
    await asyncio.sleep(max(0.0, said + 2.5 - time.monotonic()))
    quiet = first_silence(caller, spoke, said + 2.5)
    if "-v" in sys.argv:  # the bot's quiet stretches around the caller's line, in seconds from when it began
        loud = sorted(t - spoke for t, chunk in caller.heard if spoke - 3 <= t <= said + 2.5 and rms(chunk) > LOUD)
        print(f"    {line!r}: caller {0:.1f} to {said - spoke:.1f}; bot quiet " + ", ".join(
            f"{a:.1f} to {b:.1f}" for a, b in zip(loud, loud[1:]) if b - a >= 0.3))
    if interrupts:
        took = f"quiet {quiet - spoke:.1f} s after the caller began" if quiet else "the bot kept talking"
        check(quiet is not None and quiet - spoke < INTERRUPT_WITHIN, f"{line!r} stops the bot ({took})", took)
    else:
        check(quiet is None, f"{line!r} does not stop the bot",
              f"quiet {quiet - spoke:.1f} s after the caller began" if quiet else "")
    await caller.leave()
    await asyncio.sleep(2)


async def main():
    for text in [ASK, *(line for line, _ in CASES)]:  # synthesise now: mid-call it would deafen the caller
        speech(text, Line().speaker)
    for line, interrupts in CASES:
        await case(line, interrupts)
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
