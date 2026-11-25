"""Warm transfer, end to end: a scripted caller and a scripted member of staff in real LiveKit rooms.

Start the LiveKit server and the bot as for evals/livekit_call.py, then:

    python evals/transfer_call.py              every scenario
    python evals/transfer_call.py connected    one of them

Scenarios:
    connected    staff at the desk accept: the caller is held, staff are briefed, then the two are connected
                 and the bot leaves. The caller must never hear the briefing
    nobody       nobody at the desk: no hold, the bot offers to take a message
    declined     staff decline: the bot comes back and offers to take a message
    no_answer    staff ignore the transfer: after 30 s the bot comes back and the desk is told it was withdrawn
    hung_up      the caller hangs up while on hold: the desk is told
    never_mind   on hold, the caller asks how long it will take (the hold goes on), then says "never mind": the
                 desk is told the transfer is off and the bot comes back to the caller
"""
import asyncio
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from livekit_call import LOUD, Line, rms, start_call  # noqa: E402  (same folder)
from transfer import ANSWER_WITHIN, DESK, token  # noqa: E402

ASK = "Hi, I'd like to speak to a real person about a charge on my bill, please."
STAFF_SAYS = "Hi, this is Sam at the front desk. How can I help with your bill?"
results = []


def check(ok, what, detail=""):
    results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f"  ({detail})" if detail and not ok else ""))


async def at_desk():
    staff = Line("bf_emma")
    await staff.join(os.environ["LIVEKIT_URL"], token(DESK, "staff-sam", "Sam"))
    return staff


async def caller_asks():
    """A new call: wait for the greeting, then ask for a person. Returns the caller and when they finished."""
    caller = Line()
    await caller.join(*await start_call())
    await caller.reply(time.monotonic())
    return caller, caller.say(ASK)


async def connected():
    staff = await at_desk()
    caller, asked = await caller_asks()
    brief = await staff.wait_for_data("brief", 30)
    check(brief and re.search(r"bill|charge", brief["text"], re.I), "staff got a written briefing about the bill",
          brief and brief["text"])
    _, spoken = await staff.reply(asked)
    check(re.search(r"bill|charge", spoken, re.I), "staff heard the briefing spoken", spoken)
    await staff.send({"type": "accept"})
    join = await staff.wait_for_data("join", 10)
    check(join, "staff were sent into the caller's room")
    if not join:
        return await asyncio.gather(staff.leave(), caller.leave())
    await staff.leave()
    await staff.join(join["url"], join["token"])
    for _ in range(200):  # the bot says who the caller is through to, then leaves
        if caller.left:
            break
        await asyncio.sleep(0.1)
    check(caller.left, "the bot left the caller's room")
    said = staff.say(STAFF_SAYS)
    _, heard = await caller.reply(said - 4)
    check(re.search(r"front desk|Sam", heard, re.I), "the caller heard the member of staff", heard)
    everything = caller.transcript(asked - 10, time.monotonic())
    check(re.search(r"hold", everything, re.I), "the caller was told to hold", everything[:200])
    check(re.search(r"through", everything, re.I), "the caller was told they are through", everything[:300])
    check(not re.search(r"reason|transfer from", everything, re.I), "the caller never heard the briefing", everything)
    await asyncio.gather(staff.leave(), caller.leave())


async def nobody():
    caller, asked = await caller_asks()
    _, text = await caller.reply(asked)
    check(re.search(r"message|call you back", text, re.I), "the bot offered to take a message", text)
    check(not re.search(r"\bhold\b", text, re.I), "the caller was not put on hold", text)
    await caller.leave()


async def declined():
    staff = await at_desk()
    caller, asked = await caller_asks()
    brief = await staff.wait_for_data("brief", 30)
    check(brief, "staff got the briefing")
    await staff.reply(asked)
    await staff.send({"type": "decline"})
    await asyncio.sleep(12)
    text = caller.transcript(time.monotonic() - 12, time.monotonic())
    check(re.search(r"message|call you back", text, re.I), "the bot came back and offered a message", text)
    await asyncio.gather(staff.leave(), caller.leave())


async def no_answer():
    staff = await at_desk()
    caller, asked = await caller_asks()
    check(await staff.wait_for_data("brief", 30), "staff got the briefing")
    withdrawn = await staff.wait_for_data("withdrawn", ANSWER_WITHIN + 30)
    check(withdrawn, "the desk was told the transfer was withdrawn")
    await asyncio.sleep(12)
    text = caller.transcript(time.monotonic() - 12, time.monotonic())
    check(re.search(r"message|call you back", text, re.I), "the bot came back and offered a message", text)
    held = caller.transcript(asked, asked + ANSWER_WITHIN)
    check(re.search(r"still getting someone|thanks for holding", held, re.I), "the caller heard a hold reminder", held)
    # The chime plays every 4 s, so music means sound in nearly every 4-second stretch of the hold.
    bins = {int((t - asked) // 4) for t, chunk in caller.heard if asked + 8 <= t < asked + 36 and rms(chunk) > LOUD}
    check(len(bins) >= 6, "the caller heard hold music", f"sound in {len(bins)} of 7 four-second stretches")
    await asyncio.gather(staff.leave(), caller.leave())


async def hung_up():
    staff = await at_desk()
    caller, asked = await caller_asks()
    check(await staff.wait_for_data("brief", 30), "staff got the briefing")
    await caller.leave()
    withdrawn = await staff.wait_for_data("withdrawn", 10)
    check(withdrawn and "hung up" in withdrawn.get("text", ""), "the desk was told the caller hung up", withdrawn)
    await staff.leave()


async def never_mind():
    staff = await at_desk()
    caller, asked = await caller_asks()
    check(await staff.wait_for_data("brief", 30), "staff got the briefing")
    await asyncio.sleep(4)
    await asyncio.sleep(max(0.0, caller.say("How much longer will this take?") - time.monotonic()))
    check(not await staff.wait_for_data("withdrawn", 8), "a question on hold does not call the transfer off")
    await asyncio.sleep(max(0.0, caller.say("Actually, never mind. I'll call back later.") - time.monotonic()))
    withdrawn = await staff.wait_for_data("withdrawn", 15)
    check(withdrawn and "no longer" in withdrawn.get("text", ""), "the desk was told the caller called it off",
          withdrawn)
    since = time.monotonic()
    await asyncio.sleep(10)
    text = caller.transcript(since, time.monotonic())
    check(re.search(r"anything else|help|no problem|of course|okay|sure", text, re.I),
          "the bot came back to the caller", text)
    check(not caller.left and not await staff.wait_for_data("join", 1), "nobody was put through, and the bot stayed")
    await asyncio.gather(staff.leave(), caller.leave())


SCENARIOS = {f.__name__: f for f in (connected, nobody, declined, no_answer, hung_up, never_mind)}


async def main():
    for name in sys.argv[1:] or SCENARIOS:
        print(name)
        await SCENARIOS[name]()
        await asyncio.sleep(2)  # let the bot's room close before the next call
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":  # evals/trace_check.py imports the scenarios
    asyncio.run(main())
