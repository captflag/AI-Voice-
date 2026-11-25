"""Traces, end to end: run real transfer calls and check the spans the bot sends over OpenTelemetry.

This script is the trace backend: it takes OTLP/HTTP on port 4318, where Jaeger or SigNoz would. Start the
LiveKit server as for evals/livekit_call.py, then the bot with tracing on, then this:

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 LIVEKIT_URL=... python bot.py -t livekit
    python evals/trace_check.py

It runs the "connected" and "nobody" calls from evals/transfer_call.py and checks that each call is one trace:
conversation > turn > stt, llm and tts, with the tool call inside its turn and the transfer inside the tool call.
It also checks that nothing anyone said on the call reached the backend, while the timings and token counts did.
Start both the bot and this script with TRACE_CALL_CONTENT=1 to check the opposite: that the switch keeps it all.
"""
import asyncio
import gzip
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from aiohttp import web
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transfer_call  # noqa: E402  (same folder)
from transfer_call import check  # noqa: E402
import telemetry  # noqa: E402  (the project root, which transfer_call puts on the path)

PHONE = "5550142237"  # transfer_call's caller never says a number, so none should reach a span either way
KEPT = os.getenv("TRACE_CALL_CONTENT") == "1"
# Words from the call: the caller's request, and the clinic's name from the bot's greeting and system prompt.
SAID = ("real person", "charge on my bill", "bright smile")
spans = []


def value(v):
    kind = v.WhichOneof("value")
    return [value(x) for x in v.array_value.values] if kind == "array_value" else getattr(v, kind)


async def receive(request):
    body = await request.read()
    if request.headers.get("Content-Encoding") == "gzip":
        body = gzip.decompress(body)
    export = ExportTraceServiceRequest.FromString(body)
    for resource in export.resource_spans:
        for scope in resource.scope_spans:
            for s in scope.spans:
                spans.append({"trace": s.trace_id.hex(), "start": s.start_time_unix_nano, "id": s.span_id.hex(), "parent": s.parent_span_id.hex(),
                              "name": s.name, "ms": (s.end_time_unix_nano - s.start_time_unix_nano) / 1e6,
                              "error": s.status.code == 2, "events": [e.name for e in s.events],
                              "attrs": {a.key: value(a.value) for a in s.attributes},
                              "event_attrs": [{a.key: value(a.value) for a in e.attributes} for e in s.events]})
    return web.Response(content_type="application/x-protobuf", body=b"")


def tree(trace):
    """Print one trace as an indented tree, siblings in start order."""
    children = defaultdict(list)
    for s in trace:
        children[s["parent"]].append(s)

    def show(s, depth):
        detail = {k: v for k, v in s["attrs"].items() if k in ("transfer.outcome", "tool.result", "metrics.ttfb",
                                                                "turn.number", "gen_ai.request.model")}
        print(f"    {'  ' * depth}{s['name']}  {s['ms']:.0f} ms  {detail or ''}{'  ERROR' if s['error'] else ''}"
              f"{'  events: ' + ', '.join(s['events']) if s['events'] else ''}")
        for c in children[s["id"]]:
            show(c, depth + 1)

    for root in [s for s in trace if s["parent"] not in {x["id"] for x in trace}]:
        show(root, 0)


def check_trace(trace, outcome):
    by_id = {s["id"]: s for s in trace}
    parent = lambda s: by_id.get(s["parent"], {}).get("name")  # noqa: E731
    named = lambda n: [s for s in trace if s["name"] == n]  # noqa: E731
    check(len(named("conversation")) == 1 and not named("conversation")[0]["parent"], "one conversation span, the root")
    turns = named("turn")
    check(turns and all(parent(t) == "conversation" for t in turns), f"{len(turns)} turns, all in the conversation")
    # What the bot says while the caller is on hold, and its handover line, is spoken outside any turn, so those
    # tts spans sit directly in the conversation.
    for service, homes in (("stt", ("turn",)), ("llm", ("turn",)), ("tts", ("turn", "conversation"))):
        found = named(service)
        check(found and all(parent(s) in homes for s in found), f"{service} spans sit inside {' or '.join(homes)}s",
              [parent(s) for s in found])
    tool = named("tool transfer_to_staff")
    check(len(tool) == 1 and parent(tool[0]) == "turn", "the transfer_to_staff call is a span in its turn")
    transfer = named("transfer")
    check(len(transfer) == 1 and parent(transfer[0]) == "tool transfer_to_staff",
          "the transfer is a span inside the tool call")
    if transfer:
        check(transfer[0]["attrs"].get("transfer.outcome") == outcome, f"the transfer's outcome is {outcome!r}",
              transfer[0]["attrs"])
    if tool:
        check(tool[0]["error"] == (outcome == "no staff"), "the tool span is an error only when nobody answered")
    mine = [s for s in trace if s["name"].startswith(("tool ", "transfer"))]
    check(not any(PHONE in str(s["attrs"]) or "summary" in s["attrs"] for s in mine),
          "the tool and transfer spans carry no caller details")
    heard = [(s["name"], k) for s in trace for k, v in [*s["attrs"].items(),
             *(kv for e in s["event_attrs"] for kv in e.items())] if any(w in str(v).lower() for w in SAID)]
    if KEPT:
        check(heard, "with TRACE_CALL_CONTENT=1, what was said is kept", "no span holds any of it")
    else:
        check(not heard, "no span holds anything said on the call", heard[:5])
        extra = sorted({k for s in trace for k in s["attrs"] if not telemetry.safe({k: 1})})
        check(not extra, "only allowlisted attributes reached the backend", extra)
        # Not every LLM span has them: one cut off by the caller hanging up never got a timing or a token count.
        llm = named("llm")
        whole = [s for s in llm if {"metrics.ttfb", "gen_ai.usage.input_tokens"} <= set(s["attrs"])]
        check(whole, f"timings and token counts are still there ({len(whole)} of {len(llm)} LLM spans have both)",
              [sorted(s["attrs"]) for s in llm])
    return transfer[0] if transfer else None


async def main():
    app = web.Application()
    app.router.add_post("/v1/traces", receive)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "localhost", 4318).start()

    for name, outcome in (("connected", "connected"), ("nobody", "no staff")):
        print(name)
        began = time.time_ns()
        await transfer_call.SCENARIOS[name]()
        await asyncio.sleep(8)  # the bot sends spans in batches every 5 s
        traces = defaultdict(list)
        for s in spans:
            traces[s["trace"]].append(s)
        # Calls that started during this scenario. A bot also retries spans a backend missed, so older calls
        # can turn up late.
        calls = [t for t in traces.values() if any(s["name"] == "conversation" and s["start"] > began for s in t)]
        check(len(calls) == 1, "the call arrived as one trace", f"{len(calls)} traces with a conversation span")
        if calls:
            tree(calls[0])
            check_trace(calls[0], outcome)

    await runner.cleanup()
    results = transfer_call.results
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    asyncio.run(main())
