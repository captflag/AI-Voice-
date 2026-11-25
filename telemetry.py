"""Phase 8 step 3: OpenTelemetry traces of every call. Off unless OTEL_EXPORTER_OTLP_ENDPOINT is set.

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 python bot.py -t livekit

Pipecat already traces a call as a "conversation" span with a "turn" span per exchange, and stt, llm and tts
spans inside each turn that carry time to first byte, the model, token counts and the text. This file adds what
it does not cover: one span per tool call, and one for a transfer, which goes on after its turn has ended while
the caller is on hold. Any OTLP/HTTP backend works: Jaeger, SigNoz, Grafana Tempo.

What callers say stays out of the traces. Pipecat's spans record the transcripts, the text spoken, the messages
and system prompt sent to the model (which holds the caller's number) and tool arguments and results; a clinic's
calls are health records. Only the attributes listed in SAFE leave the process, so an attribute a later Pipecat
adds is dropped rather than leaked. TRACE_CALL_CONTENT=1 keeps everything, for debugging on your own machine.
"""
import dataclasses
import functools
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import StatusCode

tracer = trace.get_tracer("bright-smile")

# Timings, counts, models and outcomes: nothing a caller said.
SAFE = {
    "conversation.id", "conversation.type", "service.operation", "error.type", "is_final", "language",
    "voice_id", "vad_enabled", "stream", "transcript.is_input", "stt.incomplete", "llm.interrupted",
    "tts.interrupted", "tool.cancelled", "tool_count", "tool_choice", "tools.count", "tools.names",
    "audio.data_size_bytes", "modalities", "input_modality", "output_modality",
    "gen_ai.operation.name", "gen_ai.provider.name", "gen_ai.request.model", "gen_ai.output.type",
    "gen_ai.response.time_to_first_chunk", "gen_ai.tool.name", "gen_ai.tool.call.id",
    "staff",  # who took a transfer: a member of staff, not the caller
}
SAFE_PREFIXES = ("metrics.", "gen_ai.usage.", "turn.", "transfer.", "tool.")  # tool.*: this file's own spans


def safe(attributes):
    return {k: v for k, v in (attributes or {}).items() if k in SAFE or k.startswith(SAFE_PREFIXES)}


class Redacting(SpanExporter):
    """Passes spans on to another exporter with only the SAFE attributes, on the spans and on their events."""

    def __init__(self, inner):
        self.inner = inner

    def export(self, spans):
        return self.inner.export([ReadableSpan(
            name=s.name, context=s.context, parent=s.parent, resource=s.resource, attributes=safe(s.attributes),
            events=[Event(e.name, safe(e.attributes), e.timestamp) for e in s.events], links=s.links, kind=s.kind,
            status=s.status, start_time=s.start_time, end_time=s.end_time,
            instrumentation_scope=s.instrumentation_scope) for s in spans])

    def shutdown(self):
        self.inner.shutdown()

    def force_flush(self, timeout_millis=30000):
        return self.inner.force_flush(timeout_millis)


def setup():
    """Send spans to the OTLP endpoint in the environment, if there is one. Returns whether tracing is on."""
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from pipecat.utils.tracing.setup import setup_tracing

    exporter = OTLPSpanExporter()  # reads the endpoint itself
    if os.getenv("TRACE_CALL_CONTENT") != "1":
        exporter = Redacting(exporter)
    return setup_tracing("bright-smile-receptionist", exporter=exporter)


def traced(tool):
    """Wrap a tool so each call is a span inside the turn that made it. Pipecat reads the wrapped signature."""

    @functools.wraps(tool)
    async def run(params, **arguments):
        # ponytail: a private field, read the same way by Pipecat's own tracing decorators
        calls = getattr(params.llm, "_tracing_context", None)
        with tracer.start_as_current_span(f"tool {tool.__name__}",
                                          context=calls.get_turn_context() if calls else None) as span:
            span.set_attribute("tool.name", tool.__name__)
            span.set_attribute("tool.arguments", sorted(arguments))  # the names only: the values are names and numbers

            async def result_callback(result, **options):
                problem = result.get("error") if isinstance(result, dict) else None
                span.set_attribute("tool.result", "error" if problem else "ok")
                if problem:
                    span.set_status(StatusCode.ERROR, problem)
                await params.result_callback(result, **options)

            await tool(dataclasses.replace(params, result_callback=result_callback), **arguments)

    return run
