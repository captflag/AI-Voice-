"""Phase 8 step 3: OpenTelemetry traces of every call. Off unless OTEL_EXPORTER_OTLP_ENDPOINT is set.

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 python bot.py -t livekit

Pipecat already traces a call as a "conversation" span with a "turn" span per exchange, and stt, llm and tts
spans inside each turn that carry time to first byte, the model, token counts and the text. This file adds what
it does not cover: one span per tool call, and one for a transfer, which goes on after its turn has ended while
the caller is on hold. Any OTLP/HTTP backend works: Jaeger, SigNoz, Grafana Tempo.
"""
import dataclasses
import functools
import os

from opentelemetry import trace
from opentelemetry.trace import StatusCode

tracer = trace.get_tracer("bright-smile")


def setup():
    """Send spans to the OTLP endpoint in the environment, if there is one. Returns whether tracing is on."""
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from pipecat.utils.tracing.setup import setup_tracing

    return setup_tracing("bright-smile-receptionist", exporter=OTLPSpanExporter())  # reads the endpoint itself


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
