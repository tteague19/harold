"""OpenTelemetry and Arize Phoenix observability bootstrap.

Configures a global ``TracerProvider`` that exports spans to an Arize
Phoenix instance via OTLP HTTP. When enabled, every Pydantic AI agent
run, LLM call, and tool invocation is traced automatically.

Call ``setup_observability`` once at application startup, before any
agent runs. Requires the ``observability`` extra to be installed:
``uv sync --extra observability``.
"""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic_ai import Agent

from harold.config import HaroldSettings

logger = logging.getLogger(__name__)


def setup_observability(settings: HaroldSettings) -> None:
    """Configure OpenTelemetry tracing with Arize Phoenix as the backend.

    When ``settings.phoenix_enabled`` is ``False``, this function is a
    no-op. When enabled, it:

    1. Creates a ``TracerProvider`` with an OTLP HTTP exporter pointing
       at the configured Phoenix endpoint.
    2. Registers it as the global tracer provider.
    3. Instruments all Pydantic AI agents via ``Agent.instrument_all()``.

    Args:
        settings: Application configuration containing Phoenix toggle
            and endpoint URL.
    """
    if not settings.phoenix_enabled:
        return

    tracer_provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=settings.phoenix_endpoint)
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(tracer_provider)

    Agent.instrument_all()

    logger.info(
        "Observability enabled — exporting traces to %s",
        settings.phoenix_endpoint,
    )
