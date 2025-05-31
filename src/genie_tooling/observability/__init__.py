"""Observability Abstractions and Implementations."""

from .abc import InteractionTracerPlugin
from .manager import InteractionTracingManager
from .types import TraceEvent

# Concrete implementations will be registered via entry points
# from .impl.console_tracer import ConsoleTracerPlugin
# from .impl.otel_tracer import OpenTelemetryTracerPlugin # Example

__all__ = [
    "InteractionTracerPlugin",
    "InteractionTracingManager",
    "TraceEvent",
    # "ConsoleTracerPlugin",
    # "OpenTelemetryTracerPlugin",
]
