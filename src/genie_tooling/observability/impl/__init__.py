"""Concrete implementations of InteractionTracerPlugin."""

from .console_tracer import ConsoleTracerPlugin

# from .otel_tracer import OpenTelemetryTracerPlugin # Example

__all__ = [
    "ConsoleTracerPlugin",
    # "OpenTelemetryTracerPlugin",
]
