"""Concrete implementations of LogAdapterPlugins."""
from .default_adapter import DefaultLogAdapter

# Example: from .opentelemetry_adapter import OpenTelemetryLogAdapter

__all__ = ["DefaultLogAdapter"]
