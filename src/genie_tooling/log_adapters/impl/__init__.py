"""Implementations of LogAdapter."""
from .default_adapter import DefaultLogAdapter
from .pyvider_telemetry_adapter import PyviderTelemetryLogAdapter

__all__ = ["DefaultLogAdapter", "PyviderTelemetryLogAdapter"]
