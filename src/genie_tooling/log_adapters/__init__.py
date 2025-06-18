"""LogAdapter Abstractions and Implementations."""

from .abc import LogAdapter
from .impl import DefaultLogAdapter, PyviderTelemetryLogAdapter

__all__ = [
    "LogAdapter",
    "DefaultLogAdapter",
    "PyviderTelemetryLogAdapter",
]
