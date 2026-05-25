"""LogAdapter Abstractions and Implementations."""

from .abc import LogAdapter
from .impl import DefaultLogAdapter, PyviderTelemetryLogAdapter

__all__ = [
    "DefaultLogAdapter",
    "LogAdapter",
    "PyviderTelemetryLogAdapter",
]
