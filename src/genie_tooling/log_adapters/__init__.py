"""LogAdapter Abstractions and Implementations."""

from .abc import LogAdapter
from .impl import DefaultLogAdapter

__all__ = [
    "LogAdapter",
    "DefaultLogAdapter",
]
