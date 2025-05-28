"""ErrorHandler Abstractions and Implementations."""

from .abc import ErrorHandler
from .impl import DefaultErrorHandler

__all__ = [
    "ErrorHandler",
    "DefaultErrorHandler",
]
