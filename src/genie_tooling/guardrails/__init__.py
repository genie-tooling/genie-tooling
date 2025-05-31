"""Guardrail Abstractions and Implementations."""

from .abc import (
    GuardrailPlugin,
    InputGuardrailPlugin,
    OutputGuardrailPlugin,
    ToolUsageGuardrailPlugin,
)
from .manager import GuardrailManager
from .types import GuardrailAction, GuardrailViolation

# Concrete implementations will be registered via entry points
# from .impl.keyword_blocklist import KeywordBlocklistGuardrailPlugin

__all__ = [
    "GuardrailPlugin",
    "InputGuardrailPlugin",
    "OutputGuardrailPlugin",
    "ToolUsageGuardrailPlugin",
    "GuardrailManager",
    "GuardrailAction",
    "GuardrailViolation",
    # "KeywordBlocklistGuardrailPlugin",
]
