# src/genie_tooling/prompts/__init__.py
"""Prompt Management Abstractions and Implementations."""
from .abc import PromptRegistryPlugin, PromptTemplatePlugin
from .manager import PromptManager
from .types import FormattedPrompt, PromptData, PromptIdentifier

__all__ = [
    "PromptRegistryPlugin",
    "PromptTemplatePlugin",
    "PromptManager",
    "FormattedPrompt",
    "PromptData",
    "PromptIdentifier",
]