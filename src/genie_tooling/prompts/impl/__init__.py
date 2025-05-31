# src/genie_tooling/prompts/impl/__init__.py
"""Concrete implementations of PromptRegistryPlugin and PromptTemplatePlugin."""
from .basic_string_format_template import BasicStringFormatTemplatePlugin
from .file_system_prompt_registry import FileSystemPromptRegistryPlugin
from .jinja2_chat_template import Jinja2ChatTemplatePlugin

__all__ = [
    "FileSystemPromptRegistryPlugin",
    "BasicStringFormatTemplatePlugin",
    "Jinja2ChatTemplatePlugin",
]
