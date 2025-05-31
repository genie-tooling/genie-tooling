# src/genie_tooling/prompts/conversation/impl/__init__.py
"""Conversation State Management Abstractions and Implementations."""
from .abc import ConversationStateProviderPlugin
from .manager import ConversationStateManager
from ..types import ConversationState # This import is correct as per existing structure

# Add new imports for concrete implementations
from .in_memory_state_provider import InMemoryStateProviderPlugin
from .redis_state_provider import RedisStateProviderPlugin

__all__ = [
    "ConversationStateProviderPlugin",
    "ConversationStateManager",
    "ConversationState", # Keep existing export
    "InMemoryStateProviderPlugin", # Add new export
    "RedisStateProviderPlugin",    # Add new export
]
