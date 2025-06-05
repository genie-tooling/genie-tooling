### src/genie_tooling/llm_providers/__init__.py
# src/genie_tooling/llm_providers/__init__.py
"""
LLM Provider plugins: Abstract interfaces, concrete implementations, and type definitions
for interacting with various Large Language Models.
"""
from .abc import LLMProviderPlugin

# Optionally, expose common concrete implementations if desired for direct import,
# though often these are loaded via plugin ID.
# from .impl.openai_provider import OpenAILLMProviderPlugin (if/when implemented)
from .impl.gemini_provider import GeminiLLMProviderPlugin
from .impl.llama_cpp_internal_provider import LlamaCppInternalLLMProviderPlugin  # Added
from .impl.llama_cpp_provider import LlamaCppLLMProviderPlugin
from .impl.ollama_provider import OllamaLLMProviderPlugin
from .manager import LLMProviderManager
from .types import (
    ChatMessage,
    LLMChatResponse,
    LLMCompletionResponse,
    LLMUsageInfo,
    ToolCall,
    ToolCallFunction,
)

__all__ = [
    "LLMProviderPlugin",
    "LLMProviderManager",
    "ChatMessage",
    "LLMChatResponse",
    "LLMCompletionResponse",
    "LLMUsageInfo",
    "ToolCall",
    "ToolCallFunction",
    # Concrete Implementations
    "OllamaLLMProviderPlugin",
    "GeminiLLMProviderPlugin",
    "LlamaCppLLMProviderPlugin",
    "LlamaCppInternalLLMProviderPlugin", # Added
    # "OpenAILLMProviderPlugin",
]

###<END-OF-FILE>###
