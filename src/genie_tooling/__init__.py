"""
My Agentic Middleware Library
-----------------------------

A hyper-pluggable Python middleware for Agentic AI and LLM applications.
Async-first for performance.
"""
__version__ = "0.1.0"

# Key exports for ease of use
# Updated import structure for plugin ABCs
from .cache_providers.abc import CacheProvider as CacheProviderPlugin
from .code_executors.abc import CodeExecutionResult
from .code_executors.abc import CodeExecutor as CodeExecutorPlugin
from .command_processors.abc import CommandProcessorPlugin
from .command_processors.types import CommandProcessorResponse
from .config.models import MiddlewareConfig
from .core.plugin_manager import PluginManager
from .core.types import (
    Chunk,
    Document,
    EmbeddingVector,
    Plugin,
    RetrievedChunk,
    StructuredError,
)
from .definition_formatters.abc import DefinitionFormatter as DefinitionFormatterPlugin
from .document_loaders.abc import DocumentLoaderPlugin
from .embedding_generators.abc import EmbeddingGeneratorPlugin
from .error_formatters.abc import ErrorFormatter
from .error_handlers.abc import ErrorHandler
from .input_validators.abc import InputValidationException, InputValidator
from .invocation.invoker import ToolInvoker  # Invoker is not a plugin type itself
from .invocation_strategies.abc import InvocationStrategy
from .llm_providers.abc import LLMProviderPlugin
from .llm_providers.types import (
    ChatMessage,
    LLMChatResponse,
    LLMCompletionResponse,
    LLMUsageInfo,
    ToolCall,
    ToolCallFunction,
)
from .log_adapters.abc import LogAdapter as LogAdapterPlugin
from .lookup.service import ToolLookupService  # Service, not a plugin type
from .lookup.types import RankedToolResult
from .output_transformers.abc import OutputTransformer
from .rag.manager import RAGManager  # Manager, not a plugin type
from .redactors.abc import Redactor as RedactorPlugin
from .retrievers.abc import RetrieverPlugin
from .security.key_provider import KeyProvider  # Protocol, not a plugin
from .text_splitters.abc import TextSplitterPlugin
from .tool_lookup_providers.abc import ToolLookupProvider as ToolLookupProviderPlugin
from .tools.abc import Tool as ToolPlugin
from .tools.manager import ToolManager  # Manager, not a plugin type
from .vector_stores.abc import VectorStorePlugin

__all__ = [
    "__version__",
    "MiddlewareConfig",
    "PluginManager",
    "Plugin", "Document", "Chunk", "RetrievedChunk", "EmbeddingVector", "StructuredError",
    "KeyProvider", # Protocol
    "ToolPlugin", "ToolManager", "DefinitionFormatterPlugin",
    "ToolInvoker", "InvocationStrategy",
    "InputValidator", "InputValidationException", "OutputTransformer",
    "ErrorHandler", "ErrorFormatter",
    "RAGManager",
    "DocumentLoaderPlugin", "TextSplitterPlugin", "EmbeddingGeneratorPlugin",
    "VectorStorePlugin", "RetrieverPlugin",
    "ToolLookupService", "RankedToolResult", "ToolLookupProviderPlugin",
    "CacheProviderPlugin",
    "LogAdapterPlugin", "RedactorPlugin",
    "CodeExecutorPlugin", "CodeExecutionResult",
    "LLMProviderPlugin", "ChatMessage", "LLMChatResponse", "LLMCompletionResponse",
    "LLMUsageInfo", "ToolCall", "ToolCallFunction",
    "CommandProcessorPlugin", "CommandProcessorResponse"
]

import logging

_logger = logging.getLogger(__name__)
if not _logger.hasHandlers():
    _logger.addHandler(logging.NullHandler())
