"""
My Agentic Middleware Library
-----------------------------

A hyper-pluggable Python middleware for Agentic AI and LLM applications.
Async-first for performance.
"""
__version__ = "0.1.0"

# Key exports for ease of use
from .cache_providers.abc import CacheProvider as CacheProviderPlugin # Updated
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
from .code_executors.abc import CodeExecutionResult # Updated
from .code_executors.abc import CodeExecutor as CodeExecutorPlugin # Updated
from .error_formatters.abc import ErrorFormatter # Updated
from .error_handlers.abc import ErrorHandler # Updated
from .invocation.invoker import ToolInvoker
from .invocation_strategies.abc import InvocationStrategy # Updated
from .output_transformers.abc import OutputTransformer # Updated
from .input_validators.abc import InputValidationException, InputValidator # Updated
from .log_adapters.abc import LogAdapter as LogAdapterPlugin # Updated
from .redactors.abc import Redactor as RedactorPlugin # Updated
from .tool_lookup_providers.abc import ToolLookupProvider as ToolLookupProviderPlugin # Updated
from .lookup.service import ToolLookupService
from .lookup.types import RankedToolResult
from .rag.manager import RAGManager
# Updated RAG plugin ABC imports
from .document_loaders.abc import DocumentLoaderPlugin
from .embedding_generators.abc import EmbeddingGeneratorPlugin
from .retrievers.abc import RetrieverPlugin
from .text_splitters.abc import TextSplitterPlugin
from .vector_stores.abc import VectorStorePlugin
from .security.key_provider import KeyProvider
from .tools.abc import Tool as ToolPlugin
from .definition_formatters.abc import DefinitionFormatter as DefinitionFormatterPlugin # Updated
from .tools.manager import ToolManager

__all__ = [
    "__version__",
    "MiddlewareConfig",
    "PluginManager",
    "Plugin", "Document", "Chunk", "RetrievedChunk", "EmbeddingVector", "StructuredError",
    "KeyProvider",
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
]

import logging

# Configure a basic null handler to prevent "No handler found" warnings
# if the consuming application doesn't configure logging for the library's logger.
# The consuming app can still configure its own handlers for 'genie_tooling' or root.
_logger = logging.getLogger(__name__)
if not _logger.hasHandlers():
    _logger.addHandler(logging.NullHandler())

# Optional: A simple way for consuming apps to enable basic console logging for the library
# during development, if they haven't set up their own logging.
# def enable_library_default_logging(level: int = logging.INFO) -> None:
#     """Enables basic console logging for the library. For development/debugging."""
#     console_handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     console_handler.setFormatter(formatter)
#     # Remove NullHandler if it was added
#     for h in _logger.handlers[:]:
#         if isinstance(h, logging.NullHandler):
#             _logger.removeHandler(h)
#     if not _logger.handlers: # Add console handler only if no other handlers exist now
#         _logger.addHandler(console_handler)
#     _logger.setLevel(level)
