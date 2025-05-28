"""
My Agentic Middleware Library
-----------------------------

A hyper-pluggable Python middleware for Agentic AI and LLM applications.
Async-first for performance.
"""
__version__ = "0.1.0"

# Key exports for ease of use
from .caching.abc import CacheProvider as CacheProviderPlugin
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
from .executors.abc import CodeExecutionResult
from .executors.abc import CodeExecutor as CodeExecutorPlugin
from .invocation.errors import ErrorFormatter, ErrorHandler
from .invocation.invoker import ToolInvoker
from .invocation.strategies.abc import InvocationStrategy
from .invocation.transformation import OutputTransformer
from .invocation.validation import InputValidationException, InputValidator
from .logging_monitoring.abc import LogAdapter as LogAdapterPlugin
from .logging_monitoring.abc import Redactor as RedactorPlugin
from .lookup.providers.abc import ToolLookupProvider as ToolLookupProviderPlugin
from .lookup.service import ToolLookupService
from .lookup.types import RankedToolResult
from .rag.manager import RAGManager
from .rag.plugins.abc import (
    DocumentLoaderPlugin,
    EmbeddingGeneratorPlugin,
    RetrieverPlugin,
    TextSplitterPlugin,
    VectorStorePlugin,
)
from .security.key_provider import KeyProvider
from .tools.abc import Tool as ToolPlugin
from .tools.formatters.abc import DefinitionFormatter as DefinitionFormatterPlugin
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
