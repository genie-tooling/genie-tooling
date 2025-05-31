### src/genie_tooling/__init__.py
"""
My Agentic Middleware Library
-----------------------------

A hyper-pluggable Python middleware for Agentic AI and LLM applications.
Async-first for performance.
"""
__version__ = "0.1.0"

# Key exports for ease of use
from .cache_providers.abc import CacheProvider as CacheProviderPlugin
from .code_executors.abc import CodeExecutionResult
from .code_executors.abc import CodeExecutor as CodeExecutorPlugin
from .command_processors.abc import CommandProcessorPlugin
from .command_processors.types import CommandProcessorResponse
from .config.models import MiddlewareConfig
from .core.plugin_manager import PluginManager
from .core.types import (
    Chunk, Document, EmbeddingVector, Plugin, RetrievedChunk, StructuredError,
)
from .decorators import tool
from .definition_formatters.abc import DefinitionFormatter as DefinitionFormatterPlugin
from .document_loaders.abc import DocumentLoaderPlugin
from .embedding_generators.abc import EmbeddingGeneratorPlugin
from .error_formatters.abc import ErrorFormatter
from .error_handlers.abc import ErrorHandler
from .input_validators.abc import InputValidationException, InputValidator
from .invocation.invoker import ToolInvoker
from .invocation_strategies.abc import InvocationStrategy
from .invocation_strategies.impl.default_async import DefaultAsyncInvocationStrategy
from .llm_providers.abc import LLMProviderPlugin
from .llm_providers.types import (
    ChatMessage, LLMChatResponse, LLMCompletionResponse, LLMUsageInfo, ToolCall, ToolCallFunction,
)
from .log_adapters.abc import LogAdapter as LogAdapterPlugin
from .lookup.service import ToolLookupService
from .lookup.types import RankedToolResult
from .output_transformers.abc import OutputTransformer
from .rag.manager import RAGManager
from .redactors.abc import Redactor as RedactorPlugin
from .retrievers.abc import RetrieverPlugin
from .security.key_provider import KeyProvider
from .text_splitters.abc import TextSplitterPlugin
from .tool_lookup_providers.abc import ToolLookupProvider as ToolLookupProviderPlugin
from .tools.abc import Tool as ToolPlugin
from .tools.manager import ToolManager
from .vector_stores.abc import VectorStorePlugin

# P1.5 Exports (Observability, HITL, Token Usage, Guardrails)
from .observability.abc import InteractionTracerPlugin
from .observability.manager import InteractionTracingManager
from .observability.types import TraceEvent
from .hitl.abc import HumanApprovalRequestPlugin
from .hitl.manager import HITLManager
from .hitl.types import ApprovalRequest, ApprovalResponse, ApprovalStatus
from .token_usage.abc import TokenUsageRecorderPlugin
from .token_usage.manager import TokenUsageManager
from .token_usage.types import TokenUsageRecord
from .guardrails.abc import (
    GuardrailPlugin, InputGuardrailPlugin, OutputGuardrailPlugin, ToolUsageGuardrailPlugin,
)
from .guardrails.manager import GuardrailManager
from .guardrails.types import GuardrailAction, GuardrailViolation

# P1.5 Exports (Prompts, Conversation, LLM Output Parsing)
from .prompts.abc import PromptRegistryPlugin, PromptTemplatePlugin
from .prompts.manager import PromptManager
from .prompts.types import FormattedPrompt, PromptData, PromptIdentifier
from .prompts.conversation.impl.abc import ConversationStateProviderPlugin # CORRECTED
from .prompts.conversation.impl.manager import ConversationStateManager # CORRECTED
from .prompts.conversation.types import ConversationState # Path was correct
from .prompts.llm_output_parsers.abc import LLMOutputParserPlugin # Path was correct
from .prompts.llm_output_parsers.manager import LLMOutputParserManager # Path was correct
from .prompts.llm_output_parsers.types import ParsedOutput # Path was correct

# Interface Exports (from interfaces.py)
from .interfaces import (
    LLMInterface, RAGInterface, ObservabilityInterface, HITLInterface,
    UsageTrackingInterface, PromptInterface, ConversationInterface
)


__all__ = [
    "__version__", "MiddlewareConfig", "PluginManager", "Plugin", "Document", "Chunk", 
    "RetrievedChunk", "EmbeddingVector", "StructuredError", "KeyProvider", "ToolPlugin", 
    "ToolManager", "DefinitionFormatterPlugin", "ToolInvoker", "InvocationStrategy", 
    "DefaultAsyncInvocationStrategy", "InputValidator", "InputValidationException", 
    "OutputTransformer", "ErrorHandler", "ErrorFormatter", "RAGManager", 
    "DocumentLoaderPlugin", "TextSplitterPlugin", "EmbeddingGeneratorPlugin", 
    "VectorStorePlugin", "RetrieverPlugin", "ToolLookupService", "RankedToolResult", 
    "ToolLookupProviderPlugin", "CacheProviderPlugin", "LogAdapterPlugin", "RedactorPlugin", 
    "CodeExecutorPlugin", "CodeExecutionResult", "LLMProviderPlugin", "ChatMessage", 
    "LLMChatResponse", "LLMCompletionResponse", "LLMUsageInfo", "ToolCall", "ToolCallFunction", 
    "CommandProcessorPlugin", "CommandProcessorResponse", "tool",
    # P1.5 (Observability, HITL, Token Usage, Guardrails)
    "InteractionTracerPlugin", "InteractionTracingManager", "TraceEvent",
    "HumanApprovalRequestPlugin", "HITLManager", "ApprovalRequest", "ApprovalResponse", "ApprovalStatus",
    "TokenUsageRecorderPlugin", "TokenUsageManager", "TokenUsageRecord",
    "GuardrailPlugin", "InputGuardrailPlugin", "OutputGuardrailPlugin", "ToolUsageGuardrailPlugin",
    "GuardrailManager", "GuardrailAction", "GuardrailViolation",
    # P1.5 (Prompts, Conversation, LLM Output Parsing)
    "PromptManager", "PromptRegistryPlugin", "PromptTemplatePlugin", "FormattedPrompt", "PromptData", "PromptIdentifier",
    "ConversationStateManager", "ConversationStateProviderPlugin", "ConversationState",
    "LLMOutputParserManager", "LLMOutputParserPlugin", "ParsedOutput",
    # Interfaces
    "LLMInterface", "RAGInterface", "ObservabilityInterface", "HITLInterface",
    "UsageTrackingInterface", "PromptInterface", "ConversationInterface",
]

import logging
_logger = logging.getLogger(__name__)
if not _logger.hasHandlers():
    _logger.addHandler(logging.NullHandler())