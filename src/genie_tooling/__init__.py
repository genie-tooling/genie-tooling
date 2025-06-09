### src/genie_tooling/__init__.py
"""
genie-tooling
-----------------------------

A hyper-pluggable Python middleware for Agentic AI and LLM applications.
Async-first for performance.
"""
__version__ = "0.1.0"

# Key exports for ease of use
from .agents.base_agent import BaseAgent
from .agents.plan_and_execute_agent import PlanAndExecuteAgent
from .agents.react_agent import ReActAgent
from .agents.types import AgentOutput, PlannedStep, ReActObservation
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
from .decorators import tool
from .definition_formatters.abc import DefinitionFormatter as DefinitionFormatterPlugin
from .document_loaders.abc import DocumentLoaderPlugin
from .embedding_generators.abc import EmbeddingGeneratorPlugin
from .error_formatters.abc import ErrorFormatter
from .error_handlers.abc import ErrorHandler
from .guardrails.abc import (
    GuardrailPlugin,
    InputGuardrailPlugin,
    OutputGuardrailPlugin,
    ToolUsageGuardrailPlugin,
)
from .guardrails.manager import GuardrailManager
from .guardrails.types import GuardrailAction, GuardrailViolation
from .hitl.abc import HumanApprovalRequestPlugin
from .hitl.manager import HITLManager
from .hitl.types import ApprovalRequest, ApprovalResponse, ApprovalStatus
from .input_validators.abc import InputValidationException, InputValidator
from .interfaces import (
    ConversationInterface,
    HITLInterface,
    LLMInterface,
    ObservabilityInterface,
    PromptInterface,
    RAGInterface,
    TaskQueueInterface,  # Added for P2.5.D
    UsageTrackingInterface,
)
from .invocation.invoker import ToolInvoker
from .invocation_strategies.abc import InvocationStrategy
from .invocation_strategies.impl.default_async import DefaultAsyncInvocationStrategy

# P2.5.D: Import new distributed task strategy
from .invocation_strategies.impl.distributed_task_strategy import (
    DistributedTaskInvocationStrategy,
)
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
from .lookup.service import ToolLookupService
from .lookup.types import RankedToolResult
from .observability.abc import InteractionTracerPlugin
from .observability.manager import InteractionTracingManager
from .observability.types import TraceEvent
from .output_transformers.abc import OutputTransformer
from .prompts.abc import PromptRegistryPlugin, PromptTemplatePlugin
from .prompts.conversation.impl.abc import ConversationStateProviderPlugin
from .prompts.conversation.impl.manager import ConversationStateManager
from .prompts.conversation.types import ConversationState
from .prompts.llm_output_parsers.abc import LLMOutputParserPlugin
from .prompts.llm_output_parsers.manager import LLMOutputParserManager
from .prompts.llm_output_parsers.types import ParsedOutput
from .prompts.manager import PromptManager
from .prompts.types import FormattedPrompt, PromptData, PromptIdentifier
from .rag.manager import RAGManager
from .redactors.abc import Redactor as RedactorPlugin
from .retrievers.abc import RetrieverPlugin
from .security.key_provider import KeyProvider
from .task_queues.abc import DistributedTaskQueuePlugin, TaskStatus  # Added for P2.5.D
from .task_queues.manager import DistributedTaskQueueManager  # Added for P2.5.D
from .text_splitters.abc import TextSplitterPlugin
from .token_usage.abc import TokenUsageRecorderPlugin
from .token_usage.manager import TokenUsageManager
from .token_usage.types import TokenUsageRecord
from .tool_lookup_providers.abc import ToolLookupProvider as ToolLookupProviderPlugin
from .tools.abc import Tool as ToolPlugin
from .tools.manager import ToolManager
from .vector_stores.abc import VectorStorePlugin

__all__ = [
    "__version__", "MiddlewareConfig", "PluginManager", "Plugin", "Document", "Chunk",
    "RetrievedChunk", "EmbeddingVector", "StructuredError", "KeyProvider", "ToolPlugin",
    "ToolManager", "DefinitionFormatterPlugin", "ToolInvoker", "InvocationStrategy",
    "DefaultAsyncInvocationStrategy", "DistributedTaskInvocationStrategy", # Added for P2.5.D
    "InputValidator", "InputValidationException",
    "OutputTransformer", "ErrorHandler", "ErrorFormatter", "RAGManager",
    "DocumentLoaderPlugin", "TextSplitterPlugin", "EmbeddingGeneratorPlugin",
    "VectorStorePlugin", "RetrieverPlugin", "ToolLookupService", "RankedToolResult",
    "ToolLookupProviderPlugin", "CacheProviderPlugin", "LogAdapterPlugin", "RedactorPlugin",
    "CodeExecutorPlugin", "CodeExecutionResult", "LLMProviderPlugin", "ChatMessage",
    "LLMChatResponse", "LLMCompletionResponse", "LLMUsageInfo", "ToolCall", "ToolCallFunction",
    "CommandProcessorPlugin", "CommandProcessorResponse", "tool",
    "InteractionTracerPlugin", "InteractionTracingManager", "TraceEvent",
    "HumanApprovalRequestPlugin", "HITLManager", "ApprovalRequest", "ApprovalResponse", "ApprovalStatus",
    "TokenUsageRecorderPlugin", "TokenUsageManager", "TokenUsageRecord",
    "GuardrailPlugin", "InputGuardrailPlugin", "OutputGuardrailPlugin", "ToolUsageGuardrailPlugin",
    "GuardrailManager", "GuardrailAction", "GuardrailViolation",
    "PromptManager", "PromptRegistryPlugin", "PromptTemplatePlugin", "FormattedPrompt", "PromptData", "PromptIdentifier",
    "ConversationStateManager", "ConversationStateProviderPlugin", "ConversationState",
    "LLMOutputParserManager", "LLMOutputParserPlugin", "ParsedOutput",
    "LLMInterface", "RAGInterface", "ObservabilityInterface", "HITLInterface",
    "UsageTrackingInterface", "PromptInterface", "ConversationInterface", "TaskQueueInterface", # Added TaskQueueInterface
    "BaseAgent", "ReActAgent", "PlanAndExecuteAgent", "AgentOutput", "PlannedStep", "ReActObservation",
    "DistributedTaskQueuePlugin", "DistributedTaskQueueManager", "TaskStatus", # Added for P2.5.D
]

import logging

_logger = logging.getLogger(__name__)
if not _logger.hasHandlers():
    _logger.addHandler(logging.NullHandler())
