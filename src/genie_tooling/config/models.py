### src/genie_tooling/config/models.py
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from .features import FeatureSettings

logger = logging.getLogger(__name__)

class MiddlewareConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    features: FeatureSettings = Field(
        default_factory=FeatureSettings,
        description="High-level feature configuration settings."
    )
    plugin_dev_dirs: List[str] = Field(default_factory=list)
    default_log_level: str = Field(default="INFO")
    key_provider_id: Optional[str] = Field(
        default=None,
        description="Plugin ID of the KeyProvider to use if no instance is passed to Genie.create."
    )
    default_llm_provider_id: Optional[str] = Field(default=None)
    llm_provider_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    default_command_processor_id: Optional[str] = Field(default=None)
    command_processor_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    default_tool_lookup_provider_id: Optional[str] = Field(default=None)
    default_tool_indexing_formatter_id: Optional[str] = Field(default=None)
    tool_lookup_provider_configurations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration for specific ToolLookupProvider plugins, keyed by plugin_id."
    )

    default_rag_loader_id: Optional[str] = Field(default=None)
    default_rag_splitter_id: Optional[str] = Field(default=None)
    default_rag_embedder_id: Optional[str] = Field(default=None)
    default_rag_vector_store_id: Optional[str] = Field(default=None)
    default_rag_retriever_id: Optional[str] = Field(default=None)
    document_loader_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    text_splitter_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    embedding_generator_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    vector_store_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    retriever_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    tool_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    cache_provider_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    default_observability_tracer_id: Optional[str] = Field(
        default=None, description="Default InteractionTracerPlugin ID."
    )
    observability_tracer_configurations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Configurations for InteractionTracerPlugins. For 'otel_tracer_plugin_v1', "
            "keys can include: 'otel_service_name', 'otel_service_version', "
            "'exporter_type' ('console', 'otlp_http', 'otlp_grpc'), "
            "'otlp_http_endpoint', 'otlp_http_headers', 'otlp_http_timeout', "
            "'otlp_grpc_endpoint', 'otlp_grpc_insecure', 'otlp_grpc_timeout', "
            "'resource_attributes' (dict)."
        )
    )

    default_hitl_approver_id: Optional[str] = Field(
        default=None, description="Default HumanApprovalRequestPlugin ID."
    )
    hitl_approver_configurations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Configurations for HumanApprovalRequestPlugins."
    )

    default_token_usage_recorder_id: Optional[str] = Field(
        default=None, description="Default TokenUsageRecorderPlugin ID."
    )
    token_usage_recorder_configurations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Configurations for TokenUsageRecorderPlugins."
    )

    default_input_guardrail_ids: List[str] = Field(
        default_factory=list, description="List of default InputGuardrailPlugin IDs."
    )
    default_output_guardrail_ids: List[str] = Field(
        default_factory=list, description="List of default OutputGuardrailPlugin IDs."
    )
    default_tool_usage_guardrail_ids: List[str] = Field(
        default_factory=list, description="List of default ToolUsageGuardrailPlugin IDs."
    )
    guardrail_configurations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Configurations for GuardrailPlugins."
    )

    default_prompt_registry_id: Optional[str] = Field(
        default=None, description="Default PromptRegistryPlugin ID."
    )
    prompt_registry_configurations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Configurations for PromptRegistryPlugins."
    )
    default_prompt_template_plugin_id: Optional[str] = Field(
        default=None, description="Default PromptTemplatePlugin ID."
    )
    prompt_template_configurations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Configurations for PromptTemplatePlugins."
    )
    default_conversation_state_provider_id: Optional[str] = Field(
        default=None, description="Default ConversationStateProviderPlugin ID."
    )
    conversation_state_provider_configurations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Configurations for ConversationStateProviderPlugins."
    )
    default_llm_output_parser_id: Optional[str] = Field(
        default=None, description="Default LLMOutputParserPlugin ID."
    )
    llm_output_parser_configurations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Configurations for LLMOutputParserPlugins."
    )

    # P2.5.D: Distributed Task Queue Configuration
    default_distributed_task_queue_id: Optional[str] = Field(
        default=None, description="Default DistributedTaskQueuePlugin ID."
    )
    distributed_task_queue_configurations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Configurations for DistributedTaskQueuePlugins."
    )

    _genie_instance: Optional[Any] = PrivateAttr(default=None)

    @field_validator("default_log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        log_level_upper = value.upper()
        if log_level_upper not in valid_levels:
            logger.warning(f"Invalid log_level '{value}' in MiddlewareConfig. Defaulting to INFO.")
            return "INFO"
        return log_level_upper
