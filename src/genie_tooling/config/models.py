import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

logger = logging.getLogger(__name__)

class MiddlewareConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    plugin_dev_dirs: List[str] = Field(default_factory=list)
    default_log_level: str = Field(default="INFO")

    default_llm_provider_id: Optional[str] = Field(default=None)
    llm_provider_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    default_command_processor_id: Optional[str] = Field(default=None)
    command_processor_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    default_tool_lookup_provider_id: Optional[str] = Field(default=None)
    default_tool_indexing_formatter_id: Optional[str] = Field(default=None)
    tool_lookup_provider_configurations: Dict[str, Dict[str, Any]] = Field( # ADDED
        default_factory=dict,
        description="Configuration for specific ToolLookupProvider plugins, keyed by plugin_id."
    )

    default_rag_loader_id: Optional[str] = Field(default=None)
    default_rag_splitter_id: Optional[str] = Field(default=None)
    default_rag_embedder_id: Optional[str] = Field(default=None)
    default_rag_vector_store_id: Optional[str] = Field(default=None)
    default_rag_retriever_id: Optional[str] = Field(default=None)

    # Configurations for specific plugin instances, keyed by their plugin_id
    # These allow providing detailed setup parameters for individual plugin implementations.
    document_loader_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    text_splitter_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    embedding_generator_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    vector_store_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    retriever_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    tool_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # Add other specific plugin type configurations as needed (e.g., cache_provider_configurations)

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
