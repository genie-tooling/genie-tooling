### src/genie_tooling/config/models.py
# src/genie_tooling/config/models.py
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, PrivateAttr

logger = logging.getLogger(__name__)

class MiddlewareConfig(BaseModel):
    """
    Configuration for the Genie Tooling Middleware.
    This object is typically provided by the consuming application.
    """
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    plugin_dev_dirs: List[str] = Field(
        default_factory=list,
        description="Directories to scan for development plugins. Paths should be absolute or relative to CWD."
    )
    default_log_level: str = Field(
        default="INFO",
        description="Default logging level for middleware components (e.g., INFO, DEBUG, WARNING)."
    )

    # LLM Provider configurations
    default_llm_provider_id: Optional[str] = Field(
        default=None,
        description="Default LLM provider plugin ID to use if not specified in calls."
    )
    llm_provider_configurations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration for specific LLM provider plugins, keyed by plugin_id."
    )

    # Command Processor configurations
    default_command_processor_id: Optional[str] = Field(
        default=None,
        description="Default command processor plugin ID to use if not specified in calls."
    )
    command_processor_configurations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration for specific command processor plugins, keyed by plugin_id."
    )

    # Tool Lookup configurations
    default_tool_lookup_provider_id: Optional[str] = Field(
        default=None, # Example: "embedding_similarity_lookup_v1"
        description="Default tool lookup provider plugin ID."
    )
    default_tool_indexing_formatter_id: Optional[str] = Field(
        default=None, # Example: "llm_compact_text_v1"
        description="Default formatter ID used by ToolLookupService for indexing tools."
    )
    # tool_lookup_provider_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict) # Future

    # RAG Component Default IDs
    default_rag_loader_id: Optional[str] = Field(
        default=None, # Example: "file_system_loader_v1"
        description="Default RAG document loader plugin ID for indexing operations."
    )
    default_rag_splitter_id: Optional[str] = Field(
        default=None, # Example: "character_recursive_text_splitter_v1"
        description="Default RAG text splitter plugin ID for indexing operations."
    )
    default_rag_embedder_id: Optional[str] = Field(
        default=None, # Example: "sentence_transformer_embedder_v1"
        description="Default RAG embedding generator plugin ID for indexing and retrieval."
    )
    default_rag_vector_store_id: Optional[str] = Field(
        default=None, # Example: "faiss_vector_store_v1"
        description="Default RAG vector store plugin ID for indexing and retrieval."
    )
    default_rag_retriever_id: Optional[str] = Field(
        default=None, # Example: "basic_similarity_retriever_v1"
        description="Default RAG retriever plugin ID for search operations."
    )
    # rag_component_configurations: Dict[str, Dict[str, Any]] = Field(default_factory=dict) # Future

    # Tool configurations (e.g., for specific tool plugins if they need setup config)
    tool_configurations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration for specific tool plugins, keyed by plugin_id (not tool identifier)."
    )
    
    # Internal attribute to hold a reference to the Genie instance if needed by sub-components
    # This is not part of the persisted config schema.
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