### src/genie_tooling/config/features.py
# src/genie_tooling/config/features.py
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class FeatureSettings(BaseModel):
    """
    Defines high-level, user-friendly feature toggles and default choices
    for initializing the Genie middleware. This model is processed by
    ConfigResolver to populate the more detailed MiddlewareConfig.
    """

    # LLM Feature
    llm: Literal["ollama", "openai", "gemini", "llama_cpp", "llama_cpp_internal", "none"] = Field(
        default="none", description="Primary LLM provider choice."
    )
    llm_ollama_model_name: Optional[str] = Field(
        default="mistral:latest", description="Default model for Ollama if 'ollama' is chosen for llm."
    )
    llm_openai_model_name: Optional[str] = Field(
        default="gpt-3.5-turbo", description="Default model for OpenAI if 'openai' is chosen for llm."
    )
    llm_gemini_model_name: Optional[str] = Field(
        default="gemini-1.5-flash-latest", description="Default model for Gemini if 'gemini' is chosen for llm."
    )
    llm_llama_cpp_model_name: Optional[str] = Field(
        default="mistral:latest", description="Default model for llama.cpp server if 'llama_cpp' is chosen for llm."
    )
    llm_llama_cpp_base_url: Optional[str] = Field(
        default="http://localhost:8080", description="Base URL for llama.cpp server if 'llama_cpp' is chosen for llm."
    )
    llm_llama_cpp_api_key_name: Optional[str] = Field(
        default=None, description="Optional environment variable name for llama.cpp server API key."
    )
    llm_llama_cpp_internal_model_path: Optional[str] = Field(
        default=None, description="Path to GGUF model file for internal Llama.cpp provider."
    )
    llm_llama_cpp_internal_n_gpu_layers: int = Field(
        default=0, description="Number of GPU layers for internal Llama.cpp."
    )
    llm_llama_cpp_internal_n_ctx: int = Field(
        default=2048, description="Context size for internal Llama.cpp."
    )
    llm_llama_cpp_internal_chat_format: Optional[str] = Field(
        default=None, description="Chat format for internal Llama.cpp (e.g., 'llama-2', 'mistral')."
    )
    llm_llama_cpp_internal_model_name_for_logging: Optional[str] = Field(
        default=None, description="Optional model name to use for logging/token tracking for internal Llama.cpp."
    )


    # Cache Feature
    cache: Literal["in-memory", "redis", "none"] = Field(
        default="none", description="Caching provider choice."
    )
    cache_redis_url: Optional[str] = Field(
        default="redis://localhost:6379/0", description="URL for Redis if 'redis' is chosen for cache."
    )

    # RAG Embedder Feature
    rag_embedder: Literal["sentence_transformer", "openai", "none"] = Field(
        default="none", description="Embedding generator for RAG."
    )
    rag_embedder_st_model_name: Optional[str] = Field(
        default="all-MiniLM-L6-v2", description="Model for SentenceTransformer if chosen for rag_embedder."
    )

    # RAG Vector Store Feature
    rag_vector_store: Literal["faiss", "chroma", "qdrant", "none"] = Field(
        default="none", description="Vector store for RAG."
    )
    rag_vector_store_chroma_path: Optional[str] = Field(
        default=None, description="Path for ChromaDB if 'chroma' is chosen for rag_vector_store. If None, plugin uses its default path logic."
    )
    rag_vector_store_chroma_collection_name: Optional[str] = Field(
        default="genie_rag_collection", description="Default collection name for ChromaDB in RAG."
    )
    rag_vector_store_qdrant_url: Optional[str] = Field(default=None, description="URL for Qdrant if 'qdrant' is chosen (e.g., http://localhost:6333).")
    rag_vector_store_qdrant_path: Optional[str] = Field(default=None, description="Path for local Qdrant if 'qdrant' is chosen.")
    rag_vector_store_qdrant_api_key_name: Optional[str] = Field(default=None, description="API key name for Qdrant.")
    rag_vector_store_qdrant_collection_name: Optional[str] = Field(default="genie_qdrant_rag", description="Default collection name for Qdrant in RAG.")
    rag_vector_store_qdrant_embedding_dim: Optional[int] = Field(default=None, description="Embedding dimension for Qdrant collection (required if creating).")

    # Tool Lookup Feature
    tool_lookup: Literal["embedding", "keyword", "hybrid", "none"] = Field(
        default="none", description="Method for looking up tools by natural language."
    )
    tool_lookup_top_k: Optional[int] = Field(
        default=5, description="Default number of tools to retrieve from lookup for LLM consideration."
    )
    tool_lookup_formatter_id_alias: Optional[str] = Field(
        default="compact_text_formatter", description="Alias for the formatter used to prepare tool definitions for indexing (e.g., 'compact_text_formatter')."
    )
    tool_lookup_chroma_path: Optional[str] = Field(
        default=None, description="Path for ChromaDB if 'embedding' tool_lookup uses Chroma. If None, plugin uses its default."
    )
    tool_lookup_chroma_collection_name: Optional[str] = Field(
        default="genie_tool_lookup_embeddings", description="Default collection for tool lookup embeddings in ChromaDB."
    )
    tool_lookup_embedder_id_alias: Optional[str] = Field(
        default="st_embedder", description="Alias for the embedder *plugin ID* used by embedding-based tool lookup (e.g., 'st_embedder', 'openai_embedder')."
    )

    # Command Processor Feature
    command_processor: Literal["llm_assisted", "simple_keyword", "rewoo", "none"] = Field(
        default="none", description="Processor for interpreting user commands into tool calls."
    )
    command_processor_formatter_id_alias: Optional[str] = Field(
        default="compact_text_formatter", description="Alias for formatter used by LLM-assisted command processor for presenting tools to LLM."
    )

    # Logging Adapter Feature
    logging_adapter: Literal["default_log_adapter", "pyvider_log_adapter", "none"] = Field(
        default="default_log_adapter", description="Primary logging adapter for structured events."
    )
    logging_pyvider_service_name: Optional[str] = Field(
        default=None, description="Service name for Pyvider telemetry if 'pyvider_log_adapter' is chosen."
    )

    # Observability Feature
    observability_tracer: Literal["console_tracer", "otel_tracer", "none"] = Field(
        default="none", description="Primary interaction tracer choice."
    )
    observability_otel_endpoint: Optional[str] = Field(
        default=None, description="Endpoint for OpenTelemetry OTLP exporter if 'otel_tracer' is chosen (e.g., http://localhost:4318/v1/traces for HTTP)."
    )

    # HITL Feature
    hitl_approver: Literal["cli_hitl_approver", "none"] = Field(
        default="none", description="Human-in-the-loop approval mechanism."
    )

    # Token Usage Feature
    token_usage_recorder: Literal["in_memory_token_recorder", "otel_metrics_recorder", "none"] = Field(
        default="none", description="Token usage recording mechanism."
    )

    # Guardrails Feature
    input_guardrails: List[str] = Field(
        default_factory=list, description="List of input guardrail plugin IDs or aliases to enable."
    )
    output_guardrails: List[str] = Field(
        default_factory=list, description="List of output guardrail plugin IDs or aliases to enable."
    )
    tool_usage_guardrails: List[str] = Field(
        default_factory=list, description="List of tool usage guardrail plugin IDs or aliases to enable."
    )

    # Prompt System Features
    prompt_registry: Literal["file_system_prompt_registry", "none"] = Field(
        default="none", description="Prompt registry choice."
    )
    prompt_template_engine: Literal["basic_string_formatter", "jinja2_chat_formatter", "none"] = Field(
        default="none", description="Prompt template engine choice."
    )

    # Conversation State Features
    conversation_state_provider: Literal["in_memory_convo_provider", "redis_convo_provider", "none"] = Field(
        default="none", description="Conversation state provider choice."
    )

    # LLM Output Parser Features
    default_llm_output_parser: Literal["json_output_parser", "pydantic_output_parser", "none"] = Field(
        default="none", description="Default LLM output parser choice."
    )


    # Distributed Task Queue Feature
    task_queue: Literal["celery", "rq", "none"] = Field(
        default="none", description="Distributed task queue system to use for offloading tasks."
    )
    task_queue_celery_broker_url: Optional[str] = Field(
        default="redis://localhost:6379/1", description="Broker URL for Celery if 'celery' is chosen."
    )
    task_queue_celery_backend_url: Optional[str] = Field(
        default="redis://localhost:6379/2", description="Result backend URL for Celery if 'celery' is chosen."
    )