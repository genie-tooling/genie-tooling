# Simplified Configuration with FeatureSettings and Aliases

Genie Tooling aims to make common configurations straightforward through `FeatureSettings` and a system of plugin ID aliases. This guide explains these concepts in detail.

## `FeatureSettings`

The `FeatureSettings` model, part of `MiddlewareConfig`, provides high-level toggles for major functionalities. When you use `Genie.create(config=MiddlewareConfig(features=...))`, the internal `ConfigResolver` uses these settings to:

1.  Set default plugin IDs for various components (e.g., `default_llm_provider_id`, `default_rag_embedder_id`, `default_log_adapter_id`).
2.  Populate basic configurations for these default plugins in the respective `*_configurations` dictionaries (e.g., setting the model name for the chosen LLM provider, or service name for Pyvider log adapter).

**Example of `FeatureSettings`:**

```python
from genie_tooling.config.features import FeatureSettings

features = FeatureSettings(
    # LLM Provider
    llm="ollama",                             # Chooses OllamaLLMProviderPlugin
    llm_ollama_model_name="mistral:7b-instruct-q4_K_M", # Sets model for Ollama
    # Or for Llama.cpp Internal:
    # llm="llama_cpp_internal",
    # llm_llama_cpp_internal_model_path="/path/to/your/model.gguf",
    # llm_llama_cpp_internal_n_gpu_layers=-1,
    # llm_llama_cpp_internal_chat_format="mistral",

    # Command Processing
    command_processor="llm_assisted",         # Chooses LLMAssistedToolSelectionProcessorPlugin
    command_processor_formatter_id_alias="compact_text_formatter", # Formatter for LLM prompt

    # Tool Lookup
    tool_lookup="hybrid",                  # Uses HybridSearchLookupProvider (Recommended)
    tool_lookup_embedder_id_alias="st_embedder", # Embedder for tool lookup
    tool_lookup_formatter_id_alias="compact_text_formatter", # Formatter for tool indexing
    # tool_lookup_chroma_path="./my_tool_lookup_db", # Optional: Use ChromaDB for tool lookup embeddings

    # RAG
    rag_embedder="openai",                    # Uses OpenAIEmbeddingGenerator for RAG
    rag_vector_store="chroma",                # Uses ChromaDBVectorStore for RAG
    rag_vector_store_chroma_path="./my_rag_vector_db",
    rag_vector_store_chroma_collection_name="main_rag_docs",

    # Caching
    cache="redis",                            # Uses RedisCacheProvider
    cache_redis_url="redis://localhost:6379/1",

    # Logging Adapter (NEW)
    logging_adapter="pyvider_log_adapter",      # Use Pyvider Telemetry
    logging_pyvider_service_name="my-genie-service", # Service name for Pyvider

    # Observability & Monitoring
    observability_tracer="otel_tracer",       # Use OpenTelemetry for traces
    observability_otel_endpoint="http://localhost:4318/v1/traces", # OTLP/HTTP endpoint
    token_usage_recorder="otel_metrics_recorder", # Use OpenTelemetry for token metrics

    # Human-in-the-Loop
    hitl_approver="cli_hitl_approver",        # Use CLI for approvals

    # Guardrails
    input_guardrails=["keyword_blocklist_guardrail"], # List of guardrail aliases/IDs

    # Prompt System & Conversation
    prompt_registry="file_system_prompt_registry",
    prompt_template_engine="jinja2_chat_formatter",
    conversation_state_provider="in_memory_convo_provider",
    default_llm_output_parser="pydantic_output_parser",

    # Distributed Tasks
    task_queue="celery",                      # Use Celery
    task_queue_celery_broker_url="redis://localhost:6379/3",
    task_queue_celery_backend_url="redis://localhost:6379/4",
    # Or for RQ:
    # task_queue="rq",
    # (RQ typically uses the same Redis for broker/backend, configured via redis_url in its plugin config)
)

# This 'features' object would be passed to MiddlewareConfig:
# from genie_tooling.config.models import MiddlewareConfig
# app_config = MiddlewareConfig(features=features)
```

## Plugin ID Aliases

To make `FeatureSettings` and explicit configurations more readable, Genie uses a system of aliases. The `ConfigResolver` maps these short aliases to their full, canonical plugin IDs.

**Commonly Used Aliases (Examples):**

*   **LLM Providers:**
    *   `"ollama"`: `"ollama_llm_provider_v1"`
    *   `"openai"`: `"openai_llm_provider_v1"`
    *   `"gemini"`: `"gemini_llm_provider_v1"`
    *   `"llama_cpp"`: `"llama_cpp_llm_provider_v1"` (for server)
    *   `"llama_cpp_internal"`: `"llama_cpp_internal_llm_provider_v1"`
*   **Key Provider:**
    *   `"env_keys"`: `"environment_key_provider_v1"` (Default if no KeyProvider instance/ID is given)
*   **Caching Providers:**
    *   `"in_memory_cache"`: `"in_memory_cache_provider_v1"`
    *   `"redis_cache"`: `"redis_cache_provider_v1"`
*   **Embedding Generators (Embedders):**
    *   `"st_embedder"`: `"sentence_transformer_embedder_v1"`
    *   `"openai_embedder"`: `"openai_embedding_generator_v1"`
*   **Vector Stores:**
    *   `"faiss_vs"`: `"faiss_vector_store_v1"`
    *   `"chroma_vs"`: `"chromadb_vector_store_v1"`
    *   `"qdrant_vs"`: `"qdrant_vector_store_v1"`
*   **Tool Lookup Providers:**
    *   `"embedding_lookup"`: `"embedding_similarity_lookup_v1"`
    *   `"keyword_lookup"`: `"keyword_match_lookup_v1"`
    *   `"hybrid_lookup"`: `"hybrid_search_lookup_v1"`
*   **Definition Formatters:**
    *   `"compact_text_formatter"`: `"compact_text_formatter_plugin_v1"`
    *   `"openai_func_formatter"`: `"openai_function_formatter_plugin_v1"`
    *   `"hr_json_formatter"`: `"human_readable_json_formatter_plugin_v1"`
*   **Command Processors:**
    *   `"llm_assisted_cmd_proc"`: `"llm_assisted_tool_selection_processor_v1"`
    *   `"simple_keyword_cmd_proc"`: `"simple_keyword_processor_v1"`
*   **Log Adapters (NEW):**
    *   `"default_log_adapter"`: `"default_log_adapter_v1"`
    *   `"pyvider_log_adapter"`: `"pyvider_telemetry_log_adapter_v1"`
*   **Observability & Monitoring:**
    *   `"console_tracer"`: `"console_tracer_plugin_v1"`
    *   `"otel_tracer"`: `"otel_tracer_plugin_v1"`
    *   `"in_memory_token_recorder"`: `"in_memory_token_usage_recorder_v1"`
    *   `"otel_metrics_recorder"`: `"otel_metrics_token_recorder_v1"`
*   **HITL & Guardrails:**
    *   `"cli_hitl_approver"`: `"cli_approval_plugin_v1"`
    *   `"keyword_blocklist_guardrail"`: `"keyword_blocklist_guardrail_v1"`
*   **Prompt System & Conversation:**
    *   `"file_system_prompt_registry"`: `"file_system_prompt_registry_v1"`
    *   `"basic_string_formatter"`: `"basic_string_format_template_v1"`
    *   `"jinja2_chat_formatter"`: `"jinja2_chat_template_v1"`
    *   `"in_memory_convo_provider"`: `"in_memory_conversation_state_v1"`
    *   `"redis_convo_provider"`: `"redis_conversation_state_v1"`
*   **LLM Output Parsers:**
    *   `"json_output_parser"`: `"json_output_parser_v1"`
    *   `"pydantic_output_parser"`: `"pydantic_output_parser_v1"`
*   **Task Queues:**
    *   `"celery_task_queue"`: `"celery_task_queue_v1"`
    *   `"rq_task_queue"`: `"redis_queue_task_plugin_v1"`

*(This list is not exhaustive. Refer to `src/genie_tooling/config/resolver.py` for the complete `PLUGIN_ID_ALIASES` dictionary.)*

## Overriding Feature-Derived Settings

You can always provide more specific configurations that will take precedence over what `FeatureSettings` implies.

**1. Overriding Default Plugin IDs:**

If a feature sets a default (e.g., `features.llm = "openai"` sets `default_llm_provider_id` to `"openai_llm_provider_v1"`), you can override this directly:

```python
app_config = MiddlewareConfig(
    features=FeatureSettings(llm="openai", logging_adapter="default_log_adapter"),
    default_llm_provider_id="my_special_openai_provider_v3", # Explicit override
    default_log_adapter_id="pyvider_telemetry_log_adapter_v1" # Explicit override
)
```

**2. Overriding Plugin Configurations:**

Use the `*_configurations` dictionaries in `MiddlewareConfig` to provide settings for specific plugins. You can use either the canonical ID or an alias as the key.

```python
app_config = MiddlewareConfig(
    features=FeatureSettings(
        llm="ollama",
        llm_ollama_model_name="mistral:latest", # Feature sets this model
        logging_adapter="pyvider_log_adapter",
        logging_pyvider_service_name="FeatureServiceName"
    ),
    llm_provider_configurations={
        "ollama": { # Using alias 'ollama'
            "model_name": "llama3:8b-instruct-fp16", # Override the model
            "request_timeout_seconds": 300.0,
            # Other Ollama-specific settings
        }
    },
    log_adapter_configurations={
        "pyvider_telemetry_log_adapter_v1": { # Using canonical ID
            "service_name": "ExplicitPyviderServiceName", # Overrides feature setting
            "default_level": "DEBUG"
        }
    }
)
```

In this example:
*   The default LLM is Ollama.
*   The `mistral:latest` model set by `FeatureSettings` for Ollama is overridden by `llama3:8b-instruct-fp16`.
*   The default Log Adapter is Pyvider.
*   The `service_name` for Pyvider set by `FeatureSettings` is overridden by the explicit configuration.

This layered approach (Features -> Explicit Defaults -> Explicit Plugin Configs) provides both ease of use for common cases and fine-grained control when needed.