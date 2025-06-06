# Configuration

Genie Tooling is configured at runtime using a `MiddlewareConfig` object. For ease of use, especially for common setups, `MiddlewareConfig` integrates a `FeatureSettings` model.

## Simplified Configuration with `FeatureSettings`

The recommended way to start configuring Genie is by using the `features` attribute of `MiddlewareConfig`. `FeatureSettings` provides high-level toggles and default choices for major components like LLM providers, RAG components, caching, tool lookup, observability, HITL, token usage, guardrails, prompt system, conversation state, and distributed task queues.

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    features=FeatureSettings(
        llm="ollama",  # Use Ollama as the default LLM
        llm_ollama_model_name="mistral:latest", # Specify the Ollama model
        
        command_processor="llm_assisted", # Use LLM-assisted tool selection
        tool_lookup="embedding",          # Use embedding-based tool lookup for the LLM processor
        
        rag_embedder="sentence_transformer", # Embedder for RAG
        rag_vector_store="faiss",            # Vector store for RAG
        
        cache="in-memory", # Use in-memory cache

        observability_tracer="console_tracer", # Log traces to console
        hitl_approver="cli_hitl_approver",       # Use CLI for human approvals
        token_usage_recorder="in_memory_token_recorder", # Track token usage in memory
        input_guardrails=["keyword_blocklist_guardrail"], # Enable a keyword blocklist for inputs
        
        prompt_registry="file_system_prompt_registry",
        prompt_template_engine="jinja2_chat_formatter",
        conversation_state_provider="in_memory_convo_provider",
        default_llm_output_parser="json_output_parser",

        task_queue="celery", # Example: Use Celery for distributed tasks
        task_queue_celery_broker_url="redis://localhost:6379/1",
        task_queue_celery_backend_url="redis://localhost:6379/2"
    ),
    # Enable and configure tools explicitly
    tool_configurations={
        "calculator_tool": {}, # Enable calculator tool (no specific config needed)
        "sandboxed_fs_tool_v1": {"sandbox_base_path": "./my_agent_workspace_feature_example"}
    },
    # Configure the keyword blocklist guardrail
    guardrail_configurations={
        "keyword_blocklist_guardrail_v1": { # Canonical ID
            "blocklist": ["sensitive_data_pattern", "forbidden_command"],
            "action_on_match": "block"
        }
    },
    # Configure prompt registry if file_system_prompt_registry is used
    prompt_registry_configurations={
        "file_system_prompt_registry_v1": {"base_path": "./my_prompts"}
    }
)
```

### How `FeatureSettings` Works

When you initialize `Genie` with a `MiddlewareConfig` containing `FeatureSettings`, an internal `ConfigResolver` processes these settings. It translates your high-level choices into specific plugin IDs and default configurations for those plugins.

For example, setting `features.llm = "ollama"` will make the resolver:
1.  Set `default_llm_provider_id` to the canonical ID of the Ollama LLM provider (e.g., `"ollama_llm_provider_v1"`).
2.  Populate a basic configuration for this provider in `llm_provider_configurations`, including the specified `llm_ollama_model_name`.

Similarly, `features.input_guardrails=["keyword_blocklist_guardrail"]` will add the canonical ID of the `KeywordBlocklistGuardrailPlugin` to `default_input_guardrail_ids`.

Setting `features.rag_vector_store = "qdrant"` along with `rag_vector_store_qdrant_url` and `rag_vector_store_qdrant_embedding_dim` would set `default_rag_vector_store_id` to `"qdrant_vector_store_v1"` and populate its configuration in `vector_store_configurations` with the URL, collection name, and embedding dimension.

This layered approach (Features -> Explicit Defaults -> Explicit Plugin Configs) provides both ease of use for common cases and fine-grained control when needed.

### Aliases

The `ConfigResolver` uses a system of aliases to map short, user-friendly names (like "ollama", "st_embedder", "console_tracer") to their full canonical plugin IDs. This makes configuration more concise.

For a full list of available aliases and more details on simplified configuration, please see the [Simplified Configuration Guide](simplified_configuration.md).

## Explicit Overrides and Detailed Configuration

While `FeatureSettings` provides a convenient starting point, you can always provide more detailed, explicit configurations that will override or augment the settings derived from features.

You can directly set default plugin IDs for any component:

```python
app_config = MiddlewareConfig(
    features=FeatureSettings(llm="openai"), # Base feature
    default_llm_provider_id="my_custom_openai_provider_v2", # Override default ID
    default_observability_tracer_id="my_custom_tracer_v1" 
)
```

You can also provide specific configurations for individual plugins using the various `*_configurations` dictionaries in `MiddlewareConfig`. These dictionaries are keyed by the **canonical plugin ID** (or a recognized alias, which the resolver will convert).

**Crucially, for tools to be active and usable by `genie.execute_tool` or `genie.run_command`, their plugin ID (canonical or alias) must be a key in the `tool_configurations` dictionary.** If a tool requires no specific settings, an empty dictionary `{}` as its value is sufficient to enable it.

```python
app_config = MiddlewareConfig(
    features=FeatureSettings(
        llm="openai",
        llm_openai_model_name="gpt-3.5-turbo", 
        observability_tracer="console_tracer"
    ),
    # Override configuration for the OpenAI LLM provider
    llm_provider_configurations={
        "openai_llm_provider_v1": { # Canonical ID
            "model_name": "gpt-4-turbo-preview", 
            "request_timeout_seconds": 120
        },
        # Example for Llama.cpp Internal Provider
        "llama_cpp_internal_llm_provider_v1": {
            "model_path": "/path/to/your/model.gguf",
            "n_gpu_layers": -1,
            "chat_format": "mistral"
        }
    },
    # Enable and configure specific tools
    tool_configurations={
        "calculator_tool": {}, # Enable calculator, no specific config needed
        "sandboxed_fs_tool_v1": {"sandbox_base_path": "./agent_workspace"}
    },
    # Configure a specific observability tracer
    observability_tracer_configurations={
        "console_tracer_plugin_v1": {"log_level": "DEBUG"}
    },
    # Configure a specific guardrail
    guardrail_configurations={
        "keyword_blocklist_guardrail_v1": {
            "blocklist": ["confidential_project_alpha"],
            "case_sensitive": True,
            "action_on_match": "warn"
        }
    },
    # Configure a specific prompt registry
    prompt_registry_configurations={
        "file_system_prompt_registry_v1": {"base_path": "path/to/prompts", "template_suffix": ".txt"}
    },
    # Configure a specific conversation state provider
    conversation_state_provider_configurations={
        "redis_conversation_state_v1": {"redis_url": "redis://localhost:6379/1"}
    },
    # Configure a specific LLM output parser
    llm_output_parser_configurations={
        "json_output_parser_v1": {"strict_parsing": False}
    },
    # Configure a specific task queue
    distributed_task_queue_configurations={
        "celery_task_queue_v1": {
            "celery_app_name": "genie_worker_app_explicit",
            "celery_broker_url": "amqp://guest:guest@localhost:5672//", # Example AMQP
            "celery_backend_url": "redis://localhost:6379/3"
        }
    }
)
```

**Key Points for Explicit Configuration:**

*   **Precedence**: Explicit configurations in `default_*_id` fields or within the `*_configurations` dictionaries take precedence over settings derived from `FeatureSettings`.
*   **Tool Enablement**: A tool plugin is only loaded and made active if its ID (or alias) is present as a key in the `tool_configurations` dictionary.
*   **Canonical IDs vs. Aliases**: When providing explicit configurations in dictionaries like `llm_provider_configurations` or `tool_configurations`, you can use either the canonical plugin ID (e.g., `"ollama_llm_provider_v1"`) or a recognized alias (e.g., `"ollama"`). The `ConfigResolver` will map aliases to their canonical IDs.
*   **KeyProvider**: API keys are managed by a `KeyProvider` implementation. Genie defaults to `EnvironmentKeyProvider` (alias `"env_keys"`) if no `key_provider_instance` is passed to `Genie.create()` and `key_provider_id` is not set. Plugins requiring keys (like OpenAI or Gemini providers) will receive the configured `KeyProvider` instance.

## Plugin Development Directories

If you have custom plugins located outside your main Python path or not installed as entry points, you can specify their location:

```python
app_config = MiddlewareConfig(
    plugin_dev_dirs=["/path/to/my/custom_plugins", "./project_plugins"]
)
```
The `PluginManager` will scan these directories for valid plugin classes. Discovered plugins still need to be explicitly enabled via their respective configuration sections (e.g., `tool_configurations` for tools) to be loaded by `Genie`.
