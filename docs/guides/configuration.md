# Configuration

Genie Tooling is configured at runtime using a `MiddlewareConfig` object. For ease of use, especially for common setups, `MiddlewareConfig` integrates a `FeatureSettings` model.

## Simplified Configuration with `FeatureSettings`

The recommended way to start configuring Genie is by using the `features` attribute of `MiddlewareConfig`. `FeatureSettings` provides high-level toggles and default choices for major components like LLM providers, RAG components, caching, and tool lookup.

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    features=FeatureSettings(
        llm="ollama",  # Use Ollama as the default LLM
        llm_ollama_model_name="mistral:latest", # Specify the Ollama model
        
        command_processor="llm_assisted", # Use LLM-assisted tool selection
        tool_lookup="embedding",          # Use embedding-based tool lookup for the LLM processor
        tool_lookup_embedder_id_alias="st_embedder", # Embedder for tool lookup
        tool_lookup_formatter_id_alias="compact_text_formatter", # Formatter for tool indexing

        rag_embedder="sentence_transformer", # Embedder for RAG
        rag_vector_store="faiss",            # Vector store for RAG
        
        cache="in-memory" # Use in-memory cache
    )
)
```

### How `FeatureSettings` Works

When you initialize `Genie` with a `MiddlewareConfig` containing `FeatureSettings`, an internal `ConfigResolver` processes these settings. It translates your high-level choices into specific plugin IDs and default configurations for those plugins.

For example, setting `features.llm = "ollama"` will make the resolver:
1.  Set `default_llm_provider_id` to the canonical ID of the Ollama LLM provider (e.g., `"ollama_llm_provider_v1"`).
2.  Populate a basic configuration for this provider in `llm_provider_configurations`, including the specified `llm_ollama_model_name`.

### Aliases

The `ConfigResolver` uses a system of aliases to map short, user-friendly names (like "ollama", "st_embedder", "compact_text_formatter") to their full canonical plugin IDs. This makes configuration more concise.

For a full list of available aliases and more details on simplified configuration, please see the [Simplified Configuration Guide](simplified_configuration.md).

## Explicit Overrides and Detailed Configuration

While `FeatureSettings` provides a convenient starting point, you can always provide more detailed, explicit configurations that will override or augment the settings derived from features.

You can directly set default plugin IDs:

```python
app_config = MiddlewareConfig(
    features=FeatureSettings(llm="openai"), # Base feature
    default_llm_provider_id="my_custom_openai_provider_v2" # Override default ID
)
```

You can also provide specific configurations for individual plugins using the various `*_configurations` dictionaries in `MiddlewareConfig`. These dictionaries are keyed by the **canonical plugin ID** (or a recognized alias, which the resolver will convert).

```python
app_config = MiddlewareConfig(
    features=FeatureSettings(
        llm="openai",
        llm_openai_model_name="gpt-3.5-turbo", # Default model from features
        tool_lookup="embedding",
        tool_lookup_embedder_id_alias="st_embedder"
    ),
    # Override configuration for the OpenAI LLM provider
    llm_provider_configurations={
        "openai_llm_provider_v1": { # Canonical ID
            "model_name": "gpt-4-turbo-preview", # Override model
            "request_timeout_seconds": 120
        }
    },
    # Configure a specific tool
    tool_configurations={
        "sandboxed_fs_tool_v1": {"sandbox_base_path": "./agent_workspace"}
    },
    # Configure the embedding-based tool lookup provider
    tool_lookup_provider_configurations={
        "embedding_similarity_lookup_v1": {
            "embedder_id": "openai_embedding_generator_v1", # Use OpenAI for tool embeddings instead of ST
            "embedder_config": {"model_name": "text-embedding-3-small"},
            # Vector store config for tool lookup (e.g., Chroma)
            # "vector_store_id": "chromadb_vector_store_v1",
            # "vector_store_config": {
            #     "collection_name": "my_tool_embeddings_persistent",
            #     "path": "./tool_lookup_db"
            # }
        }
    }
)
```

**Key Points for Explicit Configuration:**

*   **Precedence**: Explicit configurations in `default_*_id` fields or within the `*_configurations` dictionaries take precedence over settings derived from `FeatureSettings`.
*   **Canonical IDs vs. Aliases**: When providing explicit configurations in dictionaries like `llm_provider_configurations`, you can use either the canonical plugin ID (e.g., `"ollama_llm_provider_v1"`) or a recognized alias (e.g., `"ollama"`). The `ConfigResolver` will map aliases to their canonical IDs.
*   **KeyProvider**: API keys are managed by a `KeyProvider` implementation. Genie defaults to `EnvironmentKeyProvider` (alias `"env_keys"`) if no `key_provider_instance` is passed to `Genie.create()` and `key_provider_id` is not set. Plugins requiring keys (like OpenAI or Gemini providers) will receive the configured `KeyProvider` instance.

## Plugin Development Directories

If you have custom plugins located outside your main Python path or not installed as entry points, you can specify their location:

```python
app_config = MiddlewareConfig(
    plugin_dev_dirs=["/path/to/my/custom_plugins", "./project_plugins"]
)
```
The `PluginManager` will scan these directories for valid plugin classes.
