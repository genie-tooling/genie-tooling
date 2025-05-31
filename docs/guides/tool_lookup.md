# Tool Lookup

Tool Lookup in Genie Tooling refers to the process of finding relevant tools based on a natural language query. This is primarily used by the `LLMAssistedToolSelectionProcessorPlugin` to narrow down the list of tools presented to the LLM, making the LLM's selection task more efficient and accurate.

## How Tool Lookup is Used

When you configure the `Genie` facade to use the `llm_assisted` command processor, you can also enable and configure tool lookup:

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    features=FeatureSettings(
        llm="ollama", # LLM for the command processor
        command_processor="llm_assisted",
        
        # Tool Lookup Configuration
        tool_lookup="embedding", # Enable embedding-based tool lookup
        # tool_lookup="keyword", # Or enable keyword-based tool lookup
        # tool_lookup="none",    # Disable tool lookup (LLM sees all tools)

        # If tool_lookup="embedding":
        tool_lookup_embedder_id_alias="st_embedder", # e.g., "sentence_transformer_embedder_v1"
        # tool_lookup_embedder_id_alias="openai_embedder", # Or "openai_embedding_generator_v1"
        
        tool_lookup_formatter_id_alias="compact_text_formatter", # Formatter for tool text before embedding
        
        # Optional: Configure ChromaDB for persistent tool lookup embeddings
        # tool_lookup_chroma_path="./my_tool_lookup_db",
        # tool_lookup_chroma_collection_name="agent_tool_embeddings"
    ),
    # Further configure the LLM-assisted processor
    command_processor_configurations={
        "llm_assisted_tool_selection_processor_v1": {
            "tool_lookup_top_k": 3 # How many top tools from lookup to show the LLM
        }
    }
)
```

The `LLMAssistedToolSelectionProcessorPlugin` internally uses the `ToolLookupService` to:
1.  **Index Tools**: When first needed (or if the index is invalidated), it takes all available tools, formats their definitions (using the `tool_lookup_formatter_id_alias`), and indexes them using the configured `ToolLookupProvider` (e.g., `EmbeddingSimilarityLookupProvider` or `KeywordMatchLookupProvider`).
2.  **Find Tools**: When processing a user command, it queries the `ToolLookupService` with the command. The service returns a ranked list of potentially relevant tools.
3.  **Filter Tools**: The processor then takes the top N tools (defined by `tool_lookup_top_k`) from this list and presents only their definitions to the LLM for final selection.

## Available Tool Lookup Providers

Genie Tooling includes built-in providers:

*   **`EmbeddingSimilarityLookupProvider` (alias: `"embedding_lookup"`)**:
    *   Embeds tool definitions (formatted text) and user queries using a configured `EmbeddingGeneratorPlugin` (e.g., Sentence Transformers, OpenAI Embeddings).
    *   Finds tools based on cosine similarity in the embedding space.
    *   Can use an in-memory NumPy index or a persistent `VectorStorePlugin` (like ChromaDB) to store tool embeddings.
*   **`KeywordMatchLookupProvider` (alias: `"keyword_lookup"`)**:
    *   Performs simple keyword matching between the query and tool definitions (name, description, tags, formatted text).
    *   Stateless and does not require an embedder or vector store.

## Configuring Tool Lookup Components

*   **`features.tool_lookup`**: Sets the default `ToolLookupProviderPlugin` (`"embedding"`, `"keyword"`, or `"none"`).
*   **`features.tool_lookup_formatter_id_alias`**: Specifies the alias of the `DefinitionFormatterPlugin` used to create the textual representation of tools for indexing (e.g., `"compact_text_formatter"`).
*   **`features.tool_lookup_embedder_id_alias`**: If `tool_lookup="embedding"`, this sets the alias of the `EmbeddingGeneratorPlugin` used for embedding tool definitions and queries (e.g., `"st_embedder"`, `"openai_embedder"`).
*   **`features.tool_lookup_chroma_path` / `tool_lookup_chroma_collection_name`**: If using embedding lookup and you want ChromaDB for persistence, these configure the `EmbeddingSimilarityLookupProvider` to use a `ChromaDBVectorStorePlugin` internally.

You can provide more detailed configurations for the chosen `ToolLookupProviderPlugin` or its sub-components (like its embedder or vector store) via the `tool_lookup_provider_configurations`, `embedding_generator_configurations`, or `vector_store_configurations` dictionaries in `MiddlewareConfig`.

Example: Overriding the embedder model for tool lookup:
```python
app_config = MiddlewareConfig(
    features=FeatureSettings(
        tool_lookup="embedding",
        tool_lookup_embedder_id_alias="st_embedder" # Default ST embedder
    ),
    embedding_generator_configurations={
        # This config applies if 'st_embedder' (sentence_transformer_embedder_v1)
        # is used by ANY component, including tool lookup.
        "sentence_transformer_embedder_v1": {
            "model_name": "paraphrase-MiniLM-L3-v2" # Use a different ST model
        }
    }
)
```

Or, to configure the `EmbeddingSimilarityLookupProvider` directly to use a specific embedder and its config:
```python
app_config = MiddlewareConfig(
    features=FeatureSettings(tool_lookup="embedding"), # Selects embedding_similarity_lookup_v1
    tool_lookup_provider_configurations={
        "embedding_similarity_lookup_v1": {
            "embedder_id": "openai_embedding_generator_v1", # Override to use OpenAI
            "embedder_config": {"model_name": "text-embedding-3-small"},
            # vector_store_id and vector_store_config can also be set here
        }
    }
)
```

The `ToolLookupService` handles the re-indexing of tools automatically if new tools are registered or if its index is explicitly invalidated.
