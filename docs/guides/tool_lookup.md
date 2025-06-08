# Tool Lookup

Genie Tooling refers to the process of finding relevant tools based on a natural language query. This is primarily used by the `LLMAssistedToolSelectionProcessorPlugin` to narrow down the list of tools presented to the LLM, making the LLM's selection task more efficient and accurate.

## How Tool Lookup is Used

When you configure the `Genie` facade to use the `llm_assisted` command processor, you can also enable and configure tool lookup:

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    features=FeatureSettings(
        llm="ollama",
        command_processor="llm_assisted",
        
        # Tool Lookup Configuration
        tool_lookup="hybrid", # Enable hybrid (embedding + keyword) search
        tool_lookup_top_k=5,  # How many tools to show the LLM
    ),
    tool_configurations={
        "calculator_tool": {},
        "open_weather_map_tool": {} 
    }
)
```

The `LLMAssistedToolSelectionProcessorPlugin` internally uses the `ToolLookupService` to:
1.  **Index Tools**: When first needed, it takes all enabled tools, formats their definitions, and indexes them using the configured `ToolLookupProvider`. This now happens incrementally when new tools are registered.
2.  **Find Tools**: When processing a user command, it queries the `ToolLookupService` with the command. The service returns a ranked list of potentially relevant tools.
3.  **Filter Tools**: The processor then takes the top N tools (defined by `tool_lookup_top_k`) from this list and presents only their definitions to the LLM for final selection.

## Available Tool Lookup Providers

Genie Tooling includes built-in providers:

*   **`EmbeddingSimilarityLookupProvider` (alias: `"embedding"`)**:
    *   Embeds tool definitions and queries for semantic search.
    *   Can use an in-memory NumPy index or a persistent `VectorStorePlugin`.
*   **`KeywordMatchLookupProvider` (alias: `"keyword"`)**:
    *   Performs simple keyword matching. Stateless and fast.
*   **`HybridSearchLookupProvider` (alias: `"hybrid"`)**:
    *   **Recommended**: Combines the results of both embedding and keyword search using Reciprocal Rank Fusion (RRF).
    *   This provides a powerful balance of semantic understanding (from embeddings) and lexical precision (from keywords), often yielding the most relevant results.

## Incremental Indexing

The `ToolLookupService` now supports incremental indexing. When you register new tools at runtime with `await genie.register_tool_functions(...)`, the service will automatically add or update just those tools in the index, rather than rebuilding the entire index from scratch. This is crucial for performance in dynamic environments.

## Debugging Tool Discoverability

It can sometimes be challenging to understand why a specific tool was or was not selected for a query. To aid in this, Genie Tooling provides a developer utility.

**`scripts/debug_tool_lookup.py`**

This command-line script allows you to test queries directly against the `ToolLookupService` and see the ranked results.

**Usage:**

1.  (Optional) Create a simple config file, e.g., `debug_config.yaml`, especially if you use `plugin_dev_dirs`:
    ```yaml
    plugin_dev_dirs:
      - ./src/my_project/custom_plugins
    features:
      tool_lookup: hybrid # Or 'embedding', 'keyword'
    ```

2.  Run the script from your project's root directory:
    ```bash
    poetry run python scripts/debug_tool_lookup.py --query "what is the weather in Paris" --config-path debug_config.yaml
    ```

**Output:**

The script will print a ranked list of tools, including their scores and any matched keywords or similarity details, helping you refine your tool descriptions for better discoverability.

```
--- Tool Lookup Debugger ---
Query: 'what is the weather in Paris'
Top K: 5
...
--- Lookup Results ---

1. Tool: open_weather_map_tool (Score: 1.6541)
   Matched Keywords: weather
   Similarity Details: {'cosine_similarity': 0.85}
   Snippet/Reason: WeatherInfo: Get current weather for a city...

2. Tool: some_other_tool (Score: 0.0164)
   ...
```
This allows you to iterate quickly on your tool's `description_llm` and `tags` to improve its ranking for relevant queries.