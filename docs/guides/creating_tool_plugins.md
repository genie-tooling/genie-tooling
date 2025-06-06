# Creating Tool Plugins

Tools are fundamental to Genie Tooling, representing actions an agent can perform. You can create custom tools by implementing the `Tool` protocol.

## The `Tool` Protocol

Located in `genie_tooling.tools.abc.Tool`, the protocol requires:

```python
from typing import Protocol, Any, Dict, Optional
from genie_tooling.core.types import Plugin # For base Plugin behavior
from genie_tooling.security.key_provider import KeyProvider

class Tool(Plugin, Protocol):
    @property
    def identifier(self) -> str:
        """A unique string identifier for this tool."""
        ...

    async def get_metadata(self) -> Dict[str, Any]:
        """
        Returns comprehensive metadata about the tool.
        Expected structure:
        {
            "identifier": str,
            "name": str, (Human-friendly)
            "description_human": str, (Detailed for UI/developers)
            "description_llm": str, (Concise for LLM prompts)
            "input_schema": Dict[str, Any], (JSON Schema for parameters)
            "output_schema": Dict[str, Any], (JSON Schema for result)
            "key_requirements": List[Dict[str, str]], (e.g., [{"name": "API_KEY_NAME", ...}])
            "tags": List[str],
            "version": str,
            "cacheable": bool, (Optional, default False)
            "cache_ttl_seconds": Optional[int] (Optional)
        }
        """
        ...

    async def execute(
        self,
        params: Dict[str, Any],
        key_provider: KeyProvider,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Executes the tool with validated parameters."""
        ...
```
Your tool class must also have a `plugin_id` attribute (usually the same as `identifier`).

## Steps to Create a Tool Plugin

1.  **Define Your Class**:
    ```python
    from genie_tooling.tools.abc import Tool
    from genie_tooling.security.key_provider import KeyProvider
    from typing import Dict, Any, Optional

    class MyCustomSearchTool(Tool):
        plugin_id: str = "my_custom_search_tool_v1" # Unique plugin ID

        @property
        def identifier(self) -> str:
            return self.plugin_id # Often same as plugin_id

        async def setup(self, config: Optional[Dict[str, Any]] = None):
            self.api_base_url = (config or {}).get("api_base_url", "https://api.customsearch.com")
            # Initialize HTTP client, etc.

        async def get_metadata(self) -> Dict[str, Any]:
            return {
                "identifier": self.identifier,
                "name": "My Custom Search",
                "description_human": "Searches my custom data source.",
                "description_llm": "CustomSearch: Finds items in my data. Args: query (str, req), limit (int, opt, default 10).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "limit": {"type": "integer", "default": 10, "description": "Max results."}
                    },
                    "required": ["query"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "results": {"type": "array", "items": {"type": "object"}},
                        "error": {"type": ["string", "null"]}
                    },
                    "required": ["results"]
                },
                "key_requirements": [{"name": "MY_CUSTOM_API_KEY", "description": "API key for custom search."}],
                "tags": ["search", "custom"],
                "version": "1.0.0"
            }

        async def execute(
            self, 
            params: Dict[str, Any], 
            key_provider: KeyProvider, 
            context: Optional[Dict[str, Any]] = None
        ) -> Any:
            query = params["query"]
            limit = params.get("limit", 10)
            api_key = await key_provider.get_key("MY_CUSTOM_API_KEY")
            if not api_key:
                return {"results": [], "error": "API key not found."}
            
            # ... actual search logic using self.api_base_url, query, limit, api_key ...
            # For example:
            # response = await self._http_client.get(f"{self.api_base_url}/search?q={query}&limit={limit}&key={api_key}")
            # search_results = response.json() 
            search_results = [{"title": f"Mock result for {query}"}] # Placeholder
            return {"results": search_results, "error": None}

        async def teardown(self):
            # Close HTTP client, etc.
            pass
    ```

2.  **Register Your Plugin**:
    *   Add an entry point in `pyproject.toml`:
        ```toml
        [tool.poetry.plugins."genie_tooling.plugins"]
        "my_custom_search_tool_v1" = "my_package.tools:MyCustomSearchTool"
        ```
    *   Or, place the Python file in a directory listed in `MiddlewareConfig.plugin_dev_dirs`.

3.  **Enable and Configure in `MiddlewareConfig`**:
    ```python
    app_config = MiddlewareConfig(
        # ...
        tool_configurations={
            "my_custom_search_tool_v1": { # This is the plugin_id
                "api_base_url": "https://prod.customsearch.com/v2" # Configuration for its setup()
            }
        }
        # ...
    )
    ```
    **Important**: Your tool will only be loaded and available if its `plugin_id` is a key in the `tool_configurations` dictionary. An empty dictionary `{}` as the value is sufficient if no specific configuration is needed for `setup()`.

Now, your `MyCustomSearchTool` can be used by `genie.execute_tool("my_custom_search_tool_v1", ...)` or selected by a command processor.
