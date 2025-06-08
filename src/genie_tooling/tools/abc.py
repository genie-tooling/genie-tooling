"""Abstract Base Class/Protocol for Tool Plugins."""
from typing import Any, Dict, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin
from genie_tooling.security.key_provider import KeyProvider  # Defined later


@runtime_checkable
class Tool(Plugin, Protocol):
    """
    Protocol for a tool that can be executed by the middleware.
    All tools must be designed to be async.
    """
    @property
    def identifier(self) -> str:
        ...

    async def get_metadata(self) -> Dict[str, Any]:
        """
        Returns comprehensive metadata about the tool.
        This metadata is crucial for tool discovery, LLM function calling, and UI display.

        Expected structure:
        {
            "identifier": str, (matches self.identifier)
            "name": str, (human-friendly name)
            "description_human": str, (detailed for developers/UI)
            "description_llm": str, (concise, token-efficient for LLM prompts/function descriptions)
            "input_schema": Dict[str, Any], (JSON Schema for tool parameters)
            "output_schema": Dict[str, Any], (JSON Schema for tool's expected output structure)
            "key_requirements": List[Dict[str, str]], (e.g., [{"name": "API_KEY_NAME", "description": "Purpose of key"}])
            "tags": List[str], (for categorization, e.g., ["weather", "api", "location"])
            "version": str, (e.g., "1.0.0")
            "cacheable": bool, (optional, hints if tool output can be cached, default False)
            "cache_ttl_seconds": Optional[int] (optional, default TTL if cacheable)
        }
        """
        ...

    async def execute(
        self,
        params: Dict[str, Any],
        key_provider: KeyProvider,
        context: Dict[str, Any]
    ) -> Any:
        """
        Executes the tool with the given parameters.
        Must be an async method.

        Args:
            params: Validated parameters for the tool, conforming to its input_schema.
            key_provider: An async key provider instance for fetching necessary API keys.
            context: Context dictionary carrying session/request-specific data, including
                     observability trace context.

        Returns:
            The result of the tool execution. The structure should align with output_schema.
            If an error occurs during execution that the tool handles, it should still
            return a structured response, possibly including an 'error' field, conforming
            to its output_schema. Unhandled exceptions will be caught by the InvocationStrategy.
        """
        ...
