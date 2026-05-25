# src/genie_tooling/tools/abc.py
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
        """A unique string identifier for this tool."""
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
            "cache_ttl_seconds": Optional[int], (optional, default TTL if cacheable)

            # --- Side-effect classification (Phase 6A.1) ---
            "side_effects": Literal["unknown","none","read","write","destructive"],
                (optional, default "unknown". Drives the Claude-Code-style permission model.
                 "none"=pure computation; "read"=external read only; "write"=external write that
                 is reversible/idempotent; "destructive"=irreversible or high-blast-radius
                 (drops data, removes resources, sends external comms, etc.).)
            "requires_approval": Optional[bool], (default None → defer to permission model defaults
                 based on side_effects. True forces HITL; False forces auto-allow.)
            "idempotent": bool, (default False. Hints that re-executing with the same params is safe.)
        }

        Tools authored via the `@tool` decorator can supply the side-effect fields via
        decorator kwargs, e.g. ``@tool(side_effects="read", idempotent=True)``.
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
            context: A dictionary carrying session or request-specific data.
                This can include framework-level information (e.g., 'correlation_id',
                'otel_context' for distributed tracing, 'genie_framework_instance')
                as well as application-specific data passed into methods like
                `genie.run_command(..., context_for_tools=...)`.

        Returns:
            The result of the tool execution. The structure should align with output_schema.
            If an error occurs during execution that the tool handles, it should still
            return a structured response, possibly including an 'error' field, conforming
            to its output_schema. Unhandled exceptions will be caught by the InvocationStrategy.
        """
        ...
