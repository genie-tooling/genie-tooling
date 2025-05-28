"""Abstract Base Class/Protocol for InvocationStrategy Plugins."""
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool


@runtime_checkable
class InvocationStrategy(Plugin, Protocol):
    """
    Protocol for a strategy that defines the complete lifecycle of invoking a tool.
    This includes validation, execution, transformation, caching, and error handling.
    All strategies must be designed to be async.
    """
    # plugin_id: str (from Plugin protocol)

    async def invoke(
        self,
        tool: Tool,
        params: Dict[str, Any],
        key_provider: KeyProvider,
        context: Optional[Dict[str, Any]],
        invoker_config: Dict[str, Any] # Contains plugin_manager and override IDs for components
    ) -> Any:
        """
        Executes the full tool invocation lifecycle according to this strategy.

        Args:
            tool: The Tool instance to invoke.
            params: The parameters for the tool.
            key_provider: The async key provider.
            context: Optional context dictionary.
            invoker_config: Configuration from the ToolInvoker, including:
                            - "plugin_manager": PluginManager instance
                            - "validator_id": Optional[str]
                            - "transformer_id": Optional[str]
                            - "error_handler_id": Optional[str]
                            - "error_formatter_id": Optional[str]
                            - "cache_provider_id": Optional[str]
                            - "cache_config": Optional[Dict[str, Any]]

        Returns:
            The result of the tool execution (possibly transformed by an OutputTransformer)
            or a structured error (formatted by an ErrorFormatter). The exact type depends
            on the formatter's output.
        """
        ...
