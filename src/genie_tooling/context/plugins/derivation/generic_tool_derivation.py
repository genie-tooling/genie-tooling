import logging
from typing import TYPE_CHECKING, Any, Dict

from genie_tooling.context.protocols import DerivationStrategyPlugin

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)


class GenericToolDerivationPlugin(DerivationStrategyPlugin):
    """
    A generic derivation strategy that executes a single tool call as
    specified in the derivation constraints.
    """

    plugin_id: str = "generic_tool_derivation_v1"
    description: str = "Derives answers by executing a specific tool."

    async def setup(self, config=None):
        pass

    async def derive(
        self, query: str, constraints: Dict[str, Any], genie: "Genie"
    ) -> Dict[str, Any]:

        tool_id = constraints.get("tool_id")
        params = constraints.get("params", {})

        if not tool_id:
            return {
                "status": "error",
                "error": "Derivation failed: No 'tool_id' specified in derivation constraints (C_D).",
            }

        if not isinstance(params, dict):
            return {
                "status": "error",
                "error": f"Derivation failed: 'params' for tool '{tool_id}' must be a dictionary.",
            }

        logger.info(
            f"[{self.plugin_id}] Deriving result by executing tool '{tool_id}' with params: {params}"
        )

        try:
            tool_result = await genie.execute_tool(tool_id, **params)
            return {"status": "success", "result": tool_result}
        except Exception as e:
            logger.error(
                f"[{self.plugin_id}] An unexpected error occurred while executing tool '{tool_id}': {e}",
                exc_info=True
            )
            return {
                "status": "error",
                "error": f"Failed to derive result via tool execution: {e!s}"
            }
