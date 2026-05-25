import logging
from typing import TYPE_CHECKING, Any, Dict

from genie_tooling.context.protocols import DerivationStrategyPlugin

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)


class GenericAgentDerivationPlugin(DerivationStrategyPlugin):
    """
    A generic derivation strategy that invokes a Command Processor or Agent
    specified in the derivation constraints.
    """

    plugin_id: str = "generic_agent_derivation_v1"
    description: str = "Derives answers by running a specified agent or command processor."

    async def setup(self, config=None):
        pass

    async def derive(
        self, query: str, constraints: Dict[str, Any], genie: "Genie"
    ) -> Dict[str, Any]:

        processor_id = constraints.get("command_processor_id")
        if not processor_id:
            return {
                "status": "error",
                "error": "Derivation failed: No 'command_processor_id' specified in derivation constraints (C_D).",
            }

        logger.info(
            f"[{self.plugin_id}] Deriving result by running command processor '{processor_id}' for query: '{query}'"
        )

        try:
            agent_output = await genie.run_command(
                command=query,
                processor_id=processor_id,
                context_for_tools={"cqs_constraints": constraints}
            )
            return {"status": "success", "result": agent_output}
        except Exception as e:
            logger.error(
                f"[{self.plugin_id}] An unexpected error occurred while running agent/processor '{processor_id}': {e}",
                exc_info=True
            )
            return {
                "status": "error",
                "error": f"Failed to derive result via agentic execution: {e!s}"
            }
