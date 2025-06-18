# src/genie_tooling/command_processors/impl/deep_research_processor.py
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from genie_tooling.agents.deep_research_agent import DeepResearchAgent
from genie_tooling.command_processors.abc import CommandProcessorPlugin
from genie_tooling.command_processors.types import CommandProcessorResponse
from genie_tooling.llm_providers.types import ChatMessage

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)

class DeepResearchProcessorPlugin(CommandProcessorPlugin):
    """
    A specialized command processor that invokes the DeepResearchAgent.
    This acts as a bridge, allowing a complex agentic workflow to be triggered
    as a single "command".
    """
    plugin_id: str = "deep_research_agent_v1"
    description: str = "Processes a command by delegating it to the DeepResearchAgent."

    _genie: "Genie"
    _agent_config: Dict[str, Any]

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        """
        Initializes the processor and stores the agent's configuration.

        Args:
            config: A dictionary containing configuration settings:
                - `genie_facade` ("Genie"): The main Genie facade instance, which is
                  required for the agent to function.
                - `agent_config` (Dict[str, Any], optional): A dictionary of
                  configuration options to be passed directly to the
                  DeepResearchAgent instance. This can include settings like
                  'web_search_tool_id', 'min_high_quality_sources', etc.
        """
        await super().setup(config)
        cfg = config or {}
        self._genie = cfg.get("genie_facade")
        if not self._genie:
            raise ValueError(f"{self.plugin_id} requires a 'genie_facade' instance in its config.")

        self._agent_config = cfg.get("agent_config", {})
        logger.info(f"{self.plugin_id}: Initialized. Will delegate commands to DeepResearchAgent.")

    async def process_command(
        self,
        command: str,
        conversation_history: Optional[List[ChatMessage]] = None,
        correlation_id: Optional[str] = None
    ) -> CommandProcessorResponse:

        await self._genie.observability.trace_event(
            "deep_research_processor.start",
            {"goal": command},
            self.plugin_id,
            correlation_id
        )

        try:
            research_agent = DeepResearchAgent(genie=self._genie, agent_config=self._agent_config)
            agent_result = await research_agent.run(goal=command)
            final_answer = agent_result.get("output", "Research did not produce a final answer.")
            llm_thought_process = json.dumps(agent_result, default=str, indent=2)

            await self._genie.observability.trace_event(
                "deep_research_processor.end",
                {"status": agent_result.get("status"), "final_answer_length": len(final_answer)},
                self.plugin_id,
                correlation_id
            )

            return {
                "final_answer": final_answer,
                "llm_thought_process": llm_thought_process,
                "raw_response": agent_result,
            }

        except Exception as e:
            logger.error(f"Error running DeepResearchAgent via processor: {e}", exc_info=True)
            await self._genie.observability.trace_event(
                "deep_research_processor.error",
                {"error": str(e), "exc_info": True},
                self.plugin_id,
                correlation_id
            )
            return {"error": f"Failed to execute deep research: {e!s}"}