# src/genie_tooling/agents/base_agent.py
"""Abstract Base Class for Agentic Loop Implementations."""
import abc
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from genie_tooling.genie import Genie  # Import Genie for type hinting

logger = logging.getLogger(__name__)

class BaseAgent(abc.ABC):
    """
    Abstract base class for agents that implement specific execution patterns
    (e.g., ReAct, Plan-and-Execute) using the Genie facade.
    """
    def __init__(self, genie: "Genie", agent_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the BaseAgent.

        Args:
            genie: An initialized instance of the Genie facade.
            agent_config: Optional dictionary for agent-specific configurations.
        """
        if not genie:
            raise ValueError("A Genie instance is required to initialize an agent.")
        self.genie = genie
        self.agent_config = agent_config or {}
        logger.info(f"Initialized {self.__class__.__name__} with Genie instance and config: {self.agent_config}")

    @abc.abstractmethod
    async def run(self, goal: str, **kwargs: Any) -> Any:
        """
        The main entry point to execute the agent's logic for a given goal.

        Args:
            goal: The high-level goal or task for the agent to accomplish.
            **kwargs: Additional runtime parameters specific to the agent's execution.

        Returns:
            The final result or outcome of the agent's execution.
            The structure of the result is specific to the agent implementation.
        """
        pass

    async def teardown(self) -> None:
        """
        Optional method for any cleanup specific to the agent.
        The Genie instance itself should be torn down separately by the application.
        """
        logger.info(f"{self.__class__.__name__} teardown initiated.")
        # Add any agent-specific cleanup here if needed in the future.
        pass
