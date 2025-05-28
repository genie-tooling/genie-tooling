# src/genie_tooling/command_processors/abc.py
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin
from genie_tooling.llm_providers.types import ChatMessage  # For conversation_history

from .types import CommandProcessorResponse

if TYPE_CHECKING:
    from genie_tooling.genie import Genie  # Import for type hinting only

logger = logging.getLogger(__name__)

@runtime_checkable
class CommandProcessorPlugin(Plugin, Protocol):
    """
    Protocol for a plugin that processes a natural language command to determine
    which tool to use and what parameters to pass to it.
    """
    plugin_id: str      # Inherited from Plugin, unique identifier for this processor plugin
    description: str    # Human-readable description of this command processor's capabilities

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        """
        Initializes the command processor.
        Args:
            config: Processor-specific configuration dictionary. Expected to contain
                    'genie_facade: Genie' for accessing other middleware components.
        """
        # Example:
        # self._genie_facade = config.get("genie_facade")
        # if not isinstance(self._genie_facade, Genie): # Assuming Genie type is available for check
        #    logger.error(f"{self.plugin_id}: Genie facade not found in config or invalid.")
        await super().setup(config) # Call Plugin's default setup
        logger.debug(f"CommandProcessorPlugin '{self.plugin_id}': Base setup logic (if any) completed.")


    async def process_command(
        self,
        command: str,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> CommandProcessorResponse:
        """
        Processes the given command, potentially using conversation history and
        other Genie components (via the facade provided in setup), to decide
        on a tool and its parameters.

        Args:
            command: The natural language command string from the user.
            conversation_history: Optional list of previous ChatMessages in the conversation.

        Returns:
            A CommandProcessorResponse dictionary.
        """
        logger.error(f"CommandProcessorPlugin '{self.plugin_id}' process_command method not implemented.")
        raise NotImplementedError(f"CommandProcessorPlugin '{self.plugin_id}' does not implement 'process_command'.")

    # teardown is inherited from Plugin protocol.
    # async def teardown(self) -> None:
    #     await super().teardown()
    #     logger.debug(f"CommandProcessorPlugin '{self.plugin_id}': Default teardown called.")
