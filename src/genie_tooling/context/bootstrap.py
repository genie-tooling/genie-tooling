import logging
from typing import TYPE_CHECKING

from genie_tooling.bootstrap import BootstrapPlugin

from .interface import ContextInterface
from .manager import ContextManager

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)

class CqsEngineBootstrapPlugin(BootstrapPlugin):
    """
    Initializes the CQS-Engine subsystem and attaches its public interface
    to the `genie` facade as `genie.context`.
    """
    plugin_id: str = "cqs_engine_bootstrap_v1"
    description: str = "Initializes and attaches the Context Scoping Engine (CQS-Engine)."

    async def bootstrap(self, genie: "Genie") -> None:
        """
        Executes the bootstrap logic for the CQS-Engine.
        """
        logger.info(f"[{self.plugin_id}] Bootstrapping CQS-Engine...")

        # Retrieve the extension-specific configuration from the main config
        context_engine_config = genie._config.extension_configurations.get("context_engine", {}) # type: ignore

        if not context_engine_config:
            logger.warning(
                f"[{self.plugin_id}] No 'context_engine' configuration found in "
                "'extension_configurations'. The engine may not function correctly."
            )

        try:
            # The manager is the internal orchestrator. It needs access to the full
            # genie instance to use its sub-managers (like plugin_manager, llm, etc.).
            context_manager = ContextManager(genie=genie, config=context_engine_config)
            await context_manager.setup()

            # The interface is the clean, public-facing API.
            context_interface = ContextInterface(manager=context_manager)

            # Attach the public interface to the genie instance.
            genie.context = context_interface
            logger.info(
                f"[{self.plugin_id}] Bootstrap complete. `genie.context` interface is now available."
            )

        except Exception as e:
            logger.error(f"[{self.plugin_id}] Failed to bootstrap CQS-Engine: {e}", exc_info=True)
            # We re-raise to halt application startup if the engine fails to initialize.
            raise
