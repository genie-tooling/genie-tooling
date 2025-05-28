# src/genie_tooling/command_processors/manager.py
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.plugin_manager import PluginManager
# KeyProvider might be useful for some processors, though not directly used by manager logic itself
from genie_tooling.security.key_provider import KeyProvider 

from .abc import CommandProcessorPlugin

if TYPE_CHECKING:
    from genie_tooling.genie import Genie # Import for type hinting only

logger = logging.getLogger(__name__)

class CommandProcessorManager:
    """
    Manages the lifecycle and access to CommandProcessorPlugin instances.
    """

    def __init__(self, plugin_manager: PluginManager, key_provider: KeyProvider, config: MiddlewareConfig):
        self._plugin_manager = plugin_manager
        self._key_provider = key_provider # Stored for consistency, plugins might request it via their config
        self._global_config = config # To access command_processor_configurations
        self._instantiated_processors: Dict[str, CommandProcessorPlugin] = {}
        logger.info("CommandProcessorManager initialized.")

    async def get_command_processor(
        self,
        processor_id: str,
        genie_facade: 'Genie', # Pass the Genie facade instance for plugin setup
        config_override: Optional[Dict[str, Any]] = None
    ) -> Optional[CommandProcessorPlugin]:
        """
        Retrieves an initialized CommandProcessorPlugin instance.
        - Checks cache for an existing instance.
        - If not found, uses PluginManager to get a new instance.
        - Merges global configuration for the processor with any runtime overrides.
        - Ensures the Genie facade is passed to the plugin's setup method.
        - Caches and returns the instance.
        """
        if processor_id in self._instantiated_processors:
            logger.debug(f"Returning cached CommandProcessorPlugin instance for '{processor_id}'.")
            if config_override:
                 logger.warning(f"Config override provided for already instantiated command processor '{processor_id}'. "
                                 "Current implementation does not re-setup. Override ignored.")
            return self._instantiated_processors[processor_id]

        processor_configs_map = getattr(self._global_config, "command_processor_configurations", {})
        global_processor_config = processor_configs_map.get(processor_id, {})

        final_setup_config = global_processor_config.copy()
        if config_override:
            final_setup_config.update(config_override)
        
        # Crucially, add the genie_facade to the config dict that PluginManager
        # will pass to the plugin's setup method.
        final_setup_config["genie_facade"] = genie_facade
        # Also pass key_provider if it's convention for plugin setups
        final_setup_config["key_provider"] = self._key_provider


        logger.debug(f"Attempting to load CommandProcessorPlugin '{processor_id}'.")

        # PluginManager's get_plugin_instance calls setup on the plugin with `final_setup_config`.
        processor_instance_any = await self._plugin_manager.get_plugin_instance(
            plugin_id=processor_id,
            config=final_setup_config
        )

        if processor_instance_any and isinstance(processor_instance_any, CommandProcessorPlugin):
            processor_instance = cast(CommandProcessorPlugin, processor_instance_any)
            self._instantiated_processors[processor_id] = processor_instance
            logger.info(f"CommandProcessorPlugin '{processor_id}' loaded and initialized successfully.")
            return processor_instance
        else:
            logger.error(f"Failed to load or invalid CommandProcessorPlugin for ID '{processor_id}'.")
            return None

    async def teardown(self) -> None:
        """
        Placeholder for any specific teardown CommandProcessorManager might need.
        Individual plugin teardowns are handled by PluginManager.
        """
        logger.info("CommandProcessorManager tearing down. Instantiated processors will be torn down by PluginManager.")
        self._instantiated_processors.clear()