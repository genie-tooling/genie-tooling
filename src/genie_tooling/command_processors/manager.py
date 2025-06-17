# src/genie_tooling/command_processors/manager.py
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, cast

from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.security.key_provider import KeyProvider

from .abc import CommandProcessorPlugin

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)

class CommandProcessorManager:
    def __init__(self, plugin_manager: PluginManager, key_provider: KeyProvider, config: MiddlewareConfig):
        self._plugin_manager = plugin_manager
        self._key_provider = key_provider
        self._global_config = config
        self._instantiated_processors: Dict[str, CommandProcessorPlugin] = {}
        logger.info("CommandProcessorManager initialized.")
        logger.debug(f"CommandProcessorManager __init__: Received MiddlewareConfig.command_processor_configurations = {getattr(config, 'command_processor_configurations', 'ATTRIBUTE_NOT_FOUND')}")


    async def get_command_processor(
        self,
        processor_id: str,
        # --- CHANGE: Added genie_facade parameter ---
        genie_facade: "Genie",
        config_override: Optional[Dict[str, Any]] = None
    ) -> Optional[CommandProcessorPlugin]:
        if processor_id in self._instantiated_processors:
            logger.debug(f"Returning cached CommandProcessorPlugin instance for '{processor_id}'.")
            if config_override:
                 logger.warning(f"Config override provided for already instantiated command processor '{processor_id}'. "
                                 "Current implementation does not re-setup. Override ignored.")
            return self._instantiated_processors[processor_id]

        if not isinstance(self._global_config, MiddlewareConfig):
            logger.error(f"CommandProcessorManager: self._global_config is not a MiddlewareConfig instance. Type: {type(self._global_config)}")
            return None

        plugin_class: Optional[Type[CommandProcessorPlugin]] = self._plugin_manager.list_discovered_plugin_classes().get(processor_id) # type: ignore
        if not plugin_class:
            logger.error(f"CommandProcessorPlugin class for ID '{processor_id}' not found in PluginManager.")
            return None

        processor_configs_map = self._global_config.command_processor_configurations
        global_processor_config = processor_configs_map.get(processor_id, {})
        logger.debug(f"CommandProcessorManager.get_command_processor: global_processor_config for '{processor_id}': {global_processor_config}")

        final_setup_config = global_processor_config.copy()
        if config_override:
            final_setup_config.update(config_override)

        # --- FIX: Inject the genie_facade instance here ---
        final_setup_config["genie_facade"] = genie_facade
        # --- END FIX ---

        final_setup_config["key_provider"] = self._key_provider
        logger.debug(f"CommandProcessorManager.get_command_processor: final_setup_config for plugin '{processor_id}': {final_setup_config}")

        try:
            # --- CHANGE: Pass agent_config from the processor's own config to its constructor ---
            # This allows agent-as-plugin to receive its specific settings.
            constructor_kwargs = {}
            if "agent_config" in final_setup_config:
                constructor_kwargs["agent_config"] = final_setup_config["agent_config"]

            instance = plugin_class(**constructor_kwargs) # type: ignore
            # --- END CHANGE ---
            
            await instance.setup(config=final_setup_config)

            if not isinstance(instance, CommandProcessorPlugin):
                logger.error(f"Instantiated plugin '{processor_id}' is not a valid CommandProcessorPlugin. Type: {type(instance)}")
                return None

            processor_instance = cast(CommandProcessorPlugin, instance)
            self._instantiated_processors[processor_id] = processor_instance
            logger.info(f"CommandProcessorPlugin '{processor_id}' instantiated, set up, and cached by CommandProcessorManager.")
            return processor_instance
        except Exception as e:
            logger.error(f"Error instantiating or setting up CommandProcessorPlugin '{processor_id}': {e}", exc_info=True)
            return None

    async def teardown(self) -> None:
        logger.info("CommandProcessorManager tearing down...")
        for processor_id, instance in list(self._instantiated_processors.items()):
            try:
                await instance.teardown()
                logger.debug(f"CommandProcessorPlugin '{processor_id}' torn down by CommandProcessorManager.")
            except Exception as e:
                logger.error(f"Error tearing down CommandProcessorPlugin '{processor_id}': {e}", exc_info=True)
        self._instantiated_processors.clear()
        logger.info("CommandProcessorManager teardown complete.")