### src/genie_tooling/llm_providers/manager.py
# src/genie_tooling/llm_providers/manager.py
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, cast

from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.security.key_provider import KeyProvider
# P1.5: Import TokenUsageManager if it's to be used here
# from genie_tooling.token_usage.manager import TokenUsageManager 
from genie_tooling.core.types import Plugin
from .abc import LLMProviderPlugin

if TYPE_CHECKING:
    from genie_tooling.token_usage.manager import TokenUsageManager # For type hinting

logger = logging.getLogger(__name__)

class LLMProviderManager:
    def __init__(
        self, 
        plugin_manager: PluginManager, 
        key_provider: KeyProvider, 
        config: MiddlewareConfig,
        token_usage_manager: Optional["TokenUsageManager"] = None # Added for P1.5
    ):
        self._plugin_manager = plugin_manager
        self._key_provider = key_provider
        self._global_config = config
        self._token_usage_manager = token_usage_manager # Store it
        self._instantiated_providers: Dict[str, LLMProviderPlugin] = {}
        logger.info("LLMProviderManager initialized.")
        if self._token_usage_manager:
            logger.info("LLMProviderManager: TokenUsageManager instance received.")
        logger.debug(f"LLMProviderManager __init__: Received MiddlewareConfig.llm_provider_configurations = {getattr(config, 'llm_provider_configurations', 'ATTRIBUTE_NOT_FOUND')}")

    async def get_llm_provider(
        self,
        provider_id: str,
        config_override: Optional[Dict[str, Any]] = None
    ) -> Optional[LLMProviderPlugin]:
        if provider_id in self._instantiated_providers:
            logger.debug(f"Returning cached LLMProviderPlugin instance for '{provider_id}'.")
            if config_override:
                 logger.warning(f"Config override provided for already instantiated LLM provider '{provider_id}'. "
                                 "Current implementation does not re-setup. Override ignored.")
            return self._instantiated_providers[provider_id]

        if not isinstance(self._global_config, MiddlewareConfig):
            logger.error(f"LLMProviderManager: self._global_config is not a MiddlewareConfig instance. Type: {type(self._global_config)}")
            return None

        plugin_class_any: Optional[Type[Plugin]] = self._plugin_manager.list_discovered_plugin_classes().get(provider_id) # type: ignore
        if not plugin_class_any:
            logger.error(f"LLMProviderPlugin class for ID '{provider_id}' not found in PluginManager.")
            return None

        plugin_class = cast(Type[LLMProviderPlugin], plugin_class_any)

        provider_configs_map = self._global_config.llm_provider_configurations
        global_provider_config = provider_configs_map.get(provider_id, {})
        final_setup_config = global_provider_config.copy()
        if config_override:
            final_setup_config.update(config_override)

        final_setup_config["key_provider"] = self._key_provider
        # P1.5: Pass TokenUsageManager to the LLM provider's setup config if it's designed to use it
        if self._token_usage_manager:
            final_setup_config["token_usage_manager"] = self._token_usage_manager
            
        logger.debug(f"LLMProviderManager.get_llm_provider: final_setup_config for plugin '{provider_id}': {final_setup_config}")

        try:
            instance = plugin_class() # type: ignore
            await instance.setup(config=final_setup_config)

            if not isinstance(instance, LLMProviderPlugin):
                logger.error(f"Instantiated plugin '{provider_id}' is not a valid LLMProviderPlugin. Type: {type(instance)}")
                return None

            provider_instance = cast(LLMProviderPlugin, instance)
            self._instantiated_providers[provider_id] = provider_instance
            logger.info(f"LLMProviderPlugin '{provider_id}' (class actual ID: {instance.plugin_id}) loaded and initialized successfully.")
            return provider_instance
        except Exception as e:
            logger.error(f"Error instantiating or setting up LLMProviderPlugin '{provider_id}': {e}", exc_info=True)
            return None

    async def teardown(self) -> None:
        logger.info("LLMProviderManager tearing down...")
        for provider_id, instance in list(self._instantiated_providers.items()):
            try:
                await instance.teardown()
                logger.debug(f"LLMProviderPlugin '{provider_id}' (instance ID: {instance.plugin_id}) torn down by LLMProviderManager.")
            except Exception as e:
                logger.error(f"Error tearing down LLMProviderPlugin '{provider_id}' (instance ID: {instance.plugin_id}): {e}", exc_info=True)
        self._instantiated_providers.clear()
        logger.info("LLMProviderManager teardown complete.")