# src/genie_tooling/llm_providers/manager.py
import logging
from typing import Any, Dict, Optional, cast

from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.security.key_provider import KeyProvider

from .abc import LLMProviderPlugin

logger = logging.getLogger(__name__)

class LLMProviderManager:
    """
    Manages the lifecycle and access to LLMProviderPlugin instances.
    It uses the PluginManager to discover and instantiate LLM provider plugins
    and handles passing necessary configurations and KeyProvider to them.
    """

    def __init__(self, plugin_manager: PluginManager, key_provider: KeyProvider, config: MiddlewareConfig):
        self._plugin_manager = plugin_manager
        self._key_provider = key_provider
        self._global_config = config # To access llm_provider_configurations
        self._instantiated_providers: Dict[str, LLMProviderPlugin] = {}
        logger.info("LLMProviderManager initialized.")

    async def get_llm_provider(
        self,
        provider_id: str,
        config_override: Optional[Dict[str, Any]] = None
    ) -> Optional[LLMProviderPlugin]:
        """
        Retrieves an initialized LLMProviderPlugin instance.
        - Checks cache for an existing instance.
        - If not found, uses PluginManager to get a new instance.
        - Merges global configuration for the provider ID with any runtime overrides.
        - Calls the plugin's setup method with the KeyProvider and merged config.
        - Caches and returns the instance.
        """
        if provider_id in self._instantiated_providers:
            logger.debug(f"Returning cached LLMProviderPlugin instance for '{provider_id}'.")
            # TODO: Consider if config_override should re-setup or if instances are immutable post-setup.
            # For now, assume setup is once. If re-configurable instances are needed, logic here changes.
            if config_override:
                 logger.warning(f"Config override provided for already instantiated LLM provider '{provider_id}'. "
                                 "Current implementation does not re-setup. Override ignored.")
            return self._instantiated_providers[provider_id]

        # Get global configuration for this specific provider_id from MiddlewareConfig
        provider_configs_map = getattr(self._global_config, "llm_provider_configurations", {})
        global_provider_config = provider_configs_map.get(provider_id, {})

        # Merge global config with runtime override
        final_setup_config = global_provider_config.copy()
        if config_override:
            final_setup_config.update(config_override)
        
        # Ensure KeyProvider is passed to the plugin's setup
        final_setup_config["key_provider"] = self._key_provider

        logger.debug(f"Attempting to load LLMProviderPlugin '{provider_id}' with config: {final_setup_config.get('model_name', 'N/A')}")

        # PluginManager's get_plugin_instance already calls setup on the plugin.
        # We pass the final_setup_config to it.
        provider_instance_any = await self._plugin_manager.get_plugin_instance(
            plugin_id=provider_id,
            config=final_setup_config # This config is passed to the plugin's setup method
        )

        if provider_instance_any and isinstance(provider_instance_any, LLMProviderPlugin):
            provider_instance = cast(LLMProviderPlugin, provider_instance_any)
            self._instantiated_providers[provider_id] = provider_instance
            logger.info(f"LLMProviderPlugin '{provider_id}' loaded and initialized successfully.")
            return provider_instance
        else:
            logger.error(f"Failed to load or invalid LLMProviderPlugin for ID '{provider_id}'.")
            return None

    async def teardown(self) -> None:
        """
        Placeholder for any specific teardown LLMProviderManager might need.
        Individual plugin teardowns are handled by PluginManager.
        """
        logger.info("LLMProviderManager tearing down. Instantiated providers will be torn down by PluginManager.")
        self._instantiated_providers.clear()