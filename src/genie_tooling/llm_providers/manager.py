### src/genie_tooling/llm_providers/manager.py
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
        # The PluginManager's get_plugin_instance is responsible for passing config to plugin.setup.
        # The LLMProviderPlugin's setup signature is (self, config, key_provider).
        # So, the config dict passed to get_plugin_instance should contain 'key_provider'
        # if the plugin's setup method expects it directly.
        # However, the standard way is for setup to receive a 'config' dict and extract what it needs.
        # For LLMProviderPlugin, the second argument is 'key_provider', not part of 'config'.
        # So, PluginManager needs to be aware of this, or we adjust.
        # The current PluginManager passes the 'config' arg directly to plugin.setup's 'config' param.
        # Let's assume the plugin's setup method will expect 'key_provider' within its 'config' dict.
        # Or, if LLMProviderPlugin setup is `async def setup(self, config: Optional[Dict[str, Any]], key_provider: KeyProvider)`
        # then `get_plugin_instance` must be adapted or this manager handles it differently.
        # Given PluginManager.get_plugin_instance(plugin_id, config=config_for_plugin, **kwargs_for_init),
        # it doesn't directly support passing extra args to setup beyond the 'config' dict.
        #
        # Simpler: LLMProviderPlugin.setup takes `config` and `key_provider`.
        # PluginManager.get_plugin_instance calls `instance.setup(config=...)`.
        # This means the `key_provider` needs to be passed when the plugin is *instantiated* if it's an init arg,
        # or the setup signature for LLMProviderPlugin should be `setup(self, config_with_kp_and_other_stuff)`.
        #
        # Let's follow the definition from TODO.md:
        # `async def setup(self, config: Optional[Dict[str, Any]], key_provider: KeyProvider) -> None`.
        # This means PluginManager's call `instance.setup(config=plugin_config)` is not directly compatible.
        #
        # RESOLUTION: PluginManager will instantiate, then this manager will call setup manually.
        # OR, we make PluginManager more flexible (harder).
        # OR, for simplicity, we can expect `key_provider` to be part of the `final_setup_config` and the plugin extracts it.
        #
        # Let's assume for now `PluginManager.get_plugin_instance` will pass `final_setup_config` as the `config` argument
        # to the plugin's `setup` method. The plugin's `setup` method signature is
        # `async def setup(self, config: Optional[Dict[str, Any]], key_provider: KeyProvider)`.
        # This is a mismatch.
        #
        # Correct approach:
        # 1. `PluginManager.get_plugin_instance(plugin_id, **init_kwargs)` for instantiation.
        # 2. Then, `await instance.setup(config=plugin_specific_config, key_provider=self._key_provider)` called by *this* manager.
        # This requires `get_plugin_instance` to *not* call `setup` automatically if a flag is passed, or
        # we just call setup again (idempotency needed).
        #
        # Given the existing PluginManager, it *does* call `setup` with the `config` kwarg.
        # So, the `LLMProviderPlugin.setup` must be changed to:
        # `async def setup(self, config: Optional[Dict[str, Any]]) -> None:`
        # And inside that setup, it must extract `key_provider` and other specific settings from the `config` dict.
        # So, this manager must *put* `key_provider` into `final_setup_config`.

        final_setup_config["key_provider"] = self._key_provider # Add key_provider to the config dict
        logger.debug(f"Attempting to load LLMProviderPlugin '{provider_id}' using PluginManager.")

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
