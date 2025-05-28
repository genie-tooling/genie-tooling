### tests/unit/llm_providers/test_llm_provider_manager.py
"""Unit tests for LLMProviderManager."""
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.llm_providers.manager import LLMProviderManager
from genie_tooling.security.key_provider import KeyProvider


class MockLLMProvider(LLMProviderPlugin):
    plugin_id: str
    description: str = "Mock LLM Provider for Manager Tests"
    setup_config_received: Optional[Dict[str, Any]] = None
    key_provider_received: Optional[KeyProvider] = None

    def __init__(self, plugin_id: str = "mock_llm_provider_v1"):
        self.plugin_id = plugin_id
        self.setup_config_received = None
        self.key_provider_received = None

    async def setup(
        self, config: Optional[Dict[str, Any]], key_provider: KeyProvider
    ) -> None:
        self.setup_config_received = config
        self.key_provider_received = key_provider
        if config and config.get("fail_setup"):
            raise RuntimeError("Simulated setup failure")

    async def generate(self, prompt: str, **kwargs: Any) -> Any:
        return {"text": f"Generated: {prompt}", "finish_reason": "stop", "usage": None, "raw_response": {}}

    async def chat(self, messages: List[Any], **kwargs: Any) -> Any:
        return {"message": {"role": "assistant", "content": "Chat response"}, "finish_reason": "stop", "usage": None, "raw_response": {}}

    async def teardown(self) -> None:
        pass


@pytest.fixture
def mock_plugin_manager_for_llm_mgr(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm

@pytest.fixture
def mock_key_provider_for_llm_mgr(mock_key_provider) -> KeyProvider: # Reuse conftest fixture
    return mock_key_provider

@pytest.fixture
def mock_middleware_config() -> MiddlewareConfig:
    return MiddlewareConfig(
        llm_provider_configurations={
            "provider_alpha": {"model_name": "alpha-model", "global_param": "global_alpha"},
            "provider_beta": {"model_name": "beta-model"},
        }
    )

@pytest.fixture
def llm_provider_manager(
    mock_plugin_manager_for_llm_mgr: PluginManager,
    mock_key_provider_for_llm_mgr: KeyProvider,
    mock_middleware_config: MiddlewareConfig,
) -> LLMProviderManager:
    return LLMProviderManager(
        plugin_manager=mock_plugin_manager_for_llm_mgr,
        key_provider=mock_key_provider_for_llm_mgr,
        config=mock_middleware_config,
    )


@pytest.mark.asyncio
async def test_get_llm_provider_success_new_instance(
    llm_provider_manager: LLMProviderManager,
    mock_plugin_manager_for_llm_mgr: PluginManager,
    mock_key_provider_for_llm_mgr: KeyProvider,
):
    """Test successfully loading a new LLM provider instance."""
    provider_id = "provider_alpha"
    mock_provider_instance = MockLLMProvider(plugin_id=provider_id)
    mock_plugin_manager_for_llm_mgr.get_plugin_instance.return_value = mock_provider_instance

    instance = await llm_provider_manager.get_llm_provider(provider_id)

    assert instance is mock_provider_instance
    mock_plugin_manager_for_llm_mgr.get_plugin_instance.assert_awaited_once()
    # Check that the config passed to get_plugin_instance (and thus to plugin's setup) is correct
    call_args = mock_plugin_manager_for_llm_mgr.get_plugin_instance.call_args
    assert call_args.kwargs["plugin_id"] == provider_id
    passed_config = call_args.kwargs["config"]
    assert passed_config["model_name"] == "alpha-model"
    assert passed_config["global_param"] == "global_alpha"
    assert passed_config["key_provider"] is mock_key_provider_for_llm_mgr


@pytest.mark.asyncio
async def test_get_llm_provider_cached_instance(
    llm_provider_manager: LLMProviderManager,
    mock_plugin_manager_for_llm_mgr: PluginManager,
):
    """Test returning a cached LLM provider instance."""
    provider_id = "provider_beta"
    mock_provider_instance = MockLLMProvider(plugin_id=provider_id)
    mock_plugin_manager_for_llm_mgr.get_plugin_instance.return_value = mock_provider_instance

    # First call - loads and caches
    instance1 = await llm_provider_manager.get_llm_provider(provider_id)
    assert instance1 is mock_provider_instance
    mock_plugin_manager_for_llm_mgr.get_plugin_instance.assert_awaited_once()

    # Second call - should return cached
    instance2 = await llm_provider_manager.get_llm_provider(provider_id)
    assert instance2 is instance1 # Same instance
    # get_plugin_instance should still only have been called once
    mock_plugin_manager_for_llm_mgr.get_plugin_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_llm_provider_config_override(
    llm_provider_manager: LLMProviderManager,
    mock_plugin_manager_for_llm_mgr: PluginManager,
    mock_key_provider_for_llm_mgr: KeyProvider,
):
    """Test config override merges with global config."""
    provider_id = "provider_alpha" # Has global_param: global_alpha
    mock_provider_instance = MockLLMProvider(plugin_id=provider_id)
    mock_plugin_manager_for_llm_mgr.get_plugin_instance.return_value = mock_provider_instance

    override_config = {"model_name": "alpha-override", "local_param": "local_val"}
    instance = await llm_provider_manager.get_llm_provider(provider_id, config_override=override_config)

    assert instance is mock_provider_instance
    call_args = mock_plugin_manager_for_llm_mgr.get_plugin_instance.call_args
    passed_config = call_args.kwargs["config"]

    assert passed_config["model_name"] == "alpha-override" # Overridden
    assert passed_config["global_param"] == "global_alpha" # From global
    assert passed_config["local_param"] == "local_val" # From override
    assert passed_config["key_provider"] is mock_key_provider_for_llm_mgr


@pytest.mark.asyncio
async def test_get_llm_provider_not_found(
    llm_provider_manager: LLMProviderManager,
    mock_plugin_manager_for_llm_mgr: PluginManager,
    caplog: pytest.LogCaptureFixture,
):
    """Test behavior when PluginManager cannot find the plugin."""
    caplog.set_level(logging.ERROR)
    provider_id = "non_existent_provider"
    mock_plugin_manager_for_llm_mgr.get_plugin_instance.return_value = None

    instance = await llm_provider_manager.get_llm_provider(provider_id)

    assert instance is None
    assert f"Failed to load or invalid LLMProviderPlugin for ID '{provider_id}'." in caplog.text


@pytest.mark.asyncio
async def test_get_llm_provider_setup_failure(
    llm_provider_manager: LLMProviderManager,
    mock_plugin_manager_for_llm_mgr: PluginManager,
    caplog: pytest.LogCaptureFixture,
):
    """Test behavior when plugin's setup method fails."""
    caplog.set_level(logging.ERROR)
    provider_id = "fail_setup_provider"
    # Simulate PluginManager returning an instance that will fail setup
    # In the actual PluginManager, if setup fails in get_plugin_instance, it returns None.
    # So, we replicate that behavior for this test.
    mock_plugin_manager_for_llm_mgr.get_plugin_instance.return_value = None
    # We also need to log the "Failed to load" message from the manager
    # if get_plugin_instance itself logs the setup error and returns None.

    instance = await llm_provider_manager.get_llm_provider(provider_id)

    assert instance is None
    assert f"Failed to load or invalid LLMProviderPlugin for ID '{provider_id}'." in caplog.text
    # The specific "Simulated setup failure" would be logged by PluginManager's get_plugin_instance,
    # so this manager's log is about the overall failure to get a valid plugin.


@pytest.mark.asyncio
async def test_llm_provider_manager_teardown(llm_provider_manager: LLMProviderManager):
    """Test the teardown method of the manager."""
    # Primarily checks that it runs and clears internal cache.
    # Individual plugin teardowns are responsibility of PluginManager.
    provider_id = "provider_alpha"
    mock_provider_instance = MockLLMProvider(plugin_id=provider_id)
    llm_provider_manager._plugin_manager.get_plugin_instance.return_value = mock_provider_instance # type: ignore

    await llm_provider_manager.get_llm_provider(provider_id)
    assert provider_id in llm_provider_manager._instantiated_providers

    await llm_provider_manager.teardown()
    assert not llm_provider_manager._instantiated_providers # Cache cleared

###<END-OF-FILE>###
