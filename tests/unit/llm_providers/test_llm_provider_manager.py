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
    _plugin_id_value: str
    description: str = "Mock LLM Provider for Manager Tests"
    setup_config_received: Optional[Dict[str, Any]] = None
    key_provider_received: Optional[KeyProvider] = None
    teardown_called: bool = False

    def __init__(self, plugin_id_val: str = "mock_llm_provider_v1"):
        self._plugin_id_value = plugin_id_val
        self.setup_config_received = None
        self.key_provider_received = None
        self.teardown_called = False

    @property
    def plugin_id(self) -> str:
        return self._plugin_id_value
        
    async def setup(self, config: Optional[Dict[str, Any]]) -> None: 
        self.setup_config_received = config
        self.key_provider_received = config.get("key_provider") if config else None
        if config and config.get("fail_setup"): raise RuntimeError("Simulated setup failure")
    async def generate(self, prompt: str, **kwargs: Any) -> Any: return {"text": f"Generated: {prompt}", "finish_reason": "stop", "usage": None, "raw_response": {}}
    async def chat(self, messages: List[Any], **kwargs: Any) -> Any: return {"message": {"role": "assistant", "content": "Chat response"}, "finish_reason": "stop", "usage": None, "raw_response": {}}
    async def teardown(self) -> None: self.teardown_called = True # Mark teardown

@pytest.fixture
def mock_plugin_manager_for_llm_mgr(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    # LLMProviderManager now gets classes, not instances from PluginManager
    pm.list_discovered_plugin_classes = mocker.MagicMock(return_value={}) 
    return pm

@pytest.fixture
async def mock_key_provider_for_llm_mgr(mock_key_provider: KeyProvider) -> KeyProvider:
     return await mock_key_provider # Ensure the conftest fixture is awaited

@pytest.fixture
def mock_middleware_config() -> MiddlewareConfig:
    return MiddlewareConfig(
        llm_provider_configurations={
            "provider_alpha": {"model_name": "alpha-model", "global_param": "global_alpha"},
            "provider_beta": {"model_name": "beta-model"}
        }
    )

@pytest.fixture
def llm_provider_manager(mock_plugin_manager_for_llm_mgr: PluginManager, mock_key_provider_for_llm_mgr: KeyProvider, mock_middleware_config: MiddlewareConfig) -> LLMProviderManager:
    return LLMProviderManager(
        plugin_manager=mock_plugin_manager_for_llm_mgr,
        key_provider=mock_key_provider_for_llm_mgr,
        config=mock_middleware_config
    )

@pytest.mark.asyncio
async def test_get_llm_provider_success_new_instance(llm_provider_manager: LLMProviderManager, mock_plugin_manager_for_llm_mgr: PluginManager, mock_key_provider_for_llm_mgr: KeyProvider):
    provider_id = "provider_alpha"
    # Configure PluginManager to return the MockLLMProvider class
    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.return_value = {
        provider_id: MockLLMProvider
    }

    instance = await llm_provider_manager.get_llm_provider(provider_id)
    
    assert instance is not None
    assert isinstance(instance, MockLLMProvider)
    assert instance.plugin_id == provider_id # MockLLMProvider sets its own plugin_id
    assert instance.setup_config_received is not None
    assert instance.setup_config_received.get("model_name") == "alpha-model"
    assert instance.setup_config_received.get("global_param") == "global_alpha"
    assert instance.key_provider_received is mock_key_provider_for_llm_mgr

@pytest.mark.asyncio
async def test_get_llm_provider_cached_instance(llm_provider_manager: LLMProviderManager, mock_plugin_manager_for_llm_mgr: PluginManager):
    provider_id = "provider_beta"
    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.return_value = {
        provider_id: MockLLMProvider
    }

    instance1 = await llm_provider_manager.get_llm_provider(provider_id)
    assert instance1 is not None
    
    # To check caching, ensure list_discovered_plugin_classes is not called again for the same class retrieval logic
    # or more directly, that the instance is the same.
    # The internal logic of PluginManager.list_discovered_plugin_classes itself isn't what we're testing for caching here,
    # but rather LLMProviderManager's caching.
    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.reset_mock() # Reset to see if it's called again

    instance2 = await llm_provider_manager.get_llm_provider(provider_id)
    assert instance2 is instance1
    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.assert_not_called()


@pytest.mark.asyncio
async def test_get_llm_provider_config_override(llm_provider_manager: LLMProviderManager, mock_plugin_manager_for_llm_mgr: PluginManager, mock_key_provider_for_llm_mgr: KeyProvider):
    provider_id = "provider_alpha"
    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.return_value = {
        provider_id: MockLLMProvider
    }
    
    override_config = {"model_name": "alpha-override", "local_param": "local_val"}
    instance = await llm_provider_manager.get_llm_provider(provider_id, config_override=override_config)

    assert instance is not None
    assert isinstance(instance, MockLLMProvider)
    assert instance.setup_config_received is not None
    assert instance.setup_config_received.get("model_name") == "alpha-override" # Override applied
    assert instance.setup_config_received.get("global_param") == "global_alpha" # Global preserved
    assert instance.setup_config_received.get("local_param") == "local_val"    # Override applied
    assert instance.key_provider_received is mock_key_provider_for_llm_mgr

@pytest.mark.asyncio
async def test_llm_provider_manager_teardown(llm_provider_manager: LLMProviderManager, mock_plugin_manager_for_llm_mgr: PluginManager):
    provider_id = "provider_alpha"
    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.return_value = {
        provider_id: MockLLMProvider
    }
    
    # Get an instance to populate the cache
    instance = await llm_provider_manager.get_llm_provider(provider_id)
    assert instance is not None
    assert provider_id in llm_provider_manager._instantiated_providers
    
    await llm_provider_manager.teardown()
    assert not llm_provider_manager._instantiated_providers # Cache should be cleared
    assert instance.teardown_called is True # Check that the plugin's teardown was called
