import logging
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.llm_providers.manager import LLMProviderManager
from genie_tooling.security.key_provider import KeyProvider

# This MockLLMProvider will be instantiated by LLMProviderManager.
# Its plugin_id will be what's defined in the class.
class MockLLMProvider(LLMProviderPlugin):
    plugin_id: str = "actual_mock_llm_provider_id_v1" # Define actual ID here
    description: str = "Mock LLM Provider for Manager Tests"
    
    setup_config_received: Optional[Dict[str, Any]] = None
    key_provider_received: Optional[KeyProvider] = None
    teardown_called: bool = False

    # No __init__ needed if plugin_id is a class var, or a simple one if required by test structure
    def __init__(self): # Removed plugin_id_val
        self.setup_config_received = None
        self.key_provider_received = None
        self.teardown_called = False
        
    async def setup(self, config: Optional[Dict[str, Any]]) -> None: 
        self.setup_config_received = config
        self.key_provider_received = config.get("key_provider") if config else None
        if config and config.get("fail_setup"): raise RuntimeError("Simulated setup failure")
    async def generate(self, prompt: str, **kwargs: Any) -> Any: return {"text": f"Generated: {prompt}", "finish_reason": "stop", "usage": None, "raw_response": {}}
    async def chat(self, messages: List[Any], **kwargs: Any) -> Any: return {"message": {"role": "assistant", "content": "Chat response"}, "finish_reason": "stop", "usage": None, "raw_response": {}}
    async def teardown(self) -> None: self.teardown_called = True

@pytest.fixture
def mock_plugin_manager_for_llm_mgr(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.list_discovered_plugin_classes = mocker.MagicMock(return_value={}) 
    return pm

@pytest.fixture
async def mock_key_provider_for_llm_mgr(mock_key_provider: KeyProvider) -> KeyProvider:
     return await mock_key_provider

@pytest.fixture
def mock_middleware_config() -> MiddlewareConfig:
    # Use the actual plugin_id of MockLLMProvider if that's what we're testing against
    return MiddlewareConfig(
        llm_provider_configurations={
            MockLLMProvider.plugin_id: {"model_name": "alpha-model", "global_param": "global_alpha"},
            "provider_beta_id_for_test": {"model_name": "beta-model"} # Another ID for other tests
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
    # The provider_id we request from the manager
    requested_provider_id = MockLLMProvider.plugin_id 
    
    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.return_value = {
        requested_provider_id: MockLLMProvider # Manager will find this class
    }

    instance = await llm_provider_manager.get_llm_provider(requested_provider_id)
    
    assert instance is not None
    assert isinstance(instance, MockLLMProvider)
    assert instance.plugin_id == MockLLMProvider.plugin_id # Instance has its own defined ID
    assert instance.setup_config_received is not None
    # Check against the config for MockLLMProvider.plugin_id
    assert instance.setup_config_received.get("model_name") == "alpha-model"
    assert instance.setup_config_received.get("global_param") == "global_alpha"
    assert instance.key_provider_received is mock_key_provider_for_llm_mgr

@pytest.mark.asyncio
async def test_get_llm_provider_cached_instance(llm_provider_manager: LLMProviderManager, mock_plugin_manager_for_llm_mgr: PluginManager):
    requested_provider_id = MockLLMProvider.plugin_id
    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.return_value = {
        requested_provider_id: MockLLMProvider
    }

    instance1 = await llm_provider_manager.get_llm_provider(requested_provider_id)
    assert instance1 is not None
    
    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.reset_mock()

    instance2 = await llm_provider_manager.get_llm_provider(requested_provider_id)
    assert instance2 is instance1
    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.assert_not_called()


@pytest.mark.asyncio
async def test_get_llm_provider_config_override(llm_provider_manager: LLMProviderManager, mock_plugin_manager_for_llm_mgr: PluginManager, mock_key_provider_for_llm_mgr: KeyProvider):
    requested_provider_id = MockLLMProvider.plugin_id
    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.return_value = {
        requested_provider_id: MockLLMProvider
    }
    
    override_config = {"model_name": "alpha-override", "local_param": "local_val"}
    instance = await llm_provider_manager.get_llm_provider(requested_provider_id, config_override=override_config)

    assert instance is not None
    assert isinstance(instance, MockLLMProvider)
    assert instance.setup_config_received is not None
    assert instance.setup_config_received.get("model_name") == "alpha-override"
    assert instance.setup_config_received.get("global_param") == "global_alpha" 
    assert instance.setup_config_received.get("local_param") == "local_val"
    assert instance.key_provider_received is mock_key_provider_for_llm_mgr

@pytest.mark.asyncio
async def test_llm_provider_manager_teardown(llm_provider_manager: LLMProviderManager, mock_plugin_manager_for_llm_mgr: PluginManager):
    requested_provider_id = MockLLMProvider.plugin_id
    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.return_value = {
        requested_provider_id: MockLLMProvider
    }
    
    instance = await llm_provider_manager.get_llm_provider(requested_provider_id)
    assert instance is not None
    assert requested_provider_id in llm_provider_manager._instantiated_providers # Check by the ID it was requested with
    
    await llm_provider_manager.teardown()
    assert not llm_provider_manager._instantiated_providers
    assert instance.teardown_called is True
