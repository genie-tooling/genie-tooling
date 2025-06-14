### tests/unit/llm_providers/test_llm_provider_manager.py
from typing import Any, Dict, List, Optional

import pytest
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.llm_providers.manager import LLMProviderManager
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.token_usage.manager import (
    TokenUsageManager,
)


class MockLLMProvider(LLMProviderPlugin):
    plugin_id: str = "actual_mock_llm_provider_id_v1"
    description: str = "Mock LLM Provider for Manager Tests"

    setup_config_received: Optional[Dict[str, Any]] = None
    key_provider_received: Optional[KeyProvider] = None
    token_usage_manager_received: Optional[TokenUsageManager] = None
    teardown_called: bool = False

    def __init__(self):
        self.setup_config_received = None
        self.key_provider_received = None
        self.token_usage_manager_received = None
        self.teardown_called = False

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        self.setup_config_received = config
        if config:
            self.key_provider_received = config.get("key_provider")
            self.token_usage_manager_received = config.get("token_usage_manager")
        if config and config.get("fail_setup"):
            raise RuntimeError("Simulated setup failure")
    async def generate(self, prompt: str, **kwargs: Any) -> Any:
        return {"text": f"Generated: {prompt}", "finish_reason": "stop", "usage": None, "raw_response": {}}
    async def chat(self, messages: List[Any], **kwargs: Any) -> Any:
        return {"message": {"role": "assistant", "content": "Chat response"}, "finish_reason": "stop", "usage": None, "raw_response": {}}
    async def teardown(self) -> None:
        self.teardown_called = True

@pytest.fixture()
def mock_plugin_manager_for_llm_mgr(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.list_discovered_plugin_classes = mocker.MagicMock(return_value={})
    return pm

@pytest.fixture()
async def mock_key_provider_for_llm_mgr(mock_key_provider: KeyProvider) -> KeyProvider:
     return await mock_key_provider

@pytest.fixture()
def mock_token_usage_manager_for_llm_mgr(mocker) -> TokenUsageManager:
    return mocker.MagicMock(spec=TokenUsageManager)

@pytest.fixture()
def mock_middleware_config() -> MiddlewareConfig:
    return MiddlewareConfig(
        llm_provider_configurations={
            MockLLMProvider.plugin_id: {"model_name": "alpha-model", "global_param": "global_alpha"},
            "provider_beta_id_for_test": {"model_name": "beta-model"}
        }
    )

@pytest.fixture()
async def llm_provider_manager(
    mock_plugin_manager_for_llm_mgr: PluginManager,
    mock_key_provider_for_llm_mgr: KeyProvider,
    mock_middleware_config: MiddlewareConfig,
    mock_token_usage_manager_for_llm_mgr: TokenUsageManager
) -> LLMProviderManager:
    # Await the async fixture to get the actual KeyProvider instance
    actual_key_provider = await mock_key_provider_for_llm_mgr
    return LLMProviderManager(
        plugin_manager=mock_plugin_manager_for_llm_mgr,
        key_provider=actual_key_provider, # Use the awaited instance
        config=mock_middleware_config,
        token_usage_manager=mock_token_usage_manager_for_llm_mgr
    )

@pytest.mark.asyncio()
async def test_get_llm_provider_success_new_instance(
    llm_provider_manager: LLMProviderManager,
    mock_plugin_manager_for_llm_mgr: PluginManager,
    mock_token_usage_manager_for_llm_mgr: TokenUsageManager
):
    manager = await llm_provider_manager
    requested_provider_id = MockLLMProvider.plugin_id

    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.return_value = {
        requested_provider_id: MockLLMProvider
    }

    instance = await manager.get_llm_provider(requested_provider_id)

    assert instance is not None
    assert isinstance(instance, MockLLMProvider)
    assert instance.plugin_id == MockLLMProvider.plugin_id
    assert instance.setup_config_received is not None
    assert instance.setup_config_received.get("model_name") == "alpha-model"
    assert instance.setup_config_received.get("global_param") == "global_alpha"
    assert instance.key_provider_received is manager._key_provider
    assert instance.token_usage_manager_received is mock_token_usage_manager_for_llm_mgr

@pytest.mark.asyncio()
async def test_get_llm_provider_cached_instance(llm_provider_manager: LLMProviderManager, mock_plugin_manager_for_llm_mgr: PluginManager):
    manager = await llm_provider_manager
    requested_provider_id = MockLLMProvider.plugin_id
    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.return_value = {
        requested_provider_id: MockLLMProvider
    }

    instance1 = await manager.get_llm_provider(requested_provider_id)
    assert instance1 is not None

    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.reset_mock()

    instance2 = await manager.get_llm_provider(requested_provider_id)
    assert instance2 is instance1
    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.assert_not_called()


@pytest.mark.asyncio()
async def test_get_llm_provider_config_override(
    llm_provider_manager: LLMProviderManager,
    mock_plugin_manager_for_llm_mgr: PluginManager,
    mock_token_usage_manager_for_llm_mgr: TokenUsageManager
):
    manager = await llm_provider_manager
    requested_provider_id = MockLLMProvider.plugin_id
    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.return_value = {
        requested_provider_id: MockLLMProvider
    }

    override_config = {"model_name": "alpha-override", "local_param": "local_val"}
    instance = await manager.get_llm_provider(requested_provider_id, config_override=override_config)

    assert instance is not None
    assert isinstance(instance, MockLLMProvider)
    assert instance.setup_config_received is not None
    assert instance.setup_config_received.get("model_name") == "alpha-override"
    assert instance.setup_config_received.get("global_param") == "global_alpha"
    assert instance.setup_config_received.get("local_param") == "local_val"
    assert instance.key_provider_received is manager._key_provider
    assert instance.token_usage_manager_received is mock_token_usage_manager_for_llm_mgr

@pytest.mark.asyncio()
async def test_llm_provider_manager_teardown(llm_provider_manager: LLMProviderManager, mock_plugin_manager_for_llm_mgr: PluginManager):
    manager = await llm_provider_manager
    requested_provider_id = MockLLMProvider.plugin_id
    mock_plugin_manager_for_llm_mgr.list_discovered_plugin_classes.return_value = {
        requested_provider_id: MockLLMProvider
    }

    instance = await manager.get_llm_provider(requested_provider_id)
    assert instance is not None
    assert requested_provider_id in manager._instantiated_providers

    await manager.teardown()
    assert not manager._instantiated_providers
    assert instance.teardown_called is True
