import logging
from typing import Any, Dict, List, Optional

import pytest
from genie_tooling.core.types import Plugin
from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.security.key_provider import KeyProvider


class DefaultImplLLMProvider(LLMProviderPlugin, Plugin):
    description: str = "Default LLM provider implementation for testing."
    _plugin_id_value: str
    _key_provider_instance_internally_set: Optional[KeyProvider] = None
    _config_instance: Optional[Dict[str, Any]] = None

    def __init__(self, plugin_id_val: str = "default_impl_llm_provider_v1"):
        self._plugin_id_value = plugin_id_val

    @property
    def plugin_id(self) -> str: return self._plugin_id_value

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        await super().setup(config)
        cfg = config or {}
        self._key_provider_instance_internally_set = cfg.get("key_provider")
        self._config_instance = cfg
    async def generate(self, prompt: str, **kwargs: Any) -> Any:
        logger = logging.getLogger(__name__)
        logger.error(f"LLMProviderPlugin '{self.plugin_id}' generate method not implemented.")
        raise NotImplementedError(f"LLMProviderPlugin '{self.plugin_id}' does not implement 'generate'.")
    async def chat(self, messages: List[ChatMessage], **kwargs: Any) -> Any:
        logger = logging.getLogger(__name__)
        logger.error(f"LLMProviderPlugin '{self.plugin_id}' chat method not implemented.")
        raise NotImplementedError(f"LLMProviderPlugin '{self.plugin_id}' does not implement 'chat'.")
    async def get_model_info(self) -> Dict[str, Any]:
        logger = logging.getLogger(__name__)
        logger.debug(f"LLMProviderPlugin '{self.plugin_id}' get_model_info method not implemented. Returning empty dict.")
        return {}

@pytest.fixture
async def default_llm_provider(mock_key_provider: KeyProvider) -> DefaultImplLLMProvider:
    provider = DefaultImplLLMProvider()
    actual_mock_key_provider = await mock_key_provider
    await provider.setup(config={"key_provider": actual_mock_key_provider})
    return provider

@pytest.mark.asyncio
async def test_llm_provider_default_setup(default_llm_provider: DefaultImplLLMProvider):
    actual_provider = await default_llm_provider
    assert actual_provider._key_provider_instance_internally_set is not None
    assert isinstance(actual_provider._key_provider_instance_internally_set, KeyProvider)
    assert actual_provider._config_instance is not None
    assert actual_provider._config_instance.get("key_provider") is actual_provider._key_provider_instance_internally_set

@pytest.mark.asyncio
async def test_llm_provider_default_get_model_info(default_llm_provider: DefaultImplLLMProvider, caplog: pytest.LogCaptureFixture):
    actual_provider = await default_llm_provider
    caplog.set_level(logging.DEBUG)
    info = await actual_provider.get_model_info()
    assert isinstance(info, dict); assert not info
    assert any(f"LLMProviderPlugin '{actual_provider.plugin_id}' get_model_info method not implemented." in rec.message for rec in caplog.records)

@pytest.mark.asyncio
async def test_llm_provider_default_generate_raises_not_implemented(default_llm_provider: DefaultImplLLMProvider):
    actual_provider = await default_llm_provider
    with pytest.raises(NotImplementedError) as excinfo: await actual_provider.generate(prompt="Test prompt")
    assert f"LLMProviderPlugin '{actual_provider.plugin_id}' does not implement 'generate'." in str(excinfo.value)

@pytest.mark.asyncio
async def test_llm_provider_default_chat_raises_not_implemented(default_llm_provider: DefaultImplLLMProvider):
    actual_provider = await default_llm_provider
    messages: List[ChatMessage] = [{"role": "user", "content": "Hello"}]
    with pytest.raises(NotImplementedError) as excinfo: await actual_provider.chat(messages=messages)
    assert f"LLMProviderPlugin '{actual_provider.plugin_id}' does not implement 'chat'." in str(excinfo.value)

@pytest.mark.asyncio
async def test_llm_provider_default_teardown(default_llm_provider: DefaultImplLLMProvider, caplog: pytest.LogCaptureFixture):
    actual_provider = await default_llm_provider
    caplog.set_level(logging.DEBUG); await actual_provider.teardown(); assert True
