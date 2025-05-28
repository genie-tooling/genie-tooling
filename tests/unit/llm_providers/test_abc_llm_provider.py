### tests/unit/llm_providers/test_abc_llm_provider.py
import logging
from typing import Any, Dict, List, Optional

import pytest
from genie_tooling.core.types import Plugin
from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.llm_providers.types import (
    ChatMessage,
)
from genie_tooling.security.key_provider import (
    KeyProvider,
)


class DefaultImplLLMProvider(LLMProviderPlugin, Plugin):
    plugin_id: str = "default_impl_llm_provider_v1"
    description: str = "Default LLM provider implementation for testing."
    _key_provider_instance_internally_set: Optional[KeyProvider] = None
    _config_instance: Optional[Dict[str, Any]] = None

    async def setup(
        self, config: Optional[Dict[str, Any]]
    ) -> None: # Changed signature
        # Call Plugin's default setup first
        # Superclass setup doesn't take key_provider as a direct arg anymore
        await super().setup(config)
        cfg = config or {}
        self._key_provider_instance_internally_set = cfg.get("key_provider")
        self._config_instance = cfg


@pytest.fixture
async def default_llm_provider(mock_key_provider: KeyProvider) -> DefaultImplLLMProvider:
    provider = DefaultImplLLMProvider()
    actual_mock_kp = await mock_key_provider # mock_key_provider from conftest IS async
    # Pass key_provider inside the config dictionary
    await provider.setup(config={"key_provider": actual_mock_kp})
    return provider


@pytest.mark.asyncio
async def test_llm_provider_default_setup(
    default_llm_provider: DefaultImplLLMProvider,
): # Removed mock_key_provider from direct injection as it's in the fixture
    provider_instance = await default_llm_provider
    assert provider_instance._key_provider_instance_internally_set is not None
    assert isinstance(provider_instance._key_provider_instance_internally_set, KeyProvider)
    # Check that the config passed to setup (which includes key_provider) was stored
    assert provider_instance._config_instance is not None
    assert provider_instance._config_instance.get("key_provider") is provider_instance._key_provider_instance_internally_set


@pytest.mark.asyncio
async def test_llm_provider_default_get_model_info(
    default_llm_provider: DefaultImplLLMProvider, caplog: pytest.LogCaptureFixture
):
    provider_instance = await default_llm_provider
    caplog.set_level(logging.DEBUG)
    info = await provider_instance.get_model_info()

    assert isinstance(info, dict)
    assert not info
    assert any(
        f"LLMProviderPlugin '{provider_instance.plugin_id}' get_model_info method not implemented."
        in rec.message for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_llm_provider_default_generate_raises_not_implemented(
    default_llm_provider: DefaultImplLLMProvider
):
    provider_instance = await default_llm_provider
    with pytest.raises(NotImplementedError) as excinfo:
        await provider_instance.generate(prompt="Test prompt")
    assert f"LLMProviderPlugin '{provider_instance.plugin_id}' does not implement 'generate'." in str(excinfo.value)


@pytest.mark.asyncio
async def test_llm_provider_default_chat_raises_not_implemented(
    default_llm_provider: DefaultImplLLMProvider
):
    provider_instance = await default_llm_provider
    messages: List[ChatMessage] = [{"role": "user", "content": "Hello"}]
    with pytest.raises(NotImplementedError) as excinfo:
        await provider_instance.chat(messages=messages)
    assert f"LLMProviderPlugin '{provider_instance.plugin_id}' does not implement 'chat'." in str(excinfo.value)


@pytest.mark.asyncio
async def test_llm_provider_default_teardown(
    default_llm_provider: DefaultImplLLMProvider, caplog: pytest.LogCaptureFixture
):
    provider_instance = await default_llm_provider
    caplog.set_level(logging.DEBUG)
    await provider_instance.teardown()
    # Default teardown in Plugin is just pass, so no specific log expected from Plugin's default.
    # The llm_provider_abc itself doesn't add logging to teardown.
    # Check a log from this module's default_llm_provider setup if desired or if teardown had specific logs.
    # For now, just assert it runs.
    assert True