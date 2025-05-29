### tests/unit/llm_providers/impl/test_ollama_provider.py
import logging
from typing import List
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from genie_tooling.llm_providers.impl.ollama_provider import OllamaLLMProviderPlugin
from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.security.key_provider import KeyProvider


@pytest.fixture
def mock_httpx_client() -> AsyncMock:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock()
    client.aclose = AsyncMock()
    return client

@pytest.fixture
async def ollama_provider(
    mock_httpx_client: AsyncMock,
    mock_key_provider: KeyProvider # Keep for type hint, but Ollama's setup won't use it directly
) -> OllamaLLMProviderPlugin:
    provider = OllamaLLMProviderPlugin()
    # actual_mock_key_provider = await mock_key_provider # Not strictly needed for Ollama setup call

    with patch("httpx.AsyncClient", return_value=mock_httpx_client) as mock_constructor:
        # Ollama's setup does not take key_provider as a direct argument.
        # It only takes 'config'.
        await provider.setup(
            config={"base_url": "http://mock-ollama:11434", "model_name": "test-ollama-model"}
            # Removed: key_provider=actual_mock_key_provider
        )
        assert provider._http_client is mock_constructor.return_value
    return provider


@pytest.mark.asyncio
async def test_ollama_setup(mock_key_provider: KeyProvider): # Keep mock_key_provider if tests need it for other things
    provider = OllamaLLMProviderPlugin()
    test_base_url = "http://custom-ollama:12345"
    test_model = "ollama-custom"
    test_timeout = 60.0
    # actual_mock_kp = await mock_key_provider # Not passed to setup

    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)

    with patch("httpx.AsyncClient", return_value=mock_client_instance) as MockAsyncClientConstructor:
        await provider.setup(
            config={"base_url": test_base_url, "model_name": test_model, "request_timeout_seconds": test_timeout}
            # Removed: key_provider=actual_mock_kp
        )
        MockAsyncClientConstructor.assert_called_once_with(timeout=test_timeout)
        assert provider._http_client is mock_client_instance
        assert isinstance(provider._http_client, AsyncMock)
        assert provider._base_url == test_base_url
    await provider.teardown()


@pytest.mark.asyncio
async def test_ollama_generate_success(
    ollama_provider: OllamaLLMProviderPlugin,
    mock_httpx_client: AsyncMock
):
    provider_instance = await ollama_provider
    prompt = "Explain Llamas."
    expected_response_text = "Llamas are South American camelids."
    mock_ollama_response_data = {
        "model": "test-ollama-model","response": expected_response_text, "done": True,
        "prompt_eval_count": 10, "eval_count": 5,
    }
    dummy_request = httpx.Request("POST", f"{provider_instance._base_url}/api/generate")
    provider_instance._http_client.post.return_value = httpx.Response( # type: ignore
        200, json=mock_ollama_response_data, request=dummy_request
    )

    result = await provider_instance.generate(prompt=prompt, options={"temperature": 0.5})

    assert result["text"] == expected_response_text
    provider_instance._http_client.post.assert_awaited_once() # type: ignore


@pytest.mark.asyncio
async def test_ollama_chat_success(
    ollama_provider: OllamaLLMProviderPlugin, mock_httpx_client: AsyncMock
):
    provider_instance = await ollama_provider
    messages: List[ChatMessage] = [{"role": "user", "content": "Hello there."}]
    expected_assistant_content = "General Kenobi!"
    mock_ollama_response_data = {
        "model": "test-ollama-model",
        "message": {"role": "assistant", "content": expected_assistant_content},
        "done": True, "prompt_eval_count": 8, "eval_count": 4,
    }
    dummy_request = httpx.Request("POST", f"{provider_instance._base_url}/api/chat")
    provider_instance._http_client.post.return_value = httpx.Response( # type: ignore
        200, json=mock_ollama_response_data, request=dummy_request
    )
    result = await provider_instance.chat(messages=messages, format="json")
    assert result["message"]["content"] == expected_assistant_content
    provider_instance._http_client.post.assert_awaited_once() # type: ignore


@pytest.mark.asyncio
async def test_ollama_http_status_error(
    ollama_provider: OllamaLLMProviderPlugin, mock_httpx_client: AsyncMock, caplog: pytest.LogCaptureFixture
):
    provider_instance = await ollama_provider
    caplog.set_level(logging.ERROR)
    prompt = "Test error."
    error_detail = "Model not found"
    mock_request = httpx.Request("POST", f"{provider_instance._base_url}/api/generate")
    provider_instance._http_client.post.side_effect = httpx.HTTPStatusError( # type: ignore
        message="404 Not Found", request=mock_request,
        response=httpx.Response(404, json={"error": error_detail}, request=mock_request)
    )
    with pytest.raises(RuntimeError) as excinfo:
        await provider_instance.generate(prompt=prompt)
    assert f"Ollama API error: 404 - {error_detail}" in str(excinfo.value)


@pytest.mark.asyncio
async def test_ollama_request_error(
    ollama_provider: OllamaLLMProviderPlugin, mock_httpx_client: AsyncMock, caplog: pytest.LogCaptureFixture
):
    provider_instance = await ollama_provider
    caplog.set_level(logging.ERROR)
    prompt = "Test connection error."
    mock_request = httpx.Request("POST", f"{provider_instance._base_url}/api/generate")
    provider_instance._http_client.post.side_effect = httpx.RequestError("Connection refused", request=mock_request) # type: ignore
    with pytest.raises(RuntimeError) as excinfo:
        await provider_instance.generate(prompt=prompt)
    assert "Ollama request failed: Connection refused" in str(excinfo.value)


@pytest.mark.asyncio
async def test_ollama_json_decode_error(
    ollama_provider: OllamaLLMProviderPlugin, mock_httpx_client: AsyncMock, caplog: pytest.LogCaptureFixture
):
    provider_instance = await ollama_provider
    caplog.set_level(logging.ERROR)
    prompt = "Test JSON error."
    dummy_request = httpx.Request("POST", f"{provider_instance._base_url}/api/generate")
    provider_instance._http_client.post.return_value = httpx.Response(200, text="Not valid JSON", request=dummy_request) # type: ignore

    with pytest.raises(RuntimeError) as excinfo:
        await provider_instance.generate(prompt=prompt)
    assert "Ollama response JSON decode error" in str(excinfo.value)


@pytest.mark.asyncio
async def test_ollama_get_model_info_success(
    ollama_provider: OllamaLLMProviderPlugin, mock_httpx_client: AsyncMock
):
    provider_instance = await ollama_provider
    mock_tags_response = {"models": [{"name": "test-ollama-model:latest"}]}
    mock_show_response = {"details": {"family": "test_fam"}}
    dummy_request_tags = httpx.Request("POST", f"{provider_instance._base_url}/api/tags")
    dummy_request_show = httpx.Request("POST", f"{provider_instance._base_url}/api/show")

    async def post_side_effect(url: str, json: dict):
        if url.endswith("/api/tags"):
            return httpx.Response(200, json=mock_tags_response, request=dummy_request_tags)
        if url.endswith("/api/show") and json.get("name") == provider_instance._default_model:
            return httpx.Response(200, json=mock_show_response, request=dummy_request_show)
        return httpx.Response(404, request=httpx.Request("POST", url))
    provider_instance._http_client.post.side_effect = post_side_effect # type: ignore
    info = await provider_instance.get_model_info()

    assert "available_models_brief" in info
    assert "test-ollama-model:latest" in info["available_models_brief"]
    assert info["default_model_details"]["family"] == "test_fam" # type: ignore


@pytest.mark.asyncio
async def test_ollama_teardown(ollama_provider: OllamaLLMProviderPlugin, mock_httpx_client: AsyncMock):
    provider_instance = await ollama_provider
    client_before_teardown = provider_instance._http_client

    await provider_instance.teardown()

    assert client_before_teardown is not None
    client_before_teardown.aclose.assert_awaited_once() # type: ignore
    assert provider_instance._http_client is None
