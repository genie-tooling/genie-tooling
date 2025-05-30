import json
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
    mock_key_provider: KeyProvider # mock_key_provider is from conftest.py
) -> OllamaLLMProviderPlugin:
    provider = OllamaLLMProviderPlugin()
    with patch("httpx.AsyncClient", return_value=mock_httpx_client) as mock_constructor:
        await provider.setup(
            config={"base_url": "http://mock-ollama:11434", "model_name": "test-ollama-model"}
        )
        assert provider._http_client is mock_constructor.return_value
    return provider


@pytest.mark.asyncio
async def test_ollama_setup(mock_key_provider: KeyProvider):
    provider = OllamaLLMProviderPlugin()
    test_base_url = "http://custom-ollama:12345"
    test_model = "ollama-custom"
    test_timeout = 60.0
    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    with patch("httpx.AsyncClient", return_value=mock_client_instance) as MockAsyncClientConstructor:
        await provider.setup(
            config={"base_url": test_base_url, "model_name": test_model, "request_timeout_seconds": test_timeout}
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
    provider_instance._http_client.post.return_value = httpx.Response(
        200, json=mock_ollama_response_data, request=dummy_request
    )
    result = await provider_instance.generate(prompt=prompt, options={"temperature": 0.5})
    assert result["text"] == expected_response_text
    provider_instance._http_client.post.assert_awaited_once()
    # Check payload for generate
    call_args, call_kwargs = provider_instance._http_client.post.call_args
    payload = call_kwargs["json"]
    assert payload["prompt"] == prompt
    assert payload["options"]["temperature"] == 0.5


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
    provider_instance._http_client.post.return_value = httpx.Response(
        200, json=mock_ollama_response_data, request=dummy_request
    )
    result = await provider_instance.chat(messages=messages) # Removed format="json" as it's not a direct param for chat
    assert result["message"]["content"] == expected_assistant_content
    provider_instance._http_client.post.assert_awaited_once()
    call_args, call_kwargs = provider_instance._http_client.post.call_args
    payload = call_kwargs["json"]
    assert payload["messages"] == messages

@pytest.mark.asyncio
async def test_ollama_chat_with_json_format_option(
    ollama_provider: OllamaLLMProviderPlugin, mock_httpx_client: AsyncMock
):
    """Test requesting JSON format via options for chat."""
    provider_instance = await ollama_provider
    messages: List[ChatMessage] = [{"role": "user", "content": "Give me JSON: {'key':'value'}"}]
    expected_json_content = {"key": "value", "status": "parsed"}
    mock_ollama_response_data = {
        "model": "test-ollama-model",
        "message": {"role": "assistant", "content": json.dumps(expected_json_content)}, # LLM returns stringified JSON
        "done": True,
    }
    dummy_request = httpx.Request("POST", f"{provider_instance._base_url}/api/chat")
    provider_instance._http_client.post.return_value = httpx.Response(
        200, json=mock_ollama_response_data, request=dummy_request
    )

    # Request JSON format via kwargs, which should go into payload["format"]
    result = await provider_instance.chat(messages=messages, format="json")

    provider_instance._http_client.post.assert_awaited_once()
    call_args, call_kwargs = provider_instance._http_client.post.call_args
    payload = call_kwargs["json"]
    assert payload["format"] == "json"
    # The plugin currently returns the raw string content from Ollama.
    # Parsing to dict would be an application-level concern or a different method.
    assert result["message"]["content"] == json.dumps(expected_json_content)

@pytest.mark.asyncio
async def test_ollama_generate_with_json_format_option(
    ollama_provider: OllamaLLMProviderPlugin, mock_httpx_client: AsyncMock
):
    """Test requesting JSON format via options for generate."""
    provider_instance = await ollama_provider
    prompt = "Generate JSON: {'data':123}"
    expected_json_string = '{"data": 123, "generated": true}'
    mock_ollama_response_data = {
        "model": "test-ollama-model",
        "response": expected_json_string, # LLM returns stringified JSON
        "done": True,
    }
    dummy_request = httpx.Request("POST", f"{provider_instance._base_url}/api/generate")
    provider_instance._http_client.post.return_value = httpx.Response(
        200, json=mock_ollama_response_data, request=dummy_request
    )

    result = await provider_instance.generate(prompt=prompt, format="json")

    provider_instance._http_client.post.assert_awaited_once()
    call_args, call_kwargs = provider_instance._http_client.post.call_args
    payload = call_kwargs["json"]
    assert payload["format"] == "json"
    assert result["text"] == expected_json_string


@pytest.mark.asyncio
async def test_ollama_http_status_error(
    ollama_provider: OllamaLLMProviderPlugin, mock_httpx_client: AsyncMock, caplog: pytest.LogCaptureFixture
):
    provider_instance = await ollama_provider
    caplog.set_level(logging.ERROR)
    prompt = "Test error."
    error_detail = "Model not found"
    mock_request = httpx.Request("POST", f"{provider_instance._base_url}/api/generate")
    provider_instance._http_client.post.side_effect = httpx.HTTPStatusError(
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
    provider_instance._http_client.post.side_effect = httpx.RequestError("Connection refused", request=mock_request)
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
    provider_instance._http_client.post.return_value = httpx.Response(200, text="Not valid JSON", request=dummy_request)

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
    provider_instance._http_client.post.side_effect = post_side_effect
    info = await provider_instance.get_model_info()

    assert "available_models_brief" in info
    assert "test-ollama-model:latest" in info["available_models_brief"]
    assert info["default_model_details"]["family"] == "test_fam"


@pytest.mark.asyncio
async def test_ollama_teardown(ollama_provider: OllamaLLMProviderPlugin, mock_httpx_client: AsyncMock):
    provider_instance = await ollama_provider
    client_before_teardown = provider_instance._http_client

    await provider_instance.teardown()

    assert client_before_teardown is not None
    client_before_teardown.aclose.assert_awaited_once()
    assert provider_instance._http_client is None
