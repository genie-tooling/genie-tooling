### tests/unit/llm_providers/impl/test_ollama_provider.py
import json
import logging
from typing import Any, AsyncIterable, List
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from genie_tooling.llm_providers.impl.ollama_provider import OllamaLLMProviderPlugin
from genie_tooling.llm_providers.types import (
    ChatMessage,
    LLMChatChunk,
    LLMCompletionChunk,
)

# KeyProvider is no longer needed here as Ollama provider doesn't use it.
# from genie_tooling.security.key_provider import KeyProvider

# --- Helper for Streaming ---
async def consume_async_iterable(
    iterable: AsyncIterable[Any],
) -> List[Any]:
    return [item async for item in iterable]

@pytest.fixture()
def mock_httpx_client() -> AsyncMock:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock()
    client.aclose = AsyncMock()
    return client

@pytest.fixture()
async def ollama_provider(
    mock_httpx_client: AsyncMock,
    # mock_key_provider: KeyProvider : Ollama provider does not use KeyProvider
) -> OllamaLLMProviderPlugin:
    provider = OllamaLLMProviderPlugin()
    with patch("genie_tooling.llm_providers.impl.ollama_provider.httpx.AsyncClient", return_value=mock_httpx_client) as mock_constructor:
        await provider.setup(
            config={"base_url": "http://mock-ollama:11434", "model_name": "test-ollama-model"}
        )
        assert provider._http_client is mock_constructor.return_value
    return provider


@pytest.mark.asyncio()
async def test_ollama_setup(
    # mock_key_provider: KeyProvider
):
    provider = OllamaLLMProviderPlugin()
    test_base_url = "http://custom-ollama:12345"
    test_model = "ollama-custom"
    test_timeout = 60.0
    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    with patch("genie_tooling.llm_providers.impl.ollama_provider.httpx.AsyncClient", return_value=mock_client_instance) as MockAsyncClientConstructor:
        await provider.setup(
            config={"base_url": test_base_url, "model_name": test_model, "request_timeout_seconds": test_timeout}
            # No key_provider needed in config for Ollama
        )
        MockAsyncClientConstructor.assert_called_once_with(timeout=test_timeout)
        assert provider._http_client is mock_client_instance
        assert isinstance(provider._http_client, AsyncMock)
        assert provider._base_url == test_base_url
    await provider.teardown()


@pytest.mark.asyncio()
async def test_ollama_generate_success(
    ollama_provider: OllamaLLMProviderPlugin,
    mock_httpx_client: AsyncMock # mock_httpx_client is already part of ollama_provider fixture
):
    provider_instance = await ollama_provider # ollama_provider is already an instance
    prompt = "Explain Llamas."
    expected_response_text = "Llamas are South American camelids."
    mock_ollama_response_data = {
        "model": "test-ollama-model","response": expected_response_text, "done": True,
        "prompt_eval_count": 10, "eval_count": 5,
    }
    dummy_request = httpx.Request("POST", f"{provider_instance._base_url}/api/generate")
    # Access the client through the provider instance
    provider_instance._http_client.post.return_value = httpx.Response( # type: ignore
        200, json=mock_ollama_response_data, request=dummy_request
    )
    result = await provider_instance.generate(prompt=prompt, options={"temperature": 0.5})
    assert result["text"] == expected_response_text
    provider_instance._http_client.post.assert_awaited_once() # type: ignore
    # Check payload for generate
    call_args, call_kwargs = provider_instance._http_client.post.call_args # type: ignore
    payload = call_kwargs["json"]
    assert payload["prompt"] == prompt
    assert payload["options"]["temperature"] == 0.5


@pytest.mark.asyncio()
async def test_ollama_chat_success(
    ollama_provider: OllamaLLMProviderPlugin, # ollama_provider is already an instance
    mock_httpx_client: AsyncMock # mock_httpx_client is already part of ollama_provider fixture
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
    result = await provider_instance.chat(messages=messages)
    assert result["message"]["content"] == expected_assistant_content
    provider_instance._http_client.post.assert_awaited_once() # type: ignore
    call_args, call_kwargs = provider_instance._http_client.post.call_args # type: ignore
    payload = call_kwargs["json"]
    assert payload["messages"] == messages

@pytest.mark.asyncio()
async def test_ollama_chat_with_json_format_option(
    ollama_provider: OllamaLLMProviderPlugin, # ollama_provider is already an instance
    mock_httpx_client: AsyncMock # mock_httpx_client is already part of ollama_provider fixture
):
    """Test requesting JSON format via options for chat."""
    provider_instance = await ollama_provider
    messages: List[ChatMessage] = [{"role": "user", "content": "Give me JSON: {'key':'value'}"}]
    expected_json_content = {"key": "value", "status": "parsed"}
    mock_ollama_response_data = {
        "model": "test-ollama-model",
        "message": {"role": "assistant", "content": json.dumps(expected_json_content)},
        "done": True,
    }
    dummy_request = httpx.Request("POST", f"{provider_instance._base_url}/api/chat")
    provider_instance._http_client.post.return_value = httpx.Response( # type: ignore
        200, json=mock_ollama_response_data, request=dummy_request
    )

    result = await provider_instance.chat(messages=messages, format="json")

    provider_instance._http_client.post.assert_awaited_once() # type: ignore
    call_args, call_kwargs = provider_instance._http_client.post.call_args # type: ignore
    payload = call_kwargs["json"]
    assert payload["format"] == "json"
    assert result["message"]["content"] == json.dumps(expected_json_content)

@pytest.mark.asyncio()
async def test_ollama_generate_with_json_format_option(
    ollama_provider: OllamaLLMProviderPlugin, # ollama_provider is already an instance
    mock_httpx_client: AsyncMock # mock_httpx_client is already part of ollama_provider fixture
):
    """Test requesting JSON format via options for generate."""
    provider_instance = await ollama_provider
    prompt = "Generate JSON: {'data':123}"
    expected_json_string = '{"data": 123, "generated": true}'
    mock_ollama_response_data = {
        "model": "test-ollama-model",
        "response": expected_json_string,
        "done": True,
    }
    dummy_request = httpx.Request("POST", f"{provider_instance._base_url}/api/generate")
    provider_instance._http_client.post.return_value = httpx.Response( # type: ignore
        200, json=mock_ollama_response_data, request=dummy_request
    )

    result = await provider_instance.generate(prompt=prompt, format="json")

    provider_instance._http_client.post.assert_awaited_once() # type: ignore
    call_args, call_kwargs = provider_instance._http_client.post.call_args # type: ignore
    payload = call_kwargs["json"]
    assert payload["format"] == "json"
    assert result["text"] == expected_json_string


@pytest.mark.asyncio()
async def test_ollama_http_status_error(
    ollama_provider: OllamaLLMProviderPlugin, # ollama_provider is already an instance
    mock_httpx_client: AsyncMock, # mock_httpx_client is already part of ollama_provider fixture
    caplog: pytest.LogCaptureFixture
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


@pytest.mark.asyncio()
async def test_ollama_request_error(
    ollama_provider: OllamaLLMProviderPlugin, # ollama_provider is already an instance
    mock_httpx_client: AsyncMock, # mock_httpx_client is already part of ollama_provider fixture
    caplog: pytest.LogCaptureFixture
):
    provider_instance = await ollama_provider
    caplog.set_level(logging.ERROR)
    prompt = "Test connection error."
    mock_request = httpx.Request("POST", f"{provider_instance._base_url}/api/generate")
    provider_instance._http_client.post.side_effect = httpx.RequestError("Connection refused", request=mock_request) # type: ignore
    with pytest.raises(RuntimeError) as excinfo:
        await provider_instance.generate(prompt=prompt)
    assert "Ollama request failed: Connection refused" in str(excinfo.value)


@pytest.mark.asyncio()
async def test_ollama_json_decode_error(
    ollama_provider: OllamaLLMProviderPlugin, # ollama_provider is already an instance
    mock_httpx_client: AsyncMock, # mock_httpx_client is already part of ollama_provider fixture
    caplog: pytest.LogCaptureFixture
):
    provider_instance = await ollama_provider
    caplog.set_level(logging.ERROR)
    prompt = "Test JSON error."
    dummy_request = httpx.Request("POST", f"{provider_instance._base_url}/api/generate")
    provider_instance._http_client.post.return_value = httpx.Response(200, text="Not valid JSON", request=dummy_request) # type: ignore

    with pytest.raises(RuntimeError) as excinfo:
        await provider_instance.generate(prompt=prompt)
    assert "Ollama response JSON decode error" in str(excinfo.value)


@pytest.mark.asyncio()
async def test_ollama_get_model_info_success(
    ollama_provider: OllamaLLMProviderPlugin, # ollama_provider is already an instance
    mock_httpx_client: AsyncMock # mock_httpx_client is already part of ollama_provider fixture
):
    provider_instance = await ollama_provider
    mock_tags_response = {"models": [{"name": "test-ollama-model:latest"}]}
    mock_show_response = {"details": {"family": "test_fam"}}
    dummy_request_tags = httpx.Request("POST", f"{provider_instance._base_url}/api/tags")
    dummy_request_show = httpx.Request("POST", f"{provider_instance._base_url}/api/show")

    async def post_side_effect(url: str, json: dict): # Corrected: 'json' is the kwarg
        if url.endswith("/api/tags"):
            return httpx.Response(200, json=mock_tags_response, request=dummy_request_tags)
        if url.endswith("/api/show") and json.get("name") == provider_instance._default_model:
            return httpx.Response(200, json=mock_show_response, request=dummy_request_show)
        return httpx.Response(404, request=httpx.Request("POST", url))
    provider_instance._http_client.post.side_effect = post_side_effect # type: ignore
    info = await provider_instance.get_model_info()

    assert "available_models_brief" in info
    assert "test-ollama-model:latest" in info["available_models_brief"]
    assert info["default_model_details"]["family"] == "test_fam"


@pytest.mark.asyncio()
async def test_ollama_teardown(
    ollama_provider: OllamaLLMProviderPlugin, # ollama_provider is already an instance
    mock_httpx_client: AsyncMock # mock_httpx_client is already part of ollama_provider fixture
):
    provider_instance = await ollama_provider
    client_before_teardown = provider_instance._http_client

    await provider_instance.teardown()

    assert client_before_teardown is not None
    client_before_teardown.aclose.assert_awaited_once() # type: ignore
    assert provider_instance._http_client is None

# --- New Tests for Coverage ---

@pytest.mark.asyncio()
async def test_ollama_generate_streaming_success(
    ollama_provider: OllamaLLMProviderPlugin, # ollama_provider is already an instance
    mock_httpx_client: AsyncMock # mock_httpx_client is already part of ollama_provider fixture
):
    provider = await ollama_provider
    prompt = "Stream a short story."

    async def mock_aiter_lines():
        yield json.dumps({"response": "Once upon ", "done": False})
        yield json.dumps({"response": "a time...", "done": False})
        yield json.dumps({"response": " The End.", "done": True, "prompt_eval_count": 1, "eval_count": 3})

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.aiter_lines = mock_aiter_lines # type: ignore
    mock_response.aclose = AsyncMock()
    provider._http_client.post.return_value = mock_response # type: ignore

    stream_result = await provider.generate(prompt=prompt, stream=True)
    assert isinstance(stream_result, AsyncIterable)

    chunks: List[LLMCompletionChunk] = []
    async for chunk in stream_result:
        chunks.append(chunk)

    await mock_response.aclose.wait_for_call() # Ensure stream is closed

    assert len(chunks) == 3
    assert chunks[0]["text_delta"] == "Once upon "
    assert chunks[1]["text_delta"] == "a time..."
    assert chunks[2]["text_delta"] == " The End."
    assert chunks[2]["finish_reason"] == "done"
    assert chunks[2]["usage_delta"]["total_tokens"] == 4 # type: ignore

@pytest.mark.asyncio()
async def test_ollama_chat_streaming_success(
    ollama_provider: OllamaLLMProviderPlugin, # ollama_provider is already an instance
    mock_httpx_client: AsyncMock # mock_httpx_client is already part of ollama_provider fixture
):
    provider = await ollama_provider
    messages: List[ChatMessage] = [{"role": "user", "content": "Hello stream"}]

    async def mock_aiter_lines_chat():
        yield json.dumps({"message": {"role": "assistant", "content": "Hi "}, "done": False})
        yield json.dumps({"message": {"role": "assistant", "content": "there!"}, "done": False})
        yield json.dumps({"message": {"role": "assistant", "content": ""}, "done": True, "prompt_eval_count": 2, "eval_count": 2})

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.aiter_lines = mock_aiter_lines_chat # type: ignore
    mock_response.aclose = AsyncMock()
    provider._http_client.post.return_value = mock_response # type: ignore

    stream_result = await provider.chat(messages=messages, stream=True)
    assert isinstance(stream_result, AsyncIterable)

    chunks: List[LLMChatChunk] = []
    async for chunk in stream_result:
        chunks.append(chunk)

    await mock_response.aclose.wait_for_call()

    assert len(chunks) == 3
    assert chunks[0]["message_delta"]["content"] == "Hi " # type: ignore
    assert chunks[1]["message_delta"]["content"] == "there!" # type: ignore
    assert chunks[2]["finish_reason"] == "done"
    assert chunks[2]["usage_delta"]["total_tokens"] == 4 # type: ignore

@pytest.mark.asyncio()
async def test_ollama_generate_streaming_json_decode_error(
    ollama_provider: OllamaLLMProviderPlugin, # ollama_provider is already an instance
    mock_httpx_client: AsyncMock, # mock_httpx_client is already part of ollama_provider fixture
    caplog: pytest.LogCaptureFixture
):
    provider = await ollama_provider
    # Use the correct logger name for the module under test
    test_logger = logging.getLogger("genie_tooling.llm_providers.impl.ollama_provider")
    original_level = test_logger.level
    test_logger.setLevel(logging.ERROR)
    caplog.set_level(logging.ERROR, logger="genie_tooling.llm_providers.impl.ollama_provider")


    prompt = "Stream with bad JSON."

    async def mock_aiter_lines_bad_json():
        yield json.dumps({"response": "Good chunk", "done": False})
        yield "This is not JSON" # This line will cause the error
        yield json.dumps({"response": "Another good chunk", "done": True})

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.aiter_lines = mock_aiter_lines_bad_json # type: ignore
    mock_response.aclose = AsyncMock()
    provider._http_client.post.return_value = mock_response # type: ignore

    stream_result = await provider.generate(prompt=prompt, stream=True)
    chunks_collected = 0
    async for _ in stream_result:
        chunks_collected += 1

    await mock_response.aclose.wait_for_call()

    assert chunks_collected == 2 # Only good chunks should be processed
    assert "Failed to decode JSON stream chunk: This is not JSON" in caplog.text
    test_logger.setLevel(original_level) # Restore original level

@pytest.mark.asyncio()
async def test_ollama_get_model_info_tags_api_error(
    ollama_provider: OllamaLLMProviderPlugin, # ollama_provider is already an instance
    mock_httpx_client: AsyncMock # mock_httpx_client is already part of ollama_provider fixture
):
    provider = await ollama_provider
    provider._http_client.post.side_effect = httpx.RequestError("Tags API down", request=httpx.Request("POST", f"{provider._base_url}/api/tags")) # type: ignore
    info = await provider.get_model_info()
    assert "model_info_error" in info
    assert "Tags API down" in info["model_info_error"]

@pytest.mark.asyncio()
async def test_ollama_get_model_info_show_api_error(
    ollama_provider: OllamaLLMProviderPlugin, # ollama_provider is already an instance
    mock_httpx_client: AsyncMock # mock_httpx_client is already part of ollama_provider fixture
):
    provider = await ollama_provider
    mock_tags_response = {"models": [{"name": "test-ollama-model:latest"}]}
    dummy_request_tags = httpx.Request("POST", f"{provider._base_url}/api/tags")
    dummy_request_show = httpx.Request("POST", f"{provider._base_url}/api/show")

    async def post_side_effect_show_fail(url: str, **kwargs): # Changed to accept **kwargs
        if url.endswith("/api/tags"):
            return httpx.Response(200, json=mock_tags_response, request=dummy_request_tags)
        if url.endswith("/api/show"):
            raise httpx.HTTPStatusError("Show API failed", request=dummy_request_show, response=httpx.Response(500, request=dummy_request_show))
        return httpx.Response(404, request=httpx.Request("POST", url))

    provider._http_client.post.side_effect = post_side_effect_show_fail # type: ignore
    info = await provider.get_model_info()
    assert "model_info_error" in info
    expected_error_message_from_make_request = "Ollama API error: 500 - "
    assert expected_error_message_from_make_request in info["model_info_error"]
    assert "available_models_brief" in info

@pytest.mark.asyncio()
async def test_ollama_setup_default_values():
    provider = OllamaLLMProviderPlugin()
    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    with patch("genie_tooling.llm_providers.impl.ollama_provider.httpx.AsyncClient", return_value=mock_client_instance) as MockAsyncClientConstructor:
        await provider.setup(config={})
        MockAsyncClientConstructor.assert_called_once_with(timeout=120.0)
        assert provider._base_url == "http://localhost:11434"
        assert provider._default_model == "llama2"
    await provider.teardown()

@pytest.mark.asyncio()
async def test_ollama_teardown_no_client():
    provider = OllamaLLMProviderPlugin()
    provider._http_client = None
    await provider.teardown() # Should not raise error

@pytest.mark.asyncio()
async def test_ollama_generate_with_custom_options(
    ollama_provider: OllamaLLMProviderPlugin, # ollama_provider is already an instance
    mock_httpx_client: AsyncMock # mock_httpx_client is already part of ollama_provider fixture
):
    provider = await ollama_provider
    prompt = "Test custom options"
    custom_options = {"temperature": 0.9, "num_predict": 50}
    mock_ollama_response_data = {"response": "Custom response", "done": True}
    dummy_request = httpx.Request("POST", f"{provider._base_url}/api/generate")
    provider._http_client.post.return_value = httpx.Response(200, json=mock_ollama_response_data, request=dummy_request) # type: ignore

    await provider.generate(prompt=prompt, options=custom_options)

    provider._http_client.post.assert_awaited_once() # type: ignore
    call_args, call_kwargs = provider._http_client.post.call_args # type: ignore
    payload = call_kwargs["json"]
    assert payload["options"] == custom_options

@pytest.mark.asyncio()
async def test_ollama_make_request_client_not_initialized(
    ollama_provider: OllamaLLMProviderPlugin # ollama_provider is already an instance
):
    provider = await ollama_provider
    provider._http_client = None
    with pytest.raises(RuntimeError, match="HTTP client not initialized"):
        await provider._make_request("/api/generate", {})

@pytest.mark.asyncio()
async def test_ollama_make_request_non_streaming_json_decode_error(
    ollama_provider: OllamaLLMProviderPlugin, # ollama_provider is already an instance
    mock_httpx_client: AsyncMock # mock_httpx_client is already part of ollama_provider fixture
):
    provider = await ollama_provider
    dummy_request = httpx.Request("POST", f"{provider._base_url}/api/generate")
    provider._http_client.post.return_value = httpx.Response(200, text="not json", request=dummy_request) # type: ignore
    with pytest.raises(RuntimeError, match="Ollama response JSON decode error"):
        await provider._make_request("/api/generate", {}, stream=False)

@pytest.mark.asyncio()
async def test_ollama_make_request_streaming_non_dict_chunk(
    ollama_provider: OllamaLLMProviderPlugin, # ollama_provider is already an instance
    mock_httpx_client: AsyncMock # mock_httpx_client is already part of ollama_provider fixture
):
    provider = await ollama_provider
    async def mock_aiter_lines_non_dict():
        yield json.dumps({"response": "Good chunk", "done": False})
        yield "not a dict string, but valid json string" # This line will be skipped
        yield json.dumps({"response": "Another good chunk", "done": True})

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.aiter_lines = mock_aiter_lines_non_dict # type: ignore
    mock_response.aclose = AsyncMock()
    provider._http_client.post.return_value = mock_response # type: ignore

    stream_result = await provider._make_request("/api/generate", {}, stream=True)
    results = []
    async for item in stream_result: # type: ignore
        results.append(item)

    await mock_response.aclose.wait_for_call()

    assert len(results) == 2
    assert results[0] == {"response": "Good chunk", "done": False}
    assert results[1] == {"response": "Another good chunk", "done": True}

@pytest.mark.asyncio()
async def test_ollama_generate_streaming_empty_stream(
    ollama_provider: OllamaLLMProviderPlugin,
    mock_httpx_client: AsyncMock
):
    provider = await ollama_provider
    async def empty_stream():
        if False:
            yield
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.aiter_lines = empty_stream # type: ignore
    mock_response.aclose = AsyncMock()
    provider._http_client.post.return_value = mock_response # type: ignore

    stream_result = await provider.generate(prompt="test", stream=True)
    chunks = await consume_async_iterable(stream_result)
    await mock_response.aclose.wait_for_call()
    assert len(chunks) == 0

@pytest.mark.asyncio()
async def test_ollama_chat_streaming_empty_stream(
    ollama_provider: OllamaLLMProviderPlugin,
    mock_httpx_client: AsyncMock
):
    provider = await ollama_provider
    async def empty_stream():
        if False:
            yield
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.aiter_lines = empty_stream # type: ignore
    mock_response.aclose = AsyncMock()
    provider._http_client.post.return_value = mock_response # type: ignore

    stream_result = await provider.chat(messages=[{"role":"user", "content":"test"}], stream=True)
    chunks = await consume_async_iterable(stream_result)
    await mock_response.aclose.wait_for_call()
    assert len(chunks) == 0

@pytest.mark.asyncio()
async def test_ollama_generate_non_streaming_missing_keys(
    ollama_provider: OllamaLLMProviderPlugin,
    mock_httpx_client: AsyncMock
):
    provider = await ollama_provider
    mock_response_data = {"model": "test-model"} # Missing "response" and "done"
    dummy_request = httpx.Request("POST", f"{provider._base_url}/api/generate")
    provider._http_client.post.return_value = httpx.Response(200, json=mock_response_data, request=dummy_request) # type: ignore
    result = await provider.generate(prompt="test")
    assert result["text"] == ""
    assert result["finish_reason"] == "unknown"

@pytest.mark.asyncio()
async def test_ollama_chat_non_streaming_missing_keys(
    ollama_provider: OllamaLLMProviderPlugin,
    mock_httpx_client: AsyncMock
):
    provider = await ollama_provider
    mock_response_data = {"model": "test-model"} # Missing "message" and "done"
    dummy_request = httpx.Request("POST", f"{provider._base_url}/api/chat")
    provider._http_client.post.return_value = httpx.Response(200, json=mock_response_data, request=dummy_request) # type: ignore
    result = await provider.chat(messages=[{"role":"user", "content":"test"}])
    assert result["message"]["role"] == "assistant"
    assert result["message"]["content"] == ""
    assert result["finish_reason"] == "unknown"

@pytest.mark.asyncio()
async def test_ollama_chat_message_content_not_string(
    ollama_provider: OllamaLLMProviderPlugin,
    mock_httpx_client: AsyncMock
):
    provider = await ollama_provider
    mock_response_data = {"message": {"role": "assistant", "content": 123}, "done": True} # Content is int
    dummy_request = httpx.Request("POST", f"{provider._base_url}/api/chat")
    provider._http_client.post.return_value = httpx.Response(200, json=mock_response_data, request=dummy_request) # type: ignore
    result = await provider.chat(messages=[{"role":"user", "content":"test"}])
    assert result["message"]["content"] == 123 # Should pass through as is

@pytest.mark.asyncio()
async def test_ollama_get_model_info_tags_malformed(
    ollama_provider: OllamaLLMProviderPlugin,
    mock_httpx_client: AsyncMock
):
    provider = await ollama_provider
    mock_tags_response = {"not_models_key": []} # Malformed
    dummy_request = httpx.Request("POST", f"{provider._base_url}/api/tags")
    provider._http_client.post.return_value = httpx.Response(200, json=mock_tags_response, request=dummy_request) # type: ignore
    info = await provider.get_model_info()
    assert "available_models_brief" not in info # Should handle gracefully

@pytest.mark.asyncio()
async def test_ollama_get_model_info_show_malformed(
    ollama_provider: OllamaLLMProviderPlugin,
    mock_httpx_client: AsyncMock
):
    provider = await ollama_provider
    mock_tags_response = {"models": [{"name": "test-ollama-model:latest"}]}
    mock_show_response_malformed = {"no_details_key": {}}
    dummy_request_tags = httpx.Request("POST", f"{provider._base_url}/api/tags")
    dummy_request_show = httpx.Request("POST", f"{provider._base_url}/api/show")

    async def post_side_effect_show_malformed(url: str, **kwargs): # Corrected signature
        if url.endswith("/api/tags"):
            return httpx.Response(200, json=mock_tags_response, request=dummy_request_tags)
        if url.endswith("/api/show"):
            return httpx.Response(200, json=mock_show_response_malformed, request=dummy_request_show)
        return httpx.Response(404, request=httpx.Request("POST", url))
    provider._http_client.post.side_effect = post_side_effect_show_malformed # type: ignore
    info = await provider.get_model_info()
    assert "default_model_details" in info
    assert info["default_model_details"] == { # Should be empty or default values
        "parameters": None, "template": None, "family": None
    }
