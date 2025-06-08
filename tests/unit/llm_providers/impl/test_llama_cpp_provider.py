import json
import logging
from typing import Any, AsyncIterable, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx as real_httpx  # Import httpx with an alias
import pytest
from genie_tooling.llm_providers.impl.llama_cpp_provider import (
    LlamaCppLLMProviderPlugin,
)
from genie_tooling.llm_providers.types import (
    ChatMessage,
)
from genie_tooling.security.key_provider import KeyProvider
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field as PydanticField

# Logger for the module under test
PROVIDER_LOGGER_NAME = "genie_tooling.llm_providers.impl.llama_cpp_provider"


# --- Mocks & Fixtures ---
@pytest.fixture
def mock_httpx_client_instance() -> AsyncMock:
    # This mock represents an *instance* of httpx.AsyncClient.
    # It doesn't need a spec if we're just mocking its methods like post/aclose.
    client = AsyncMock()
    client.post = AsyncMock()
    client.get = AsyncMock()  # Add mock for GET for /v1/models
    client.aclose = AsyncMock()
    return client


@pytest.fixture
def mock_key_provider_llama() -> AsyncMock:
    kp = AsyncMock(spec=KeyProvider)
    kp.get_key = AsyncMock(return_value="test_llama_cpp_api_key")
    return kp


@pytest.fixture
async def llama_cpp_provider(
    mock_httpx_client_instance: AsyncMock, mock_key_provider_llama: AsyncMock
) -> LlamaCppLLMProviderPlugin:
    provider = LlamaCppLLMProviderPlugin()
    # Patch httpx.AsyncClient specifically within the module where it's used by the provider
    with patch(
        "genie_tooling.llm_providers.impl.llama_cpp_provider.httpx.AsyncClient",
        return_value=mock_httpx_client_instance,  # The constructor will return our mock instance
    ):
        await provider.setup(
            config={
                "base_url": "http://mock-llama:8080",
                "model_name": "test-model-alias",
                "key_provider": mock_key_provider_llama,
                "api_key_name": "LLAMA_CPP_API_KEY_TEST",
            }
        )
    return provider


class SimpleOutputSchema(PydanticBaseModel):
    result: str = PydanticField(description="The result of the operation.")
    count: int


# --- Helper for Streaming ---
async def consume_async_iterable(
    iterable: AsyncIterable[Any],
) -> List[Any]:
    return [item async for item in iterable]


# --- Test Cases ---
@pytest.mark.asyncio
class TestLlamaCppProviderSetup:
    async def test_setup_defaults(self):
        provider = LlamaCppLLMProviderPlugin()
        # Patch the constructor in the module where it's used by the provider
        with patch(
            "genie_tooling.llm_providers.impl.llama_cpp_provider.httpx.AsyncClient"
        ) as MockedAsyncClientConstructor:
            # The constructor mock will return a new AsyncMock instance by default
            # if we don't set return_value. Or we can set it explicitly.
            mock_returned_instance = AsyncMock()  # No spec needed for the instance here
            MockedAsyncClientConstructor.return_value = mock_returned_instance

            await provider.setup(config={})

        MockedAsyncClientConstructor.assert_called_once_with(
            timeout=120.0, headers={}
        )
        assert provider._http_client is mock_returned_instance
        assert provider._base_url == "http://localhost:8080"
        assert provider._default_model_alias is None
        assert provider._request_timeout == 120.0
        assert provider._api_key_name is None
        assert provider._key_provider is None

    async def test_setup_custom_params(
        self, mock_key_provider_llama: AsyncMock
    ):
        provider = LlamaCppLLMProviderPlugin()
        custom_config = {
            "base_url": "http://custom:1234/",
            "model_name": "custom-alias",
            "request_timeout_seconds": 60.0,
            "api_key_name": "MY_LLAMA_KEY",
            "key_provider": mock_key_provider_llama,
        }
        with patch(
            "genie_tooling.llm_providers.impl.llama_cpp_provider.httpx.AsyncClient"
        ) as MockedAsyncClientConstructor:
            mock_returned_instance = AsyncMock()  # No spec needed for the instance here
            MockedAsyncClientConstructor.return_value = mock_returned_instance

            await provider.setup(config=custom_config)

        MockedAsyncClientConstructor.assert_called_once_with(
            timeout=60.0, headers={"Authorization": "Bearer test_llama_cpp_api_key"}
        )
        assert provider._http_client is mock_returned_instance
        assert provider._base_url == "http://custom:1234"
        assert provider._default_model_alias == "custom-alias"
        assert provider._request_timeout == 60.0
        assert provider._api_key_name == "MY_LLAMA_KEY"
        assert provider._key_provider is mock_key_provider_llama
        mock_key_provider_llama.get_key.assert_awaited_with("MY_LLAMA_KEY")

    async def test_setup_api_key_not_found(
        self, mock_key_provider_llama: AsyncMock, caplog
    ):
        caplog.set_level(logging.WARNING, logger=PROVIDER_LOGGER_NAME)
        provider = LlamaCppLLMProviderPlugin()
        mock_key_provider_llama.get_key.return_value = None
        with patch(
            "genie_tooling.llm_providers.impl.llama_cpp_provider.httpx.AsyncClient"
        ) as MockedAsyncClientConstructor:
            mock_returned_instance = AsyncMock()  # No spec needed for the instance here
            MockedAsyncClientConstructor.return_value = mock_returned_instance

            await provider.setup(
                config={
                    "key_provider": mock_key_provider_llama,
                    "api_key_name": "MISSING_KEY",
                }
            )
        assert "API key 'MISSING_KEY' configured but not found" in caplog.text
        MockedAsyncClientConstructor.assert_called_once()
        call_args, call_kwargs = MockedAsyncClientConstructor.call_args
        assert call_kwargs.get("headers") == {}


@pytest.mark.asyncio
class TestLlamaCppProviderGenerate:
    async def test_generate_success(self, llama_cpp_provider: LlamaCppLLMProviderPlugin):
        provider = await llama_cpp_provider
        prompt = "Once upon a time"
        expected_text = "there was a llama."
        mock_response_data = {
            "content": expected_text,
            "stop": True,
            "stopped_eos": True,
            "tokens_evaluated": 10,
            "tokens_predicted": 5,
        }
        provider._http_client.post.return_value = real_httpx.Response(  # type: ignore
            200,
            json=mock_response_data,
            request=real_httpx.Request("POST", provider._base_url),
        )

        result = await provider.generate(prompt=prompt, temperature=0.7, max_tokens=50)

        assert isinstance(result, dict)
        assert result["text"] == expected_text
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10  # type: ignore
        assert result["usage"]["completion_tokens"] == 5  # type: ignore
        assert result["usage"]["total_tokens"] == 15  # type: ignore
        provider._http_client.post.assert_awaited_once()  # type: ignore
        call_kwargs = provider._http_client.post.call_args.kwargs  # type: ignore
        payload = call_kwargs["json"]
        assert payload["prompt"] == prompt
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 50
        assert payload["stream"] is False

    @patch(
        "genie_tooling.llm_providers.impl.llama_cpp_provider.generate_gbnf_grammar_from_pydantic_models"
    )
    async def test_generate_with_gbnf_pydantic(
        self,
        mock_generate_gbnf: MagicMock,
        llama_cpp_provider: LlamaCppLLMProviderPlugin,
    ):
        provider = await llama_cpp_provider
        mock_generate_gbnf.return_value = 'root ::= "test"'
        provider._http_client.post.return_value = real_httpx.Response(  # type: ignore
            200,
            json={"content": "test", "stop": True},
            request=real_httpx.Request("POST", provider._base_url),
        )

        await provider.generate(prompt="Test GBNF", output_schema=SimpleOutputSchema)
        mock_generate_gbnf.assert_called_once_with([SimpleOutputSchema])
        call_kwargs = provider._http_client.post.call_args.kwargs  # type: ignore
        assert call_kwargs["json"]["grammar"] == 'root ::= "test"'

    @patch(
        "genie_tooling.llm_providers.impl.llama_cpp_provider.create_dynamic_models_from_dictionaries"
    )
    @patch(
        "genie_tooling.llm_providers.impl.llama_cpp_provider.generate_gbnf_grammar_from_pydantic_models"
    )
    async def test_generate_with_gbnf_dict_schema(
        self,
        mock_generate_gbnf: MagicMock,
        mock_create_dynamic: MagicMock,
        llama_cpp_provider: LlamaCppLLMProviderPlugin,
    ):
        provider = await llama_cpp_provider
        dict_schema = {"type": "object", "properties": {"key": {"type": "string"}}}
        mock_dynamic_model = MagicMock()
        mock_create_dynamic.return_value = [mock_dynamic_model]
        mock_generate_gbnf.return_value = 'root ::= "key_value"'
        provider._http_client.post.return_value = real_httpx.Response(  # type: ignore
            200,
            json={"content": "key_value", "stop": True},
            request=real_httpx.Request("POST", provider._base_url),
        )

        await provider.generate(prompt="Test GBNF dict", output_schema=dict_schema)
        mock_create_dynamic.assert_called_once_with([dict_schema])
        mock_generate_gbnf.assert_called_once_with([mock_dynamic_model])
        call_kwargs = provider._http_client.post.call_args.kwargs  # type: ignore
        assert call_kwargs["json"]["grammar"] == 'root ::= "key_value"'

    async def test_generate_streaming_success(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        provider = await llama_cpp_provider

        async def mock_aiter_lines():
            yield "data: " + json.dumps({"content": "Hello", "stop": False})
            yield "data: " + json.dumps({"content": " llama.cpp", "stop": False})
            yield "data: " + json.dumps(
                {
                    "content": "!",
                    "stop": True,
                    "stopped_eos": True,
                    "tokens_evaluated": 5,
                    "tokens_predicted": 3,
                }
            )
            yield "data: [DONE]"

        mock_response = AsyncMock(spec=real_httpx.Response)
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines  # type: ignore
        mock_response.aclose = AsyncMock()
        provider._http_client.post.return_value = mock_response  # type: ignore

        result_stream = await provider.generate(prompt="Stream test", stream=True)
        chunks = await consume_async_iterable(result_stream)

        assert len(chunks) == 3
        assert chunks[0]["text_delta"] == "Hello"
        assert chunks[1]["text_delta"] == " llama.cpp"
        assert chunks[2]["text_delta"] == "!"
        assert chunks[2]["finish_reason"] == "stop"
        assert chunks[2]["usage_delta"]["prompt_tokens"] == 5  # type: ignore
        assert chunks[2]["usage_delta"]["completion_tokens"] == 3  # type: ignore
        assert chunks[2]["usage_delta"]["total_tokens"] == 8  # type: ignore

    async def test_generate_http_status_error(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        provider = await llama_cpp_provider
        provider._http_client.post.side_effect = real_httpx.HTTPStatusError(  # type: ignore
            "Server Error",
            request=real_httpx.Request("POST", provider._base_url),
            response=real_httpx.Response(
                503,
                text="Service Unavailable",
                request=real_httpx.Request("POST", provider._base_url),
            ),
        )
        with pytest.raises(
            RuntimeError, match="llama.cpp API error: 503 - Service Unavailable"
        ):
            await provider.generate(prompt="test")

    async def test_generate_request_error(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        provider = await llama_cpp_provider
        provider._http_client.post.side_effect = real_httpx.RequestError(  # type: ignore
            "Network issue", request=real_httpx.Request("POST", provider._base_url)
        )
        with pytest.raises(RuntimeError, match="llama.cpp request failed: Network issue"):
            await provider.generate(prompt="test")

    async def test_generate_json_decode_error_non_streaming(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        provider = await llama_cpp_provider
        provider._http_client.post.return_value = real_httpx.Response(  # type: ignore
            200,
            text="not valid json",
            request=real_httpx.Request("POST", provider._base_url),
        )
        with pytest.raises(RuntimeError, match="llama.cpp response JSON decode error"):
            await provider.generate(prompt="test", stream=False)

    async def test_generate_gbnf_server_returns_non_stream_error(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        provider = await llama_cpp_provider
        # Simulate _make_request returning a dict even when stream=True was passed to it
        # This happens if the server doesn't stream despite the "stream": true in payload
        # when GBNF is used.
        mock_non_stream_response_dict = {
            "content": "GBNF result",
            "stop": True,
            "tokens_evaluated": 5,
            "tokens_predicted": 2,
        }
        provider._make_request = AsyncMock(return_value=mock_non_stream_response_dict)  # type: ignore

        with pytest.raises(
            RuntimeError,
            match="Expected stream from _make_request for generate when server_stream_request was True",
        ):
            await provider.generate(
                prompt="Test GBNF non-stream error",
                output_schema=SimpleOutputSchema,
                stream=False,
            )


@pytest.mark.asyncio
class TestLlamaCppProviderChat:
    async def test_chat_success(self, llama_cpp_provider: LlamaCppLLMProviderPlugin):
        provider = await llama_cpp_provider
        messages: List[ChatMessage] = [{"role": "user", "content": "Hi"}]
        expected_content = "Hello from llama.cpp chat!"
        mock_response_data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": expected_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 6, "total_tokens": 13},
        }
        provider._http_client.post.return_value = real_httpx.Response(  # type: ignore
            200,
            json=mock_response_data,
            request=real_httpx.Request("POST", provider._base_url),
        )

        result = await provider.chat(messages=messages)

        assert isinstance(result, dict)
        assert result["message"]["role"] == "assistant"
        assert result["message"]["content"] == expected_content
        assert result["finish_reason"] == "stop"
        assert result["usage"]["total_tokens"] == 13  # type: ignore
        provider._http_client.post.assert_awaited_once()  # type: ignore
        call_kwargs = provider._http_client.post.call_args.kwargs  # type: ignore
        payload = call_kwargs["json"]
        assert payload["messages"] == messages
        assert payload["stream"] is False

    async def test_chat_streaming_success(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        provider = await llama_cpp_provider

        async def mock_aiter_lines_chat():
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"role": "assistant", "content": "Chatting "}}]}
            )
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": "stream..."}}]}
            )
            yield "data: " + json.dumps(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            )
            yield "data: [DONE]"

        mock_response = AsyncMock(spec=real_httpx.Response)
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines_chat  # type: ignore
        mock_response.aclose = AsyncMock()
        provider._http_client.post.return_value = mock_response  # type: ignore

        result_stream = await provider.chat(
            messages=[{"role": "user", "content": "Stream chat"}], stream=True
        )
        chunks = await consume_async_iterable(result_stream)

        assert len(chunks) == 3
        assert chunks[0]["message_delta"]["role"] == "assistant"  # type: ignore
        assert chunks[0]["message_delta"]["content"] == "Chatting "  # type: ignore
        assert chunks[1]["message_delta"]["content"] == "stream..."  # type: ignore
        assert chunks[2]["finish_reason"] == "stop"
        assert chunks[2]["usage_delta"]["total_tokens"] == 15  # type: ignore

    async def test_chat_http_status_error(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        provider = await llama_cpp_provider
        provider._http_client.post.side_effect = real_httpx.HTTPStatusError(  # type: ignore
            "Bad Request",
            request=real_httpx.Request("POST", provider._base_url),
            response=real_httpx.Response(
                400,
                text="Invalid input",
                request=real_httpx.Request("POST", provider._base_url),
            ),
        )
        with pytest.raises(
            RuntimeError, match="llama.cpp API error: 400 - Invalid input"
        ):
            await provider.chat(messages=[{"role": "user", "content": "test"}])

    async def test_chat_request_error(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        provider = await llama_cpp_provider
        provider._http_client.post.side_effect = real_httpx.RequestError(  # type: ignore
            "Timeout", request=real_httpx.Request("POST", provider._base_url)
        )
        with pytest.raises(RuntimeError, match="llama.cpp request failed: Timeout"):
            await provider.chat(messages=[{"role": "user", "content": "test"}])

    async def test_chat_json_decode_error_non_streaming(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        provider = await llama_cpp_provider
        provider._http_client.post.return_value = real_httpx.Response(  # type: ignore
            200,
            text="bad json data",
            request=real_httpx.Request("POST", provider._base_url),
        )
        with pytest.raises(RuntimeError, match="llama.cpp response JSON decode error"):
            await provider.chat(messages=[{"role": "user", "content": "test"}], stream=False)

    async def test_chat_gbnf_server_returns_non_stream_error(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        provider = await llama_cpp_provider
        mock_non_stream_response_dict = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "GBNF chat result"},
                    "finish_reason": "stop",
                }
            ]
        }
        provider._make_request = AsyncMock(return_value=mock_non_stream_response_dict)  # type: ignore

        with pytest.raises(
            RuntimeError,
            match="Expected stream from _make_request for chat when server_stream_request_chat was True",
        ):
            await provider.chat(
                messages=[{"role": "user", "content": "Test GBNF chat non-stream"}],
                output_schema=SimpleOutputSchema,
                stream=False,
            )


@pytest.mark.asyncio
class TestLlamaCppProviderErrorsAndInfo:
    async def test_get_model_info_success(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        """Test successful retrieval of model info from /v1/models."""
        provider = await llama_cpp_provider
        mock_models_response = {
            "data": [{"id": "model-a"}, {"id": "model-b"}]
        }
        provider._http_client.get.return_value = real_httpx.Response( # type: ignore
            200,
            json=mock_models_response,
            request=real_httpx.Request("GET", f"{provider._base_url}/v1/models"),
        )

        info = await provider.get_model_info()

        assert info["provider"] == "llama.cpp"
        assert "available_models_on_server" in info
        assert info["available_models_on_server"] == ["model-a", "model-b"]
        assert "model_info_error" not in info
        provider._http_client.get.assert_awaited_once_with(f"{provider._base_url}/v1/models")

    async def test_get_model_info_api_error(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        """Test handling of API error when fetching model info."""
        provider = await llama_cpp_provider
        provider._http_client.get.side_effect = real_httpx.RequestError(  # type: ignore
            "Models endpoint down",
            request=real_httpx.Request("GET", f"{provider._base_url}/v1/models"),
        )
        info = await provider.get_model_info()
        assert "model_info_error" in info
        assert "Models endpoint down" in info["model_info_error"]

    async def test_teardown(self, llama_cpp_provider: LlamaCppLLMProviderPlugin):
        provider = await llama_cpp_provider
        client_mock = provider._http_client
        await provider.teardown()
        client_mock.aclose.assert_awaited_once()  # type: ignore
        assert provider._http_client is None
        assert provider._key_provider is None

    async def test_client_not_initialized(self):
        provider = LlamaCppLLMProviderPlugin()  # No setup call
        with pytest.raises(RuntimeError, match="HTTP client not initialized"):
            await provider.generate(prompt="test")
        with pytest.raises(RuntimeError, match="HTTP client not initialized"):
            await provider.chat(messages=[])

    async def test_generate_streaming_non_json_line(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin, caplog
    ):
        provider = await llama_cpp_provider
        # Set the specific logger for this test
        test_logger = logging.getLogger(PROVIDER_LOGGER_NAME)
        original_level = test_logger.level
        test_logger.setLevel(
            logging.ERROR
        )  # Ensure ERROR logs are captured by caplog for this logger
        caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)  # Also for caplog itself

        async def mock_aiter_lines_bad_json():
            yield "data: " + json.dumps({"content": "Good chunk", "stop": False})
            yield "data: This is not JSON"  # Bad line - Corrected
            yield "data: " + json.dumps({"content": "!", "stop": True})
            yield "data: [DONE]"

        mock_response = AsyncMock(spec=real_httpx.Response)
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines_bad_json  # type: ignore
        mock_response.aclose = AsyncMock()  # Ensure aclose is an AsyncMock
        provider._http_client.post.return_value = mock_response  # type: ignore

        stream = await provider.generate(prompt="test stream bad json", stream=True)
        results = await consume_async_iterable(stream)
        await mock_response.aclose.wait_for_call()  # Ensure stream is closed

        assert len(results) == 2  # Only good chunks
        assert "Failed to decode JSON stream chunk: This is not JSON" in caplog.text
        test_logger.setLevel(original_level)  # Restore original level

    async def test_chat_streaming_non_json_line(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin, caplog
    ):
        provider = await llama_cpp_provider
        test_logger = logging.getLogger(PROVIDER_LOGGER_NAME)
        original_level = test_logger.level
        test_logger.setLevel(logging.ERROR)
        caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)

        async def mock_aiter_lines_bad_json_chat():
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": "Good "}}]}
            )
            yield "data: Malformed line"  # Bad line - Corrected
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": "chunk."}}]}
            )
            yield "data: [DONE]"

        mock_response = AsyncMock(spec=real_httpx.Response)
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines_bad_json_chat  # type: ignore
        mock_response.aclose = AsyncMock()  # Ensure aclose is an AsyncMock
        provider._http_client.post.return_value = mock_response  # type: ignore

        stream = await provider.chat(
            messages=[{"role": "user", "content": "test"}], stream=True
        )
        results = await consume_async_iterable(stream)
        await mock_response.aclose.wait_for_call()

        assert len(results) == 2
        assert "Failed to decode JSON stream chunk: Malformed line" in caplog.text
        test_logger.setLevel(original_level)

    async def test_generate_streaming_non_dict_chunk(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        provider = await llama_cpp_provider

        async def mock_aiter_lines_non_dict_chunk():
            yield "data: " + json.dumps("a string, not a dict")
            yield "data: " + json.dumps({"content": "final chunk", "stop": True})
            yield "data: [DONE]"

        mock_response = AsyncMock(spec=real_httpx.Response)
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines_non_dict_chunk  # type: ignore
        mock_response.aclose = AsyncMock()  # Ensure aclose is an AsyncMock
        provider._http_client.post.return_value = mock_response  # type: ignore

        stream = await provider.generate(prompt="test stream non-dict", stream=True)
        results = await consume_async_iterable(stream)
        await mock_response.aclose.wait_for_call()
        assert len(results) == 1  # Skips the non-dict chunk
        assert results[0]["text_delta"] == "final chunk"

    async def test_chat_streaming_non_dict_chunk(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        provider = await llama_cpp_provider

        async def mock_aiter_lines_non_dict_chat_chunk():
            yield "data: " + json.dumps(["not a dict"])
            yield "data: " + json.dumps(
                {"choices": [{"delta": {"content": "final "}}]}
            )
            yield "data: [DONE]"

        mock_response = AsyncMock(spec=real_httpx.Response)
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines_non_dict_chat_chunk  # type: ignore
        mock_response.aclose = AsyncMock()  # Ensure aclose is an AsyncMock
        provider._http_client.post.return_value = mock_response  # type: ignore

        stream = await provider.chat(
            messages=[{"role": "user", "content": "test"}], stream=True
        )
        results = await consume_async_iterable(stream)
        await mock_response.aclose.wait_for_call()
        assert len(results) == 1
        assert results[0]["message_delta"]["content"] == "final "  # type: ignore
