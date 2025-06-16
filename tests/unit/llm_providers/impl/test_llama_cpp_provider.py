#
# FILE: tests/unit/llm_providers/impl/test_llama_cpp_provider.py
#
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
@pytest.fixture()
def mock_httpx_client_instance() -> AsyncMock:
    # This mock represents an *instance* of httpx.AsyncClient.
    # It doesn't need a spec if we're just mocking its methods like post/aclose.
    client = AsyncMock()
    client.post = AsyncMock()
    client.get = AsyncMock()  # Add mock for GET for /v1/models
    client.aclose = AsyncMock()
    return client


@pytest.fixture()
def mock_key_provider_llama() -> AsyncMock:
    kp = AsyncMock(spec=KeyProvider)
    kp.get_key = AsyncMock(return_value="test_llama_cpp_api_key")
    return kp


@pytest.fixture()
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
@pytest.mark.asyncio()
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
        assert isinstance(provider._http_client, AsyncMock)
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


@pytest.mark.asyncio()
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

        mock_response.headers = {"content-type": "text/event-stream"}
        type(mock_response).is_closed = MagicMock(return_value=False)
        async def set_closed_on_aclose():
            type(mock_response).is_closed = MagicMock(return_value=True)
        mock_response.aclose.side_effect = set_closed_on_aclose

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

    async def test_http_status_error(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        provider = await llama_cpp_provider
        mock_response_obj = real_httpx.Response(
            503,
            text="Service Unavailable",
            request=real_httpx.Request("POST", provider._base_url),
        )
        provider._http_client.post.side_effect = real_httpx.HTTPStatusError(  # type: ignore
            "Server Error",
            request=real_httpx.Request("POST", provider._base_url),
            response=mock_response_obj,
        )
        type(mock_response_obj).aclose = AsyncMock()

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

    async def test_generate_gbnf_server_returns_non_stream(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        provider = await llama_cpp_provider
        mock_non_stream_response_dict = {
            "content": "GBNF result",
            "stop": True,
            "stopped_eos": True,
            "tokens_evaluated": 5,
            "tokens_predicted": 2,
        }
        dummy_request = real_httpx.Request("POST", provider._base_url)
        mock_response = real_httpx.Response(
            200,
            json=mock_non_stream_response_dict,
            request=dummy_request,
            headers={"content-type": "application/json"},
        )
        mock_response.aread = AsyncMock(
            return_value=json.dumps(mock_non_stream_response_dict).encode("utf-8")
        )
        provider._http_client.post.return_value = mock_response  # type: ignore

        result = await provider.generate(
            prompt="Test GBNF non-stream",
            output_schema=SimpleOutputSchema,
            stream=True,  # User requests stream, but server responds with single JSON
        )

        chunks = await consume_async_iterable(result)
        assert len(chunks) == 1
        assert chunks[0]["text_delta"] == "GBNF result"
        assert chunks[0]["finish_reason"] == "stop"
        assert chunks[0]["usage_delta"]["total_tokens"] == 7  # type: ignore


@pytest.mark.asyncio()
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
        mock_response.headers = {"content-type": "text/event-stream"}
        mock_response.aiter_lines = mock_aiter_lines_chat  # type: ignore
        mock_response.aclose = AsyncMock()
        type(mock_response).is_closed = MagicMock(return_value=False)
        async def set_closed_on_aclose_chat():
            type(mock_response).is_closed = MagicMock(return_value=True)
        mock_response.aclose.side_effect = set_closed_on_aclose_chat

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

    async def test_chat_gbnf_server_returns_non_stream(
        self, llama_cpp_provider: LlamaCppLLMProviderPlugin
    ):
        provider = await llama_cpp_provider
        mock_non_stream_response_dict = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "GBNF chat result"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        dummy_request = real_httpx.Request("POST", provider._base_url)
        mock_response = real_httpx.Response(
            200,
            json=mock_non_stream_response_dict,
            request=dummy_request,
            headers={"content-type": "application/json"},
        )
        mock_response.aread = AsyncMock(
            return_value=json.dumps(mock_non_stream_response_dict).encode("utf-8")
        )
        provider._http_client.post.return_value = mock_response  # type: ignore
        result_stream = await provider.chat(
            messages=[{"role": "user", "content": "Test GBNF chat"}],
            output_schema=SimpleOutputSchema,
            stream=True,
        )
        chunks = await consume_async_iterable(result_stream)
        assert len(chunks) == 1
        assert chunks[0]["message_delta"]["content"] == "GBNF chat result"

@pytest.mark.asyncio()
async def test_generate_handles_gbnf_non_stream_response(
    llama_cpp_provider: LlamaCppLLMProviderPlugin,
):
    provider = await llama_cpp_provider
    mock_response_dict = {
        "content": '{"result": "parsed", "count": 1}',
        "stop": True,
        "stopped_eos": True,  # Ensure a stop reason
        "tokens_evaluated": 10,
        "tokens_predicted": 20,
        "generation_settings": {},
    }
    dummy_request = real_httpx.Request("POST", provider._base_url)
    mock_response = real_httpx.Response(
        200,
        json=mock_response_dict,
        request=dummy_request,
        headers={"content-type": "application/json"},
    )
    mock_response.aread = AsyncMock(
        return_value=json.dumps(mock_response_dict).encode("utf-8")
    )  # type: ignore
    provider._http_client.post.return_value = mock_response  # type: ignore

    result = await provider.generate(
        prompt="generate json", output_schema=SimpleOutputSchema, stream=False
    )

    provider._http_client.post.assert_awaited_once()
    assert result["text"] == '{"result": "parsed", "count": 1}'
    assert result["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 30


@pytest.mark.asyncio()
async def test_chat_handles_gbnf_non_stream_response(
    llama_cpp_provider: LlamaCppLLMProviderPlugin,
):
    provider = await llama_cpp_provider
    mock_response_dict = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"result": "chat parsed", "count": 2}',
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
    }
    dummy_request = real_httpx.Request("POST", provider._base_url)
    mock_response = real_httpx.Response(
        200,
        json=mock_response_dict,
        request=dummy_request,
        headers={"content-type": "application/json"},
    )
    mock_response.aread = AsyncMock(
        return_value=json.dumps(mock_response_dict).encode("utf-8")
    )
    provider._http_client.post.return_value = mock_response  # type: ignore

    result = await provider.chat(
        messages=[{"role": "user", "content": "test"}],
        output_schema=SimpleOutputSchema,
        stream=False,
    )

    provider._http_client.post.assert_awaited_once()
    assert result["message"]["content"] == '{"result": "chat parsed", "count": 2}'
    assert result["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 40