### tests/unit/llm_providers/impl/test_openai_provider.py
import json
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.llm_providers.impl.openai_provider import OpenAILLMProviderPlugin
from genie_tooling.llm_providers.types import ChatMessage, ToolCall
from genie_tooling.security.key_provider import KeyProvider

# Mock OpenAI SDK types
try:
    import httpx  # Actual OpenAI errors often wrap httpx.Response
    from openai import APIError as ActualOpenAI_APIError
    from openai import APIStatusError as ActualAPIStatusError  # Added
    from openai import AsyncOpenAI as ActualAsyncOpenAI
    from openai import AuthenticationError as ActualAuthenticationError  # Added
    from openai import RateLimitError as ActualOpenAI_RateLimitError
    from openai.types.chat import ChatCompletionMessage as ActualOpenAIChatMessage
    from openai.types.chat import ChatCompletionMessageToolCall as ActualOpenAIToolCall
    from openai.types.chat.chat_completion import Choice as ActualOpenAIChoice
    from openai.types.completion_usage import CompletionUsage as ActualOpenAIUsage

    APIError_ToUse = ActualOpenAI_APIError
    APIStatusError_ToUse = ActualAPIStatusError # Added
    AuthenticationError_ToUse = ActualAuthenticationError # Added
    RateLimitError_ToUse = ActualOpenAI_RateLimitError
    OpenAIChatMessage_ToUse = ActualOpenAIChatMessage
    OpenAIToolCall_ToUse = ActualOpenAIToolCall
    OpenAIChoice_ToUse = ActualOpenAIChoice
    OpenAIUsage_ToUse = ActualOpenAIUsage
    AsyncOpenAI_ToUse = ActualAsyncOpenAI

except ImportError:
    httpx = None # type: ignore
    class MockMinimalHTTPXRequest: # Minimal mock for request object
        def __init__(self, method: str, url: str):
            self.method = method
            self.url = url

    class MockMinimalHTTPXResponse: # Minimal mock for error responses
        def __init__(self, status_code: int, json_data: Optional[Dict] = None, text_data: Optional[str] = None, headers: Optional[Dict] = None):
            self.status_code = status_code
            self._json_data = json_data
            self._text_data = text_data if text_data is not None else (json.dumps(json_data) if json_data else "")
            self.headers = headers or {}
            self.request: Any = None # Ensure request attribute exists
        def json(self) -> Any:
            if self._json_data is not None:
                return self._json_data
            raise json.JSONDecodeError("No JSON data", self._text_data, 0)
        @property
        def text(self) -> str:
            return self._text_data

    class MockOpenAIError(Exception):
        def __init__(self, message: Optional[str], request: Any, *, body: Optional[object]):
            super().__init__(message)
            self.message = message or ""
            self.request = request
            self.body = body
            self.response: Any = None
            self.status_code: Optional[int] = None
            self.headers: Optional[Any] = None

    class MockAPIStatusError(MockOpenAIError): # For mocking status-specific errors
        def __init__(self, message: str, *, response: Any, body: object | None) -> None:
            # APIStatusError calls super().__init__(message, request=response.request, body=body)
            req = getattr(response, "request", None)
            if req is None: # Create a dummy request if not present on the mock response
                req = MockMinimalHTTPXRequest("POST", "http://mock.url")
            super().__init__(message, request=req, body=body)
            self.response = response
            self.status_code = getattr(response, "status_code", None)
            self.headers = getattr(response, "headers", None)


    APIError_ToUse = MockOpenAIError # type: ignore
    APIStatusError_ToUse = MockAPIStatusError # type: ignore
    AuthenticationError_ToUse = MockAPIStatusError # type: ignore (AuthenticationError is an APIStatusError)
    RateLimitError_ToUse = MockAPIStatusError # type: ignore (RateLimitError is an APIStatusError)
    OpenAIChatMessage_ToUse = MagicMock # type: ignore
    OpenAIToolCall_ToUse = MagicMock # type: ignore
    OpenAIChoice_ToUse = MagicMock # type: ignore
    OpenAIUsage_ToUse = MagicMock # type: ignore
    AsyncOpenAI_ToUse = AsyncMock # type: ignore


@pytest.fixture
def mock_openai_client_instance() -> AsyncMock:
    client_instance = AsyncMock(spec=AsyncOpenAI_ToUse)
    client_instance.chat = MagicMock()
    client_instance.chat.completions = MagicMock()
    client_instance.chat.completions.create = AsyncMock(name="MockChatCompletionsCreate")
    client_instance.close = AsyncMock(name="MockAsyncOpenAIClose")
    return client_instance

@pytest.fixture
async def openai_provider(
    mock_openai_client_instance: AsyncMock,
    mock_key_provider: KeyProvider
) -> OpenAILLMProviderPlugin:
    provider = OpenAILLMProviderPlugin()
    actual_mock_key_provider = await mock_key_provider

    with patch("genie_tooling.llm_providers.impl.openai_provider.AsyncOpenAI", return_value=mock_openai_client_instance):
        await provider.setup(
            config={
                "key_provider": actual_mock_key_provider,
                "model_name": "gpt-test-model",
                "api_key_name": OpenAILLMProviderPlugin._api_key_name
            }
        )
    assert provider._client is mock_openai_client_instance, \
        f"Provider client not set. API key for '{OpenAILLMProviderPlugin._api_key_name}' likely not found by mock_key_provider."
    return provider

@pytest.mark.asyncio
async def test_openai_setup_success(mock_key_provider: KeyProvider):
    provider = OpenAILLMProviderPlugin()
    actual_kp = await mock_key_provider
    mock_client_constructor = AsyncMock(return_value=AsyncMock(spec=AsyncOpenAI_ToUse))

    with patch("genie_tooling.llm_providers.impl.openai_provider.AsyncOpenAI", mock_client_constructor):
        await provider.setup(config={
            "key_provider": actual_kp,
            "model_name": "gpt-custom",
            "api_key_name": "OPENAI_API_KEY",
            "openai_api_base": "http://localhost:1234",
            "openai_organization": "org-test"
        })
    mock_client_constructor.assert_called_once_with(
        api_key="test_openai_key_from_conftest_fixture",
        base_url="http://localhost:1234",
        organization="org-test"
    )
    assert provider._client is not None
    assert provider._model_name == "gpt-custom"

@pytest.mark.asyncio
async def test_openai_setup_no_api_key(mock_key_provider: KeyProvider, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR)
    provider = OpenAILLMProviderPlugin()
    actual_kp = await mock_key_provider
    actual_kp.get_key = AsyncMock(return_value=None) # type: ignore

    with patch("genie_tooling.llm_providers.impl.openai_provider.AsyncOpenAI"):
        await provider.setup(config={"key_provider": actual_kp, "api_key_name": "MISSING_KEY"})
    assert provider._client is None
    assert "API key 'MISSING_KEY' not found via KeyProvider." in caplog.text

def _create_mock_openai_chat_completion(
    content: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    finish_reason: str = "stop",
    usage_dict: Optional[Dict[str, int]] = None
) -> MagicMock:
    mock_completion = MagicMock(name="MockChatCompletionResponse")
    mock_message = MagicMock(spec=OpenAIChatMessage_ToUse)
    mock_message.role = "assistant"
    mock_message.content = content
    mock_message.tool_calls = None

    if tool_calls:
        mock_message.tool_calls = []
        for tc_data in tool_calls:
            mock_tc = MagicMock(spec=OpenAIToolCall_ToUse)
            mock_tc.id = tc_data["id"]
            mock_tc.type = tc_data["type"]
            mock_tc.function = MagicMock()
            mock_tc.function.name = tc_data["function"]["name"]
            mock_tc.function.arguments = tc_data["function"]["arguments"]
            mock_message.tool_calls.append(mock_tc)

    mock_choice = MagicMock(spec=OpenAIChoice_ToUse)
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason
    mock_completion.choices = [mock_choice]

    if usage_dict:
        mock_usage = MagicMock(spec=OpenAIUsage_ToUse)
        mock_usage.prompt_tokens = usage_dict.get("prompt_tokens")
        mock_usage.completion_tokens = usage_dict.get("completion_tokens")
        mock_usage.total_tokens = usage_dict.get("total_tokens")
        mock_completion.usage = mock_usage
    else:
        mock_completion.usage = None
    mock_completion.model_dump = MagicMock(return_value={"id": "chatcmpl-mock", "choices": [{"message": {"content": content}}]})
    return mock_completion


@pytest.mark.asyncio
async def test_openai_generate_success(openai_provider: OpenAILLMProviderPlugin):
    provider = await openai_provider
    assert provider._client is not None, "Client should be initialized by fixture"
    prompt = "Write a haiku."
    expected_text = "An old silent pond...\nA frog jumps into the pond,\nsplash! Silence again."
    mock_response = _create_mock_openai_chat_completion(
        content=expected_text,
        usage_dict={"prompt_tokens": 10, "completion_tokens": 17, "total_tokens": 27}
    )
    provider._client.chat.completions.create.return_value = mock_response

    result = await provider.generate(prompt=prompt, temperature=0.7)

    assert result["text"] == expected_text
    assert result["finish_reason"] == "stop"
    assert result["usage"] is not None
    assert result["usage"]["total_tokens"] == 27
    provider._client.chat.completions.create.assert_awaited_once()
    call_kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-test-model"
    assert call_kwargs["messages"] == [{"role": "user", "content": prompt}]
    assert call_kwargs["temperature"] == 0.7

@pytest.mark.asyncio
async def test_openai_chat_success_text_response(openai_provider: OpenAILLMProviderPlugin):
    provider = await openai_provider
    assert provider._client is not None, "Client should be initialized by fixture"
    messages: List[ChatMessage] = [{"role": "user", "content": "Hello!"}]
    expected_content = "Hi there! How can I help you today?"
    mock_response = _create_mock_openai_chat_completion(
        content=expected_content,
        usage_dict={"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
    )
    provider._client.chat.completions.create.return_value = mock_response

    result = await provider.chat(messages=messages)

    assert result["message"]["role"] == "assistant"
    assert result["message"]["content"] == expected_content
    assert result["message"].get("tool_calls") is None
    assert result["finish_reason"] == "stop"
    assert result["usage"] is not None
    assert result["usage"]["total_tokens"] == 15

@pytest.mark.asyncio
async def test_openai_chat_tool_call_response(openai_provider: OpenAILLMProviderPlugin):
    provider = await openai_provider
    assert provider._client is not None, "Client should be initialized by fixture"
    messages: List[ChatMessage] = [{"role": "user", "content": "What's the weather in Boston?"}]
    tool_calls_data = [{
        "id": "call_abc123", "type": "function",
        "function": {"name": "get_weather", "arguments": '{"location": "Boston, MA"}'}
    }]
    mock_response = _create_mock_openai_chat_completion(
        tool_calls=tool_calls_data,
        finish_reason="tool_calls",
        usage_dict={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
    )
    provider._client.chat.completions.create.return_value = mock_response

    openai_tools_param = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
        }
    }]
    result = await provider.chat(messages=messages, tools=openai_tools_param, tool_choice="auto")

    assert result["finish_reason"] == "tool_calls"
    assert result["message"]["role"] == "assistant"
    assert result["message"]["content"] is None
    assert result["message"]["tool_calls"] is not None
    returned_tool_calls: List[ToolCall] = result["message"]["tool_calls"] # type: ignore
    assert len(returned_tool_calls) == 1
    assert returned_tool_calls[0]["id"] == "call_abc123"
    assert returned_tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(returned_tool_calls[0]["function"]["arguments"]) == {"location": "Boston, MA"}

    provider._client.chat.completions.create.assert_awaited_once()
    call_kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tools"] == openai_tools_param
    assert call_kwargs["tool_choice"] == "auto"

def _create_mock_error_response_for_openai(status_code: int, error_message: str, request_obj: Any, error_type: Optional[str] = "api_error") -> Any:
    """Helper to create a mock response object that OpenAI errors might carry."""
    if httpx: # If real httpx is available
        # Ensure request_obj is a valid httpx.Request or compatible mock
        if not isinstance(request_obj, httpx.Request):
            # Create a minimal httpx.Request if a generic mock was passed
            request_obj = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

        mock_resp = httpx.Response(
            status_code,
            json={"error": {"message": error_message, "type": error_type}},
            request=request_obj
        )
        return mock_resp
    # Fallback to minimal mock if httpx is not available
    mock_resp = MagicMock(name=f"MockHTTPXResponse_{status_code}")
    mock_resp.status_code = status_code
    mock_resp.json = MagicMock(return_value={"error": {"message": error_message, "type": error_type}})
    mock_resp.text = json.dumps({"error": {"message": error_message, "type": error_type}})
    mock_resp.headers = {}
    mock_resp.request = request_obj # Manually set for the mock
    return mock_resp

def _create_mock_request_for_openai() -> Any:
    if httpx:
        return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    # Fallback to minimal mock if httpx is not available
    return MagicMock(name="MockMinimalHTTPXRequest", method="POST", url="https://api.openai.com/v1/chat/completions")


@pytest.mark.asyncio
async def test_openai_api_error_handling(openai_provider: OpenAILLMProviderPlugin, caplog: pytest.LogCaptureFixture):
    provider = await openai_provider
    assert provider._client is not None, "Client should be initialized by fixture"
    caplog.set_level(logging.ERROR)

    error_message = "Invalid API key"
    status_code = 401
    error_body = {"error": {"message": error_message, "type": "invalid_request_error"}}

    mock_request_obj = _create_mock_request_for_openai()
    mock_response_obj = _create_mock_error_response_for_openai(status_code, error_message, mock_request_obj, error_type="invalid_request_error")

    # Use AuthenticationError_ToUse for a 401 error, as it's an APIStatusError
    error_to_raise = AuthenticationError_ToUse(
        message=error_message,
        response=mock_response_obj,
        body=error_body
    )

    provider._client.chat.completions.create.side_effect = error_to_raise

    with pytest.raises(RuntimeError) as excinfo:
        await provider.chat(messages=[{"role": "user", "content": "test"}])

    assert f"OpenAI API Error: {status_code} - {error_message}" in str(excinfo.value)
    assert f"OpenAI API Error during chat: {status_code} - {error_message}" in caplog.text

@pytest.mark.asyncio
async def test_openai_rate_limit_error_handling(openai_provider: OpenAILLMProviderPlugin, caplog: pytest.LogCaptureFixture):
    provider = await openai_provider
    assert provider._client is not None, "Client should be initialized by fixture"
    caplog.set_level(logging.ERROR)

    error_message = "Rate limit exceeded"
    status_code = 429
    error_body = {"error": {"message": error_message, "type": "rate_limit_error"}}

    mock_request_obj = _create_mock_request_for_openai()
    mock_response_obj = _create_mock_error_response_for_openai(status_code, error_message, mock_request_obj, error_type="rate_limit_error")

    error_to_raise = RateLimitError_ToUse(
        message=error_message,
        response=mock_response_obj,
        body=error_body
    )

    provider._client.chat.completions.create.side_effect = error_to_raise

    with pytest.raises(RuntimeError) as excinfo:
        await provider.chat(messages=[{"role": "user", "content": "test"}])
    assert f"OpenAI API Error: {status_code} - {error_message}" in str(excinfo.value)
    assert "OpenAI API Error during chat" in caplog.text

@pytest.mark.asyncio
async def test_openai_get_model_info(openai_provider: OpenAILLMProviderPlugin):
    provider = await openai_provider
    assert provider._client is not None, "Client should be initialized by fixture"
    info = await provider.get_model_info()
    assert info["provider"] == "OpenAI"
    assert info["configured_model_name"] == "gpt-test-model"
    assert "OpenAI documentation" in info["notes"]

@pytest.mark.asyncio
async def test_openai_teardown(openai_provider: OpenAILLMProviderPlugin):
    provider = await openai_provider
    assert provider._client is not None, "Client should be initialized by fixture"
    client_instance_mock = provider._client
    await provider.teardown()
    client_instance_mock.close.assert_awaited_once()
    assert provider._client is None
    assert provider._key_provider is None
