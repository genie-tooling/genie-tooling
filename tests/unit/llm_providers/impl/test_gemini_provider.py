### tests/unit/llm_providers/impl/test_gemini_provider.py
import logging
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.llm_providers.impl.gemini_provider import (
    GEMINI_SDK_AVAILABLE,
    GeminiLLMProviderPlugin,
)
from genie_tooling.llm_providers.types import ChatMessage, LLMChatResponse
from genie_tooling.security.key_provider import KeyProvider

# Mock the genai library if it's not installed, for basic test structure validation
if GEMINI_SDK_AVAILABLE:
    import google.genai.types as genai_types
    from google import genai
    from pydantic import BaseModel
else:
    # Create mock objects if the library isn't available
    genai = MagicMock()
    genai_types = MagicMock()
    # Configure the mocks
    class MockFinishReasonEnum:
        STOP = "STOP"
        TOOL_CALL = "TOOL_CALL"
        MAX_TOKENS = "MAX_TOKENS"
        SAFETY = "SAFETY"
        RECITATION = "RECITATION"
        OTHER = "OTHER"
    genai_types.FinishReason = MockFinishReasonEnum()

    class MockBlockReasonEnum:
        SAFETY = "SAFETY"
        OTHER = "OTHER"
    genai_types.BlockReason = MockBlockReasonEnum()

    # Mock Classes
    def mock_from_text(text):
        part_instance = MagicMock(name="MockPartInstanceFromText")
        part_instance.text = text
        part_instance.function_response = None
        part_instance.function_call = None
        return part_instance

    def mock_from_function_response(name, response):
        part_instance = MagicMock(name="MockPartInstanceFromFuncResp")
        function_response_instance = MagicMock(name="MockFuncRespInstance")
        function_response_instance.name = name
        function_response_instance.response = response
        part_instance.function_response = function_response_instance
        part_instance.text = None
        part_instance.function_call = None
        return part_instance

    def mock_from_function_call(name, args):
        part_instance = MagicMock(name="MockPartInstanceFromFuncCall")
        function_call_instance = MagicMock(name="MockFuncCallInstance")
        function_call_instance.name = name
        function_call_instance.args = args
        part_instance.function_call = function_call_instance
        part_instance.text = None
        part_instance.function_response = None
        return part_instance

    MockPartClass = type("MockPart", (), {})
    MockPartClass.from_text = mock_from_text
    MockPartClass.from_function_response = mock_from_function_response
    MockPartClass.from_function_call = mock_from_function_call
    genai_types.Part = MockPartClass

    MockContentClass = type("MockContent", (), {})
    def mock_content_init(self, role=None, parts=None):
        self.role = role
        self.parts = parts or []
    MockContentClass.__init__ = mock_content_init
    genai_types.Content = MockContentClass

    genai_types.FunctionCall = type("MockFunctionCall", (), {})
    genai_types.GenerateContentResponse = type("MockGenerateContentResponse", (), {})
    genai_types.FunctionResponse = type("MockFunctionResponse", (), {})

    BaseModel = object

PROVIDER_LOGGER_NAME = "genie_tooling.llm_providers.impl.gemini_provider"


# --- FIX: Helper function to create a more realistic mock response ---
def create_mock_gemini_response(
    content: str = "Default response",
    finish_reason_enum=genai_types.FinishReason.STOP,
    tool_calls: Optional[List[Any]] = None
) -> MagicMock:
    """Creates a structured mock for genai.types.GenerateContentResponse."""
    response = MagicMock(spec=genai_types.GenerateContentResponse)

    # Mock the candidate part
    candidate = MagicMock()
    candidate.finish_reason = finish_reason_enum

    # Mock the content part
    content_obj = MagicMock()
    parts_list = []
    if content:
        part_obj = MagicMock()
        part_obj.text = content
        parts_list.append(part_obj)
    if tool_calls:
        for tc in tool_calls:
            parts_list.append(tc)

    content_obj.parts = parts_list
    candidate.content = content_obj

    response.candidates = [candidate]
    response.text = content
    response.function_calls = tool_calls
    response.usage_metadata = None
    response.prompt_feedback = None

    return response


@pytest.fixture()
async def gemini_provider_with_mocks(
    mock_key_provider: KeyProvider,
) -> GeminiLLMProviderPlugin:
    """
    Provides an initialized GeminiLLMProviderPlugin and attaches its mocked internal client
    to a test-only attribute for easy access in tests.
    """
    provider = GeminiLLMProviderPlugin()
    kp = await mock_key_provider

    # This mock represents the genai.Client instance
    mock_client = AsyncMock(name="MockGenAIClientInstance")
    mock_client.aio = AsyncMock(name="MockAIOClient")
    mock_client.aio.models = AsyncMock(name="MockAIOModels")
    mock_client.aio.models.get = AsyncMock(name="MockGetModelInfo")
    mock_client.aio.models.generate_content = AsyncMock(name="MockGenerateContent")
    mock_client.aio.models.generate_content_stream = AsyncMock(
        name="MockGenerateContentStream"
    )

    # Patch the constructor to return our mock instance
    with patch(
        "genie_tooling.llm_providers.impl.gemini_provider.genai.Client",
        return_value=mock_client,
    ):
        # Attach the mock to the provider instance for test access
        provider._test_mock_client = mock_client  # type: ignore
        await provider.setup(config={"key_provider": kp})
        return provider


@pytest.mark.asyncio()
async def test_gemini_setup_no_api_key(
    mock_key_provider: KeyProvider, caplog: pytest.LogCaptureFixture
):
    """Test setup warns when the API key is not found but still tries ADC."""
    caplog.set_level(logging.INFO, logger=PROVIDER_LOGGER_NAME)
    provider = GeminiLLMProviderPlugin()
    actual_kp = await mock_key_provider
    actual_kp.get_key = AsyncMock(return_value=None)  # type: ignore

    with patch(
        "genie_tooling.llm_providers.impl.gemini_provider.genai.Client"
    ) as mock_client_constructor:
        await provider.setup(
            config={"api_key_name": "ANY_KEY_NAME_HERE", "key_provider": actual_kp}
        )

    assert (
        "API key 'ANY_KEY_NAME_HERE' not found" in caplog.text
    ), "Warning about missing key not found in logs"
    # It should still attempt to initialize with ADC
    mock_client_constructor.assert_called_once()
    assert "api_key" not in mock_client_constructor.call_args.kwargs


@pytest.mark.asyncio()
async def test_gemini_setup_client_init_fails(
    caplog: pytest.LogCaptureFixture, mock_key_provider: KeyProvider
):
    """Test setup fails gracefully when the Gemini client constructor raises an error."""
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = GeminiLLMProviderPlugin()
    kp = await mock_key_provider

    with patch(
        "genie_tooling.llm_providers.impl.gemini_provider.genai.Client",
        side_effect=ValueError("Invalid model name"),
    ):
        await provider.setup(config={"key_provider": kp})

    assert provider._client is None
    assert "Failed to initialize Gemini client: Invalid model name" in caplog.text


@pytest.mark.skipif(not GEMINI_SDK_AVAILABLE, reason="Gemini SDK not installed")
@pytest.mark.asyncio()
class TestGeminiMessageConversion:
    """
    Tests the internal message conversion logic by inspecting the payload
    sent to the mocked Gemini SDK via the public `chat` method.
    """

    async def test_convert_user_and_assistant_roles(
        self, gemini_provider_with_mocks: GeminiLLMProviderPlugin
    ):
        provider = await gemini_provider_with_mocks
        mock_client = provider._test_mock_client  # type: ignore
        messages: List[ChatMessage] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        # --- FIX: Use helper to create a realistic mock response ---
        mock_client.aio.models.generate_content.return_value = create_mock_gemini_response()
        await provider.chat(messages)
        mock_client.aio.models.generate_content.assert_awaited_once()
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs.get("system_instruction") is None
        contents = call_kwargs.get("contents", [])
        assert len(contents) == 2
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "Hello"
        assert contents[1].role == "model"
        assert contents[1].parts[0].text == "Hi there!"

    async def test_convert_system_role(
        self, gemini_provider_with_mocks: GeminiLLMProviderPlugin
    ):
        provider = await gemini_provider_with_mocks
        mock_client = provider._test_mock_client  # type: ignore
        messages: List[ChatMessage] = [
            {"role": "system", "content": "You are a helpful bot."}
        ]
        # --- FIX: Use helper to create a realistic mock response ---
        mock_client.aio.models.generate_content.return_value = create_mock_gemini_response()
        await provider.chat(messages)
        mock_client.aio.models.generate_content.assert_awaited_once()
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs.get("contents") == []
        assert call_kwargs.get("system_instruction") is not None
        assert call_kwargs["system_instruction"].parts[0].text == "You are a helpful bot."

    async def test_convert_tool_call_request(
        self, gemini_provider_with_mocks: GeminiLLMProviderPlugin
    ):
        provider = await gemini_provider_with_mocks
        mock_client = provider._test_mock_client  # type: ignore
        messages: List[ChatMessage] = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "London"}',
                        },
                    }
                ],
            }
        ]
        # --- FIX: Use helper to create a realistic mock response ---
        mock_client.aio.models.generate_content.return_value = create_mock_gemini_response()
        await provider.chat(messages)
        mock_client.aio.models.generate_content.assert_awaited_once()
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        contents = call_kwargs.get("contents", [])
        assert len(contents) == 1
        assert contents[0].role == "model"
        assert isinstance(contents[0].parts[0], genai_types.Part)
        assert contents[0].parts[0].function_call.name == "get_weather"
        assert contents[0].parts[0].function_call.args == {"city": "London"}

    async def test_convert_tool_response(
        self, gemini_provider_with_mocks: GeminiLLMProviderPlugin
    ):
        provider = await gemini_provider_with_mocks
        mock_client = provider._test_mock_client  # type: ignore
        messages: List[ChatMessage] = [
            {
                "role": "tool",
                "tool_call_id": "tc1",
                "name": "get_weather",
                "content": '{"temperature": 15, "unit": "celsius"}',
            }
        ]
        # --- FIX: Use helper to create a realistic mock response ---
        mock_client.aio.models.generate_content.return_value = create_mock_gemini_response()
        await provider.chat(messages)
        mock_client.aio.models.generate_content.assert_awaited_once()
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        contents = call_kwargs.get("contents", [])
        assert len(contents) == 1
        assert contents[0].role == "function"
        assert isinstance(contents[0].parts[0], genai_types.Part)
        assert contents[0].parts[0].function_response.name == "get_weather"
        assert contents[0].parts[0].function_response.response == {"temperature": 15, "unit": "celsius"}


    async def test_convert_tool_message_no_content(
        self, gemini_provider_with_mocks: GeminiLLMProviderPlugin
    ):
        provider = await gemini_provider_with_mocks
        mock_client = provider._test_mock_client  # type: ignore
        messages: List[ChatMessage] = [
            {"role": "tool", "tool_call_id": "tc1", "name": "tool_name"}
        ]
        # --- FIX: Use helper to create a realistic mock response ---
        mock_client.aio.models.generate_content.return_value = create_mock_gemini_response()
        await provider.chat(messages)
        mock_client.aio.models.generate_content.assert_awaited_once()
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        contents = call_kwargs.get("contents", [])
        assert len(contents) == 1
        assert contents[0].role == "function"
        assert contents[0].parts[0].function_response.response == {"content": "None"}

    async def test_convert_tool_response_non_json_content(
        self, gemini_provider_with_mocks: GeminiLLMProviderPlugin
    ):
        provider = await gemini_provider_with_mocks
        mock_client = provider._test_mock_client  # type: ignore
        messages: List[ChatMessage] = [
            {
                "role": "tool",
                "tool_call_id": "tc1",
                "name": "get_status",
                "content": "OK",
            }
        ]
        # --- FIX: Use helper to create a realistic mock response ---
        mock_client.aio.models.generate_content.return_value = create_mock_gemini_response()
        await provider.chat(messages)
        mock_client.aio.models.generate_content.assert_awaited_once()
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        contents = call_kwargs.get("contents", [])
        assert len(contents) == 1
        assert contents[0].role == "function"
        assert contents[0].parts[0].function_response.response == {"content": "OK"}

    async def test_convert_assistant_message_no_content_no_tool_calls(
        self, gemini_provider_with_mocks: GeminiLLMProviderPlugin
    ):
        provider = await gemini_provider_with_mocks
        mock_client = provider._test_mock_client  # type: ignore
        messages: List[ChatMessage] = [{"role": "assistant"}]
        # --- FIX: Use helper to create a realistic mock response ---
        mock_client.aio.models.generate_content.return_value = create_mock_gemini_response()
        await provider.chat(messages)
        mock_client.aio.models.generate_content.assert_awaited_once()
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs.get("contents") == []


@pytest.mark.asyncio()
async def test_generate_api_error(gemini_provider_with_mocks: GeminiLLMProviderPlugin):
    provider = await gemini_provider_with_mocks
    mock_client = provider._test_mock_client  # type: ignore
    async def mock_side_effect(*args, **kwargs):
        raise Exception("Gemini API is down")
    mock_client.aio.models.generate_content.side_effect = mock_side_effect

    with pytest.raises(RuntimeError, match="Gemini API call failed: Gemini API is down"):
        await provider.generate("test prompt")


@pytest.mark.asyncio()
async def test_chat_api_error(gemini_provider_with_mocks: GeminiLLMProviderPlugin):
    provider = await gemini_provider_with_mocks
    mock_client = provider._test_mock_client  # type: ignore
    async def mock_side_effect(*args, **kwargs):
        raise Exception("Gemini Chat API is down")
    mock_client.aio.models.generate_content.side_effect = mock_side_effect

    with pytest.raises(
        RuntimeError, match="Gemini API call failed: Gemini Chat API is down"
    ):
        await provider.chat([{"role": "user", "content": "test"}])


@pytest.mark.skipif(not GEMINI_SDK_AVAILABLE, reason="Gemini SDK not installed")
@pytest.mark.asyncio()
async def test_chat_streaming_success(gemini_provider_with_mocks: GeminiLLMProviderPlugin):
    provider = await gemini_provider_with_mocks
    mock_client = provider._test_mock_client  # type: ignore

    mock_chunk1 = MagicMock()
    type(mock_chunk1).text = "Hello "
    type(mock_chunk1).function_calls = None
    type(mock_chunk1).usage_metadata = None
    mock_chunk2 = MagicMock()
    type(mock_chunk2).text = "World!"
    type(mock_chunk2).function_calls = None
    type(mock_chunk2).usage_metadata = None

    agg_response = MagicMock()
    mock_finish_reason_member = MagicMock()
    mock_finish_reason_member.name = "STOP"
    type(agg_response).candidates = [
        MagicMock(finish_reason=mock_finish_reason_member)
    ]
    type(agg_response).usage_metadata = MagicMock(
        prompt_token_count=10, candidates_token_count=5, total_token_count=15
    )
    mock_stream_obj = AsyncMock()
    mock_stream_obj.__aiter__.return_value = [mock_chunk1, mock_chunk2]
    mock_stream_obj.aggregate_response = AsyncMock(return_value=agg_response)
    mock_client.aio.models.generate_content_stream.return_value = mock_stream_obj

    result_stream = await provider.chat(
        [{"role": "user", "content": "test"}], stream=True
    )
    chunks = [chunk async for chunk in result_stream]

    assert len(chunks) == 3
    assert chunks[0]["message_delta"]["content"] == "Hello "
    assert chunks[1]["message_delta"]["content"] == "World!"
    assert chunks[2]["finish_reason"] == "stop"
    assert chunks[2]["usage_delta"]["total_tokens"] == 15


@pytest.mark.skipif(not GEMINI_SDK_AVAILABLE, reason="Gemini SDK not installed")
@pytest.mark.asyncio()
async def test_chat_streaming_with_tool_calls(
    gemini_provider_with_mocks: GeminiLLMProviderPlugin,
):
    provider = await gemini_provider_with_mocks
    mock_client = provider._test_mock_client  # type: ignore

    mock_fc1 = MagicMock(spec=genai_types.FunctionCall)
    type(mock_fc1).name = "get_weather"
    type(mock_fc1).args = {"city": "Lon"}
    mock_chunk1 = MagicMock()
    type(mock_chunk1).function_calls = [mock_fc1]
    type(mock_chunk1).text = None
    mock_fc2 = MagicMock(spec=genai_types.FunctionCall)
    type(mock_fc2).name = "get_weather"
    type(mock_fc2).args = {"city": "London"}
    mock_chunk2 = MagicMock()
    type(mock_chunk2).function_calls = [mock_fc2]
    type(mock_chunk2).text = None

    mock_stream_obj = AsyncMock()
    mock_stream_obj.__aiter__.return_value = [mock_chunk1, mock_chunk2]
    agg_response = MagicMock()
    mock_finish_reason_member = MagicMock()
    mock_finish_reason_member.name = "TOOL_CALL"
    type(agg_response).candidates = [
        MagicMock(finish_reason=mock_finish_reason_member)
    ]
    # --- FIX: Add usage_metadata to the aggregated response mock ---
    type(agg_response).usage_metadata = MagicMock(
        prompt_token_count=20, candidates_token_count=15, total_token_count=35
    )
    mock_stream_obj.aggregate_response = AsyncMock(return_value=agg_response)
    mock_client.aio.models.generate_content_stream.return_value = mock_stream_obj

    result_stream = await provider.chat(
        [{"role": "user", "content": "test"}], stream=True
    )
    chunks = [chunk async for chunk in result_stream]

    assert len(chunks) == 3
    assert chunks[0]["message_delta"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert '"city": "Lon"' in chunks[0]["message_delta"]["tool_calls"][0]["function"]["arguments"]
    assert '"city": "London"' in chunks[1]["message_delta"]["tool_calls"][0]["function"]["arguments"]
    assert chunks[2]["finish_reason"] == "tool_call"
    # --- FIX: Assert on the final chunk's usage data ---
    assert chunks[2]["usage_delta"]["total_tokens"] == 35


@pytest.mark.skipif(not GEMINI_SDK_AVAILABLE, reason="Gemini SDK not installed")
@pytest.mark.asyncio()
async def test_chat_blocked_response(gemini_provider_with_mocks: GeminiLLMProviderPlugin):
    provider = await gemini_provider_with_mocks
    mock_client = provider._test_mock_client  # type: ignore
    mock_response = MagicMock(spec=genai_types.GenerateContentResponse)
    type(mock_response).candidates = []
    mock_prompt_feedback = MagicMock()
    mock_block_reason_member = MagicMock()
    mock_block_reason_member.name = "SAFETY"
    type(mock_prompt_feedback).block_reason = mock_block_reason_member
    type(mock_response).prompt_feedback = mock_prompt_feedback
    mock_client.aio.models.generate_content.return_value = mock_response

    result: LLMChatResponse = await provider.chat([{"role": "user", "content": "risky"}])

    assert result["finish_reason"] == "blocked: safety"
    assert "[Chat blocked: SAFETY]" in result["message"]["content"]


@pytest.mark.skipif(not GEMINI_SDK_AVAILABLE, reason="Gemini SDK not installed")
@pytest.mark.asyncio()
async def test_get_model_info_success(gemini_provider_with_mocks: GeminiLLMProviderPlugin):
    provider = await gemini_provider_with_mocks
    mock_client = provider._test_mock_client  # type: ignore
    mock_model_info = MagicMock(spec=genai_types.Model)
    type(mock_model_info).name = "models/gemini-1.5-flash-latest"
    type(mock_model_info).display_name = "Gemini 1.5 Flash"
    type(mock_model_info).version = "1.0"
    type(mock_model_info).input_token_limit = 1048576
    type(mock_model_info).output_token_limit = 8192
    type(mock_model_info).supported_generation_methods = ["generateContent", "streamGenerateContent"]
    type(mock_model_info).temperature = 0.9
    type(mock_model_info).top_p = 1.0
    type(mock_model_info).top_k = 32
    mock_client.aio.models.get.return_value = mock_model_info

    info = await provider.get_model_info()

    assert info["provider"] == "Google Gemini"
    assert info["display_name"] == "Gemini 1.5 Flash"
    assert info["version"] == "1.0"
    assert info["input_token_limit"] == 1048576


@pytest.mark.asyncio()
async def test_get_model_info_api_fails(gemini_provider_with_mocks: GeminiLLMProviderPlugin):
    provider = await gemini_provider_with_mocks
    mock_client = provider._test_mock_client  # type: ignore
    mock_client.aio.models.get.side_effect = Exception("API call failed")

    info = await provider.get_model_info()

    assert "error_retrieving_details" in info
    assert "API call failed" in info["error_retrieving_details"]
