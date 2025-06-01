### tests/unit/llm_providers/impl/test_gemini_provider.py
import json
import logging
from typing import Any, AsyncIterable, Dict, List, NamedTuple, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genie_tooling.llm_providers.impl.gemini_provider import GeminiLLMProviderPlugin
from genie_tooling.llm_providers.types import (
    ChatMessage,
    LLMChatChunk,
    LLMCompletionChunk,
    ToolCall,
)
from genie_tooling.security.key_provider import KeyProvider


# --- Mocks for Gemini SDK types ---
class MockGeminiCandidate(NamedTuple): content: Optional[Any]; finish_reason: Optional[Any]
class MockGeminiContent(NamedTuple): parts: List[Any]; role: str
class MockGeminiPart(NamedTuple): text: Optional[str] = None; function_call: Optional[Any] = None
class MockGeminiFunctionCall(NamedTuple): name: str; args: Dict[str, Any]
class MockGeminiGenerateContentResponse(NamedTuple):
    candidates: Optional[List[MockGeminiCandidate]]; prompt_feedback: Optional[Any]
    text: Optional[str]; usage_metadata: Optional[Any]
    def to_dict(self) -> dict: return {"candidates": self.candidates, "prompt_feedback": self.prompt_feedback, "usage_metadata": self.usage_metadata}
class MockGeminiUsageMetadata(NamedTuple): prompt_token_count: int; candidates_token_count: int; total_token_count: int
class MockFinishReasonEnum:
    STOP = 1; MAX_TOKENS = 2; SAFETY = 3; RECITATION = 4; UNSPECIFIED = 0; OTHER = 5; TOOL_CODE = 6
    def __init__(self, value: int): self.value = value; self.name = {v: k for k,v in self.__class__.__dict__.items() if isinstance(v,int)}.get(value, "UNKNOWN")

# Mock for streaming response chunks
class MockAsyncGenerateContentResponseChunk:
    def __init__(self, text_delta: Optional[str] = None, candidates: Optional[List[MockGeminiCandidate]] = None, usage_metadata: Optional[MockGeminiUsageMetadata] = None):
        self.text = text_delta # For generate stream
        self.candidates = candidates # For chat stream
        self.usage_metadata = usage_metadata # For final chunk with usage
        self.prompt_feedback = None # Usually None for intermediate chunks

    def to_dict(self) -> dict: # Simplified for testing
        return {"text_delta": self.text, "candidates": self.candidates, "usage_metadata": self.usage_metadata}


@pytest.fixture
def mock_genai_lib():
    mock_lib = MagicMock(name="MockGoogleGenerativeAIModule")
    mock_lib.configure = MagicMock()
    mock_model_inst = MagicMock(name="MockGenerativeModelInstance")
    # Ensure generate_content_async is an AsyncMock
    mock_model_inst.generate_content_async = AsyncMock(return_value=MockGeminiGenerateContentResponse([], None, None, None))
    mock_lib.GenerativeModel = MagicMock(return_value=mock_model_inst)
    mock_lib.types = MagicMock()
    mock_lib.types.GenerationConfig = MagicMock()
    mock_lib.types.ContentDict = dict
    mock_lib.types.Tool = MagicMock()
    mock_lib.types.FunctionDeclaration = MagicMock()
    mock_lib.types.generation_types = MagicMock()
    mock_lib.types.generation_types.GenerateContentResponse = MockGeminiGenerateContentResponse
    mock_lib.types.generation_types.Candidate = MockGeminiCandidate
    # Add mock for AsyncGenerateContentResponse for streaming
    mock_lib.types.AsyncGenerateContentResponse = type("MockAsyncGenerateContentResponse", (), {})

    with patch("genie_tooling.llm_providers.impl.gemini_provider.genai", mock_lib) as p_genai, \
         patch("genie_tooling.llm_providers.impl.gemini_provider._GenerativeModelTypePlaceholder", new=mock_lib.GenerativeModel), \
         patch("genie_tooling.llm_providers.impl.gemini_provider._GenerationConfigType", new=mock_lib.types.GenerationConfig), \
         patch("genie_tooling.llm_providers.impl.gemini_provider._ContentDictType", new=dict), \
         patch("genie_tooling.llm_providers.impl.gemini_provider._GeminiSDKToolType", new=mock_lib.types.Tool), \
         patch("genie_tooling.llm_providers.impl.gemini_provider._FunctionDeclarationType", new=mock_lib.types.FunctionDeclaration), \
         patch("genie_tooling.llm_providers.impl.gemini_provider._GenerateContentResponseType", new=mock_lib.types.generation_types.GenerateContentResponse), \
         patch("genie_tooling.llm_providers.impl.gemini_provider._CandidateType", new=mock_lib.types.generation_types.Candidate), \
         patch("genie_tooling.llm_providers.impl.gemini_provider._AsyncGenerateContentResponseType", new=mock_lib.types.AsyncGenerateContentResponse): # Patch for streaming
        yield p_genai

@pytest.fixture
async def gemini_provider(mock_genai_lib: MagicMock, mock_key_provider: KeyProvider) -> GeminiLLMProviderPlugin:
    provider = GeminiLLMProviderPlugin()
    actual_mock_key_provider = await mock_key_provider
    await provider.setup(config={"model_name": "test-gemini-model", "key_provider": actual_mock_key_provider})
    assert provider._model_client is mock_genai_lib.GenerativeModel.return_value
    return provider

@pytest.mark.asyncio
async def test_gemini_setup_success(mock_genai_lib: MagicMock, mock_key_provider: KeyProvider):
    provider = GeminiLLMProviderPlugin()
    actual_kp = await mock_key_provider
    api_key_name_for_setup = provider._api_key_name
    model_name="gemini-pro-test"
    system_instruction="You are a test bot."
    safety_settings=[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT", "threshold":"BLOCK_ONLY_HIGH"}]
    await provider.setup(config={"api_key_name": api_key_name_for_setup, "model_name": model_name, "system_instruction": system_instruction, "safety_settings": safety_settings, "key_provider": actual_kp})
    mock_genai_lib.configure.assert_called_once_with(api_key="test_google_key_from_conftest_fixture")
    mock_genai_lib.GenerativeModel.assert_called_once_with(model_name=model_name, system_instruction=system_instruction, safety_settings=safety_settings)
    assert provider._model_client is not None

@pytest.mark.asyncio
async def test_gemini_setup_no_api_key(mock_genai_lib: MagicMock, caplog: pytest.LogCaptureFixture, mock_key_provider: KeyProvider):
    caplog.set_level(logging.ERROR)
    provider = GeminiLLMProviderPlugin()
    actual_kp = await mock_key_provider
    actual_kp.get_key = AsyncMock(return_value=None) # type: ignore

    await provider.setup(config={"api_key_name": "ANY_KEY_NAME_HERE", "key_provider": actual_kp})
    assert provider._model_client is None
    assert "API key 'ANY_KEY_NAME_HERE' not found via KeyProvider." in caplog.text

@pytest.mark.asyncio
async def test_gemini_convert_messages_complex_tool_calls_and_responses(gemini_provider: GeminiLLMProviderPlugin):
    provider = await gemini_provider
    messages: List[ChatMessage] = [
        {"role": "user", "content": "What's the weather in London and Paris?"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_london", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "London", "unit": "celsius"}'}},
                {"id": "call_paris", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris", "unit": "celsius"}'}}
            ]
        },
        {"role": "tool", "tool_call_id": "call_london", "name": "get_weather", "content": '{"temperature": 15, "condition": "Cloudy"}'},
        {"role": "tool", "tool_call_id": "call_paris", "name": "get_weather", "content": '{"temperature": 18, "condition": "Sunny", "details": {"humidity": 60}}'},
        {"role": "assistant", "content": "London is 15C and Cloudy. Paris is 18C and Sunny."}
    ]
    gemini_msgs = provider._convert_messages_to_gemini(messages)
    assert len(gemini_msgs) == 5
    assert gemini_msgs[0]["role"] == "user"
    assert gemini_msgs[1]["role"] == "model"
    assert len(gemini_msgs[1]["parts"]) == 2
    assert gemini_msgs[1]["parts"][0]["function_call"]["name"] == "get_weather"
    assert gemini_msgs[1]["parts"][0]["function_call"]["args"] == {"city": "London", "unit": "celsius"}
    assert gemini_msgs[1]["parts"][1]["function_call"]["name"] == "get_weather"
    assert gemini_msgs[1]["parts"][1]["function_call"]["args"] == {"city": "Paris", "unit": "celsius"}

    assert gemini_msgs[2]["role"] == "tool"
    assert gemini_msgs[2]["parts"][0]["function_response"]["name"] == "get_weather"
    assert gemini_msgs[2]["parts"][0]["function_response"]["response"] == {"temperature": 15, "condition": "Cloudy"}

    assert gemini_msgs[3]["role"] == "tool"
    assert gemini_msgs[3]["parts"][0]["function_response"]["name"] == "get_weather"
    assert gemini_msgs[3]["parts"][0]["function_response"]["response"] == {"temperature": 18, "condition": "Sunny", "details": {"humidity": 60}}

    assert gemini_msgs[4]["role"] == "model"
    assert gemini_msgs[4]["parts"][0]["text"] == "London is 15C and Cloudy. Paris is 18C and Sunny."

@pytest.mark.asyncio
async def test_gemini_chat_with_tool_schema_and_gemini_response_parsing(gemini_provider: GeminiLLMProviderPlugin, mock_genai_lib: MagicMock):
    provider = await gemini_provider
    get_weather_func_decl = mock_genai_lib.types.FunctionDeclaration(
        name="get_current_weather",
        description="Get the current weather in a given location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    )
    tools_for_api = [mock_genai_lib.types.Tool(function_declarations=[get_weather_func_decl])]
    mock_gemini_fc_args = {"location": "Boston, MA", "unit": "celsius"}
    mock_gemini_fc = MockGeminiFunctionCall(name="get_current_weather", args=mock_gemini_fc_args)
    mock_candidate_tool_call = MockGeminiCandidate(
        content=MockGeminiContent(parts=[MockGeminiPart(function_call=mock_gemini_fc)], role="model"),
        finish_reason=MockFinishReasonEnum(MockFinishReasonEnum.TOOL_CODE)
    )
    # Use generate_content_async for the mock
    provider._model_client.generate_content_async.return_value = MockGeminiGenerateContentResponse(
        candidates=[mock_candidate_tool_call],
        prompt_feedback=None, text=None,
        usage_metadata=MockGeminiUsageMetadata(prompt_token_count=10, candidates_token_count=5, total_token_count=15)
    )
    messages: List[ChatMessage] = [{"role": "user", "content": "What's the weather like in Boston?"}]
    chat_response = await provider.chat(messages=messages, tools=tools_for_api) # stream=False is default
    assert chat_response["finish_reason"] == "tool_calls"
    assert chat_response["message"]["role"] == "assistant"
    assert chat_response["message"].get("content") is None
    assert chat_response["message"]["tool_calls"] is not None
    tool_calls_from_plugin: List[ToolCall] = chat_response["message"]["tool_calls"] # type: ignore
    assert len(tool_calls_from_plugin) == 1
    assert tool_calls_from_plugin[0]["type"] == "function"
    assert tool_calls_from_plugin[0]["function"]["name"] == "get_current_weather"
    assert json.loads(tool_calls_from_plugin[0]["function"]["arguments"]) == mock_gemini_fc_args
    assert chat_response["usage"]["total_tokens"] == 15 # type: ignore

@pytest.mark.asyncio
async def test_gemini_generate_success(gemini_provider: GeminiLLMProviderPlugin, mock_genai_lib: MagicMock):
    actual_provider = await gemini_provider
    mock_candidate = MockGeminiCandidate(MockGeminiContent([MockGeminiPart("Haiku text")], "model"), MockFinishReasonEnum(MockFinishReasonEnum.STOP))
    mock_usage = MockGeminiUsageMetadata(5,17,22)
    # Use generate_content_async for the mock
    actual_provider._model_client.generate_content_async.return_value = MockGeminiGenerateContentResponse([mock_candidate], None, None, mock_usage)
    result = await actual_provider.generate(prompt="Haiku") # stream=False is default
    assert result["text"] == "Haiku text"
    assert result["usage"]["total_tokens"] == 22 # type: ignore

@pytest.mark.asyncio
async def test_gemini_generate_api_error(gemini_provider: GeminiLLMProviderPlugin, mock_genai_lib: MagicMock, caplog: pytest.LogCaptureFixture):
    actual_provider = await gemini_provider
    caplog.set_level(logging.ERROR)
    # Use generate_content_async for the mock's side_effect
    actual_provider._model_client.generate_content_async.side_effect = RuntimeError("Simulated API failure")
    with pytest.raises(RuntimeError) as excinfo:
        await actual_provider.generate(prompt="fail") # stream=False is default
    assert "Gemini API call failed: Simulated API failure" in str(excinfo.value)

@pytest.mark.asyncio
async def test_gemini_teardown(gemini_provider: GeminiLLMProviderPlugin):
    actual_provider = await gemini_provider
    assert actual_provider._model_client is not None
    await actual_provider.teardown()
    assert actual_provider._model_client is None

# --- New Tests for Increased Coverage ---

@pytest.mark.asyncio
async def test_gemini_generate_streaming_success(gemini_provider: GeminiLLMProviderPlugin):
    provider = await gemini_provider
    prompt = "Stream a story."

    async def mock_stream_response():
        yield MockAsyncGenerateContentResponseChunk(text_delta="Once upon ")
        yield MockAsyncGenerateContentResponseChunk(text_delta="a time, ")
        yield MockAsyncGenerateContentResponseChunk(text_delta="The End.", candidates=[MockGeminiCandidate(None, MockFinishReasonEnum(MockFinishReasonEnum.STOP))], usage_metadata=MockGeminiUsageMetadata(1,3,4))

    provider._model_client.generate_content_async.return_value = mock_stream_response()

    stream_result = await provider.generate(prompt=prompt, stream=True)
    assert isinstance(stream_result, AsyncIterable)

    chunks: List[LLMCompletionChunk] = []
    async for chunk in stream_result:
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0]["text_delta"] == "Once upon "
    assert chunks[1]["text_delta"] == "a time, "
    assert chunks[2]["text_delta"] == "The End."
    assert chunks[2]["finish_reason"] == "stop"
    assert chunks[2]["usage_delta"]["total_tokens"] == 4

@pytest.mark.asyncio
async def test_gemini_chat_streaming_success(gemini_provider: GeminiLLMProviderPlugin):
    provider = await gemini_provider
    messages: List[ChatMessage] = [{"role": "user", "content": "Stream a chat response."}]

    async def mock_chat_stream():
        yield MockAsyncGenerateContentResponseChunk(candidates=[MockGeminiCandidate(MockGeminiContent([MockGeminiPart(text="Hello ")], "model"), None)])
        yield MockAsyncGenerateContentResponseChunk(candidates=[MockGeminiCandidate(MockGeminiContent([MockGeminiPart(text="there!")], "model"), None)])
        yield MockAsyncGenerateContentResponseChunk(candidates=[MockGeminiCandidate(MockGeminiContent([], "model"), MockFinishReasonEnum(MockFinishReasonEnum.STOP))], usage_metadata=MockGeminiUsageMetadata(2,2,4))

    provider._model_client.generate_content_async.return_value = mock_chat_stream()

    stream_result = await provider.chat(messages=messages, stream=True)
    assert isinstance(stream_result, AsyncIterable)

    chunks: List[LLMChatChunk] = []
    async for chunk in stream_result:
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0]["message_delta"]["content"] == "Hello "
    assert chunks[1]["message_delta"]["content"] == "there!"
    assert chunks[2]["finish_reason"] == "stop"
    assert chunks[2]["usage_delta"]["total_tokens"] == 4

@pytest.mark.asyncio
async def test_gemini_convert_messages_unsupported_role(gemini_provider: GeminiLLMProviderPlugin, caplog: pytest.LogCaptureFixture):
    provider = await gemini_provider
    caplog.set_level(logging.WARNING)
    messages: List[ChatMessage] = [{"role": "system", "content": "System prompt."}, {"role": "invalid_role", "content": "Test"}] # type: ignore
    gemini_msgs = provider._convert_messages_to_gemini(messages)
    assert len(gemini_msgs) == 1 # System becomes user, invalid_role is skipped
    assert gemini_msgs[0]["role"] == "user"
    assert "Unsupported role 'invalid_role'" in caplog.text

@pytest.mark.asyncio
async def test_gemini_convert_messages_tool_call_invalid_json_args(gemini_provider: GeminiLLMProviderPlugin, caplog: pytest.LogCaptureFixture):
    provider = await gemini_provider
    caplog.set_level(logging.ERROR)
    messages: List[ChatMessage] = [
        {"role": "assistant", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "func", "arguments": "not_json"}}]}
    ]
    gemini_msgs = provider._convert_messages_to_gemini(messages)
    assert "Could not parse tool_call args for func" in caplog.text
    # The part should still be created, but args might be empty or handled by Gemini SDK
    assert len(gemini_msgs) == 1
    assert gemini_msgs[0]["parts"][0]["function_call"]["args"] == {} # Default if parsing fails

@pytest.mark.asyncio
async def test_gemini_parse_candidate_no_content_parts(gemini_provider: GeminiLLMProviderPlugin):
    provider = await gemini_provider
    # Candidate with content but no parts array
    mock_candidate_no_parts = MockGeminiCandidate(content=MockGeminiContent(parts=[], role="model"), finish_reason=MockFinishReasonEnum(MockFinishReasonEnum.STOP))
    chat_msg, reason = provider._parse_gemini_candidate(mock_candidate_no_parts)
    assert chat_msg["role"] == "assistant"
    assert chat_msg.get("content") is None # No text parts
    assert chat_msg.get("tool_calls") is None # No function_call parts
    assert reason == "stop"

    # Candidate with no content attribute at all
    mock_candidate_no_content = MockGeminiCandidate(content=None, finish_reason=MockFinishReasonEnum(MockFinishReasonEnum.OTHER))
    chat_msg_no_content, reason_no_content = provider._parse_gemini_candidate(mock_candidate_no_content)
    assert chat_msg_no_content["role"] == "assistant"
    assert chat_msg_no_content.get("content") is None
    assert chat_msg_no_content.get("tool_calls") is None
    assert reason_no_content == "other"


@pytest.mark.asyncio
async def test_gemini_generate_blocked_response(gemini_provider: GeminiLLMProviderPlugin):
    provider = await gemini_provider
    mock_prompt_feedback = MagicMock()
    mock_prompt_feedback.block_reason = MagicMock()
    mock_prompt_feedback.block_reason.name = "SAFETY_BLOCK" # Example
    provider._model_client.generate_content_async.return_value = MockGeminiGenerateContentResponse(
        candidates=None, prompt_feedback=mock_prompt_feedback, text=None, usage_metadata=None
    )
    result = await provider.generate(prompt="risky prompt")
    assert result["text"] == "[Blocked: SAFETY_BLOCK]"
    assert result["finish_reason"] == "blocked: SAFETY_BLOCK"

@pytest.mark.asyncio
async def test_gemini_chat_blocked_response(gemini_provider: GeminiLLMProviderPlugin):
    provider = await gemini_provider
    mock_prompt_feedback = MagicMock()
    mock_prompt_feedback.block_reason = MagicMock()
    mock_prompt_feedback.block_reason.name = "SAFETY_BLOCK_CHAT"
    provider._model_client.generate_content_async.return_value = MockGeminiGenerateContentResponse(
        candidates=None, prompt_feedback=mock_prompt_feedback, text=None, usage_metadata=None
    )
    result = await provider.chat(messages=[{"role": "user", "content": "risky chat"}])
    assert result["message"]["content"] == "[Chat blocked: SAFETY_BLOCK_CHAT]"
    assert result["finish_reason"] == "blocked: SAFETY_BLOCK_CHAT"

@pytest.mark.asyncio
async def test_gemini_get_model_info_sdk_error(gemini_provider: GeminiLLMProviderPlugin, mock_genai_lib: MagicMock, caplog: pytest.LogCaptureFixture):
    provider = await gemini_provider
    caplog.set_level(logging.WARNING)
    mock_genai_lib.get_model.side_effect = RuntimeError("SDK call failed")
    info = await provider.get_model_info()
    assert "model_info_error" in info
    assert info["model_info_error"] == "SDK call failed"
    assert f"Could not retrieve detailed model info for '{provider._model_name}'" in caplog.text

@pytest.mark.asyncio
async def test_gemini_chat_tool_call_but_finish_reason_stop(gemini_provider: GeminiLLMProviderPlugin):
    provider = await gemini_provider
    mock_gemini_fc = MockGeminiFunctionCall(name="test_func", args={"arg1": "val1"})
    mock_candidate = MockGeminiCandidate(
        content=MockGeminiContent(parts=[MockGeminiPart(function_call=mock_gemini_fc)], role="model"),
        finish_reason=MockFinishReasonEnum(MockFinishReasonEnum.STOP) # Finish reason is STOP
    )
    provider._model_client.generate_content_async.return_value = MockGeminiGenerateContentResponse(
        candidates=[mock_candidate], prompt_feedback=None, text=None, usage_metadata=None
    )
    result = await provider.chat(messages=[{"role": "user", "content": "call func"}])
    # Even if finish_reason is STOP, if tool_calls are present, we map it to "tool_calls"
    assert result["finish_reason"] == "tool_calls"
    assert result["message"]["tool_calls"] is not None
    assert len(result["message"]["tool_calls"]) == 1
    assert result["message"]["tool_calls"][0]["function"]["name"] == "test_func"

@pytest.mark.asyncio
async def test_gemini_convert_messages_tool_response_non_string_dict_content(gemini_provider: GeminiLLMProviderPlugin):
    provider = await gemini_provider
    messages: List[ChatMessage] = [
        {"role": "tool", "tool_call_id": "tc1", "name": "tool_name", "content": 123} # Content is int
    ]
    gemini_msgs = provider._convert_messages_to_gemini(messages)
    assert len(gemini_msgs) == 1
    assert gemini_msgs[0]["role"] == "tool"
    assert gemini_msgs[0]["parts"][0]["function_response"]["response"] == {"output": 123}

    messages_list_content: List[ChatMessage] = [
        {"role": "tool", "tool_call_id": "tc2", "name": "tool_name_list", "content": ["item1", "item2"]} # Content is list
    ]
    gemini_msgs_list = provider._convert_messages_to_gemini(messages_list_content)
    assert len(gemini_msgs_list) == 1
    assert gemini_msgs_list[0]["parts"][0]["function_response"]["response"] == {"output": ["item1", "item2"]}

@pytest.mark.asyncio
async def test_gemini_client_not_initialized_raises_runtime_error(mock_key_provider: KeyProvider):
    provider = GeminiLLMProviderPlugin()
    # Intentionally do not call setup or ensure _model_client is None
    provider._model_client = None
    with pytest.raises(RuntimeError, match="Client not initialized"):
        await provider.generate(prompt="test")
    with pytest.raises(RuntimeError, match="Client not initialized"):
        await provider.chat(messages=[{"role": "user", "content": "test"}])

@pytest.mark.asyncio
async def test_gemini_library_not_available_runtime_check(mock_key_provider: KeyProvider):
    provider = GeminiLLMProviderPlugin()
    actual_kp = await mock_key_provider
    # Simulate genai library not being available at runtime for _execute_gemini_request
    with patch("genie_tooling.llm_providers.impl.gemini_provider.genai", None):
        # Setup might still pass if initial genai check is only at import time
        # but _execute_gemini_request should fail.
        # For this test, let's assume setup was attempted but we are testing execute path.
        provider._model_client = MagicMock() # Give it a dummy client to pass initial check in generate/chat
        with pytest.raises(RuntimeError, match="Google Generative AI library not available at runtime"):
            await provider.generate(prompt="test")
        with pytest.raises(RuntimeError, match="Google Generative AI library not available at runtime"):
            await provider.chat(messages=[{"role": "user", "content": "test"}])

@pytest.mark.asyncio
async def test_gemini_convert_messages_empty_content_user_assistant(gemini_provider: GeminiLLMProviderPlugin):
    provider = await gemini_provider
    messages: List[ChatMessage] = [
        {"role": "user", "content": ""}, # Empty content
        {"role": "assistant", "content": "  "}, # Whitespace content
        {"role": "user", "content": None} # None content
    ]
    gemini_msgs = provider._convert_messages_to_gemini(messages)
    # Empty/whitespace content should result in messages with empty parts or be skipped.
    # Current logic skips messages that result in empty parts.
    assert len(gemini_msgs) == 0

    messages_with_valid: List[ChatMessage] = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "Valid response"}
    ]
    gemini_msgs_valid = provider._convert_messages_to_gemini(messages_with_valid)
    assert len(gemini_msgs_valid) == 1
    assert gemini_msgs_valid[0]["role"] == "model"
    assert gemini_msgs_valid[0]["parts"][0]["text"] == "Valid response"

@pytest.mark.asyncio
async def test_gemini_convert_messages_assistant_content_and_tool_calls(gemini_provider: GeminiLLMProviderPlugin):
    provider = await gemini_provider
    messages: List[ChatMessage] = [
        {
            "role": "assistant",
            "content": "Thinking about calling a tool...", # Assistant provides text *and* tool_calls
            "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "func", "arguments": "{}"}}]
        }
    ]
    gemini_msgs = provider._convert_messages_to_gemini(messages)
    assert len(gemini_msgs) == 1
    assert gemini_msgs[0]["role"] == "model"
    # Gemini expects parts to be either text or function_call, not both in the same part.
    # The current conversion logic prioritizes function_call if present.
    # If content is also present, it should create a separate text part.
    # The current logic for assistant with tool_calls: if content is None, it's fine.
    # If content is NOT None, it will add a text part.
    # If tool_calls are also present, it will add function_call parts.
    # So, we expect two parts: one text, one function_call.
    assert len(gemini_msgs[0]["parts"]) == 2
    assert any(p.get("text") == "Thinking about calling a tool..." for p in gemini_msgs[0]["parts"])
    assert any(p.get("function_call", {}).get("name") == "func" for p in gemini_msgs[0]["parts"])

@pytest.mark.asyncio
async def test_gemini_convert_messages_tool_message_no_content(gemini_provider: GeminiLLMProviderPlugin, caplog: pytest.LogCaptureFixture):
    provider = await gemini_provider
    caplog.set_level(logging.DEBUG)
    messages: List[ChatMessage] = [
        {"role": "tool", "tool_call_id": "tc1", "name": "tool_name"} # No "content" field
    ]
    gemini_msgs = provider._convert_messages_to_gemini(messages)
    # A tool message without content is unusual but should be handled.
    # The current logic will create a function_response part with `{"output": None}` or similar.
    assert len(gemini_msgs) == 1
    assert gemini_msgs[0]["role"] == "tool"
    assert gemini_msgs[0]["parts"][0]["function_response"]["name"] == "tool_name"
    assert gemini_msgs[0]["parts"][0]["function_response"]["response"] == {"output": None}
    assert "Processed tool message" in caplog.text
