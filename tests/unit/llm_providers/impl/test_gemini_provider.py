import json
import logging
from typing import Any, Dict, List, NamedTuple, Optional
from unittest.mock import AsyncMock, MagicMock, patch  # ADDED AsyncMock

import pytest
from genie_tooling.llm_providers.impl.gemini_provider import GeminiLLMProviderPlugin
from genie_tooling.llm_providers.types import ChatMessage, ToolCall
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

@pytest.fixture
def mock_genai_lib():
    mock_lib = MagicMock(name="MockGoogleGenerativeAIModule")
    mock_lib.configure = MagicMock()
    mock_model_inst = MagicMock(name="MockGenerativeModelInstance")
    mock_model_inst.generate_content = MagicMock(return_value=MockGeminiGenerateContentResponse([], None, None, None))
    mock_lib.GenerativeModel = MagicMock(return_value=mock_model_inst)
    mock_lib.types = MagicMock()
    mock_lib.types.GenerationConfig = MagicMock()
    mock_lib.types.ContentDict = dict
    mock_lib.types.Tool = MagicMock()
    mock_lib.types.FunctionDeclaration = MagicMock()
    mock_lib.types.generation_types = MagicMock()
    mock_lib.types.generation_types.GenerateContentResponse = MockGeminiGenerateContentResponse
    mock_lib.types.generation_types.Candidate = MockGeminiCandidate
    with patch("genie_tooling.llm_providers.impl.gemini_provider.genai", mock_lib) as p_genai, \
         patch("genie_tooling.llm_providers.impl.gemini_provider._GenerativeModelTypePlaceholder", new=mock_lib.GenerativeModel), \
         patch("genie_tooling.llm_providers.impl.gemini_provider._GenerationConfigType", new=mock_lib.types.GenerationConfig), \
         patch("genie_tooling.llm_providers.impl.gemini_provider._ContentDictType", new=dict), \
         patch("genie_tooling.llm_providers.impl.gemini_provider._GeminiSDKToolType", new=mock_lib.types.Tool), \
         patch("genie_tooling.llm_providers.impl.gemini_provider._FunctionDeclarationType", new=mock_lib.types.FunctionDeclaration), \
         patch("genie_tooling.llm_providers.impl.gemini_provider._GenerateContentResponseType", new=mock_lib.types.generation_types.GenerateContentResponse), \
         patch("genie_tooling.llm_providers.impl.gemini_provider._CandidateType", new=mock_lib.types.generation_types.Candidate):
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
    # FIXED: Use AsyncMock for mocking the method
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
    provider._model_client.generate_content.return_value = MockGeminiGenerateContentResponse(
        candidates=[mock_candidate_tool_call],
        prompt_feedback=None, text=None,
        usage_metadata=MockGeminiUsageMetadata(prompt_token_count=10, candidates_token_count=5, total_token_count=15)
    )
    messages: List[ChatMessage] = [{"role": "user", "content": "What's the weather like in Boston?"}]
    chat_response = await provider.chat(messages=messages, tools=tools_for_api)
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
    actual_provider._model_client.generate_content.return_value = MockGeminiGenerateContentResponse([mock_candidate], None, None, mock_usage)
    result = await actual_provider.generate(prompt="Haiku")
    assert result["text"] == "Haiku text"
    assert result["usage"]["total_tokens"] == 22 # type: ignore

@pytest.mark.asyncio
async def test_gemini_generate_api_error(gemini_provider: GeminiLLMProviderPlugin, mock_genai_lib: MagicMock, caplog: pytest.LogCaptureFixture):
    actual_provider = await gemini_provider
    caplog.set_level(logging.ERROR)
    actual_provider._model_client.generate_content.side_effect = RuntimeError("Simulated API failure")
    with pytest.raises(RuntimeError) as excinfo:
        await actual_provider.generate(prompt="fail")
    assert "Gemini API call failed: Simulated API failure" in str(excinfo.value)

@pytest.mark.asyncio
async def test_gemini_teardown(gemini_provider: GeminiLLMProviderPlugin):
    actual_provider = await gemini_provider
    assert actual_provider._model_client is not None
    await actual_provider.teardown()
    assert actual_provider._model_client is None
