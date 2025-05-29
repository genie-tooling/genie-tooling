import logging
from typing import Any, Dict, List, NamedTuple, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.llm_providers.impl.gemini_provider import GeminiLLMProviderPlugin
from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.security.key_provider import KeyProvider

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
    mock_lib = MagicMock(name="MockGoogleGenerativeAIModule"); mock_lib.configure = MagicMock(); mock_lib.get_model = MagicMock()
    mock_model_inst = MagicMock(name="MockGenerativeModelInstance"); mock_model_inst.generate_content = MagicMock(return_value=MockGeminiGenerateContentResponse([], None, None, None))
    mock_lib.GenerativeModel = MagicMock(return_value=mock_model_inst); mock_lib.types = MagicMock(); mock_lib.types.GenerationConfig = MagicMock()
    mock_lib.types.ContentDict = dict; mock_lib.types.Tool = MagicMock(); mock_lib.types.FunctionDeclaration = MagicMock(); mock_lib.types.generation_types = MagicMock()
    mock_lib.types.generation_types.GenerateContentResponse = MockGeminiGenerateContentResponse; mock_lib.types.generation_types.Candidate = MockGeminiCandidate
    with patch("genie_tooling.llm_providers.impl.gemini_provider.genai", mock_lib) as p_genai, \
         patch("genie_tooling.llm_providers.impl.gemini_provider._GenerativeModelTypePlaceholder", new=mock_lib.GenerativeModel), \
         patch("genie_tooling.llm_providers.impl.gemini_provider._GenerationConfigType", new=mock_lib.types.GenerationConfig), \
         patch("genie_tooling.llm_providers.impl.gemini_provider._CandidateType", new=mock_lib.types.generation_types.Candidate):
        yield p_genai

@pytest.fixture
async def gemini_provider(mock_genai_lib: MagicMock, mock_key_provider: KeyProvider) -> GeminiLLMProviderPlugin:
    provider = GeminiLLMProviderPlugin()
    actual_mock_key_provider = await mock_key_provider # Await the async fixture
    await provider.setup(config={"model_name": "test-gemini-model", "key_provider": actual_mock_key_provider})
    assert provider._model_client is mock_genai_lib.GenerativeModel.return_value
    return provider

@pytest.mark.asyncio
async def test_gemini_setup_success(mock_genai_lib: MagicMock, mock_key_provider: KeyProvider):
    provider = GeminiLLMProviderPlugin(); actual_kp = await mock_key_provider
    api_key_name_for_setup = "MY_TEST_GOOGLE_KEY"; original_get_key = actual_kp.get_key
    async def specific_get_key(key_name_req: str) -> Optional[str]:
        if key_name_req == api_key_name_for_setup: return "google_key_for_setup_test"
        return await original_get_key(key_name_req)
    actual_kp.get_key = specific_get_key
    model_name="gemini-pro-test"; system_instruction="You are a test bot."; safety_settings=[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT", "threshold":"BLOCK_ONLY_HIGH"}]
    await provider.setup(config={"api_key_name": api_key_name_for_setup, "model_name": model_name, "system_instruction": system_instruction, "safety_settings": safety_settings, "key_provider": actual_kp})
    mock_genai_lib.configure.assert_called_once_with(api_key="google_key_for_setup_test")
    mock_genai_lib.GenerativeModel.assert_called_once_with(model_name=model_name, system_instruction=system_instruction, safety_settings=safety_settings)
    assert provider._model_client is not None; actual_kp.get_key = original_get_key

@pytest.mark.asyncio
async def test_gemini_setup_no_api_key(mock_genai_lib: MagicMock, caplog: pytest.LogCaptureFixture, mock_key_provider: KeyProvider):
    caplog.set_level(logging.ERROR); provider = GeminiLLMProviderPlugin(); actual_kp = await mock_key_provider
    original_get_key = actual_kp.get_key
    async def get_key_returns_none(key_name_req: str) -> Optional[str]:
        if key_name_req == "NON_EXISTENT_KEY": return None
        return await original_get_key(key_name_req)
    actual_kp.get_key = get_key_returns_none
    await provider.setup(config={"api_key_name": "NON_EXISTENT_KEY", "key_provider": actual_kp})
    assert provider._model_client is None; assert "API key 'NON_EXISTENT_KEY' not found via KeyProvider." in caplog.text
    actual_kp.get_key = original_get_key

@pytest.mark.asyncio
async def test_gemini_convert_messages(gemini_provider: GeminiLLMProviderPlugin):
    actual_provider = await gemini_provider
    messages: List[ChatMessage] = [{"role": "system", "content": "Sys prompt."}, {"role": "user", "content": "Hello"}, {"role": "assistant", "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "London"}'}}]}, {"role": "tool", "tool_call_id": "tc1", "name": "get_weather", "content": '{"temp": 15}'}]
    gemini_msgs = actual_provider._convert_messages_to_gemini(messages)
    assert len(gemini_msgs) == 4; assert gemini_msgs[2]["parts"][0]["function_call"]["name"] == "get_weather" # type: ignore
    assert gemini_msgs[3]["parts"][0]["function_response"]["name"] == "get_weather" # type: ignore
    assert gemini_msgs[3]["parts"][0]["function_response"]["response"] == {"temp": 15} # type: ignore

@pytest.mark.asyncio
async def test_gemini_parse_candidate_text_only(gemini_provider: GeminiLLMProviderPlugin):
    actual_provider = await gemini_provider
    mock_candidate = MockGeminiCandidate(MockGeminiContent([MockGeminiPart(text="Resp.")], "model"), MockFinishReasonEnum(MockFinishReasonEnum.STOP))
    chat_msg, finish_reason = actual_provider._parse_gemini_candidate(mock_candidate) # type: ignore
    assert chat_msg.get("tool_calls") is None; assert finish_reason == "stop"

@pytest.mark.asyncio
async def test_gemini_parse_candidate_with_tool_calls(gemini_provider: GeminiLLMProviderPlugin):
    actual_provider = await gemini_provider
    mock_candidate = MockGeminiCandidate(MockGeminiContent([MockGeminiPart(function_call=MockGeminiFunctionCall("tool1", {"p": "v"}))], "model"), MockFinishReasonEnum(MockFinishReasonEnum.TOOL_CODE))
    chat_msg, finish_reason = actual_provider._parse_gemini_candidate(mock_candidate) # type: ignore
    assert chat_msg.get("tool_calls") is not None; assert len(chat_msg["tool_calls"]) == 1 # type: ignore
    assert finish_reason == "tool_calls"

@pytest.mark.asyncio
async def test_gemini_generate_success(gemini_provider: GeminiLLMProviderPlugin, mock_genai_lib: MagicMock):
    actual_provider = await gemini_provider
    mock_candidate = MockGeminiCandidate(MockGeminiContent([MockGeminiPart("Haiku text")], "model"), MockFinishReasonEnum(MockFinishReasonEnum.STOP))
    mock_usage = MockGeminiUsageMetadata(5,17,22)
    actual_provider._model_client.generate_content.return_value = MockGeminiGenerateContentResponse([mock_candidate], None, None, mock_usage) # type: ignore
    result = await actual_provider.generate(prompt="Haiku")
    assert result["text"] == "Haiku text"; assert result["usage"]["total_tokens"] == 22 # type: ignore

@pytest.mark.asyncio
async def test_gemini_chat_with_tool_call_flow(gemini_provider: GeminiLLMProviderPlugin, mock_genai_lib: MagicMock):
    actual_provider = await gemini_provider
    mock_fc = MockGeminiFunctionCall("get_weather", {"city": "Paris"})
    mock_cand1 = MockGeminiCandidate(MockGeminiContent([MockGeminiPart(function_call=mock_fc)], "model"), MockFinishReasonEnum(MockFinishReasonEnum.TOOL_CODE))
    actual_provider._model_client.generate_content.return_value = MockGeminiGenerateContentResponse([mock_cand1], None, None, MockGeminiUsageMetadata(10,5,15)) # type: ignore
    resp1 = await actual_provider.chat(messages=[{"role":"user", "content":"Weather?"}])
    assert resp1["finish_reason"] == "tool_calls"
    tool_call_id = resp1["message"]["tool_calls"][0]["id"] # type: ignore
    messages_with_tool_resp = [{"role":"user", "content":"Weather?"}, resp1["message"], {"role":"tool", "tool_call_id": tool_call_id, "name": "get_weather", "content": '{"temp":"20C"}'}]
    mock_cand2 = MockGeminiCandidate(MockGeminiContent([MockGeminiPart("Paris: 20C")], "model"), MockFinishReasonEnum(MockFinishReasonEnum.STOP))
    actual_provider._model_client.generate_content.return_value = MockGeminiGenerateContentResponse([mock_cand2], None, None, MockGeminiUsageMetadata(20,10,30)) # type: ignore
    resp2 = await actual_provider.chat(messages=messages_with_tool_resp)
    assert resp2["message"]["content"] == "Paris: 20C"; assert resp2["finish_reason"] == "stop"

@pytest.mark.asyncio
async def test_gemini_generate_api_error(gemini_provider: GeminiLLMProviderPlugin, mock_genai_lib: MagicMock, caplog: pytest.LogCaptureFixture):
    actual_provider = await gemini_provider
    caplog.set_level(logging.ERROR)
    actual_provider._model_client.generate_content.side_effect = RuntimeError("Simulated API failure") # type: ignore
    with pytest.raises(RuntimeError) as excinfo:
        await actual_provider.generate(prompt="fail")
    assert "Gemini API call failed: Simulated API failure" in str(excinfo.value)

@pytest.mark.asyncio
async def test_gemini_teardown(gemini_provider: GeminiLLMProviderPlugin):
    actual_provider = await gemini_provider
    assert actual_provider._model_client is not None
    await actual_provider.teardown()
    assert actual_provider._model_client is None
