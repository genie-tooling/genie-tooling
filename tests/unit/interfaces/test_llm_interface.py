### tests/unit/interfaces/test_llm_interface.py
import asyncio
from typing import Any, AsyncIterable, Dict, List, Optional, cast
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from genie_tooling.guardrails.manager import GuardrailManager
from genie_tooling.guardrails.types import GuardrailViolation
from genie_tooling.interfaces import LLMInterface
from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.llm_providers.manager import LLMProviderManager
from genie_tooling.llm_providers.types import (
    ChatMessage,
    LLMChatChunk,
    LLMChatResponse,
    LLMChatChunkDeltaMessage,
    LLMCompletionChunk,
    LLMCompletionResponse,
    LLMUsageInfo,
)
from genie_tooling.observability.manager import InteractionTracingManager
from genie_tooling.prompts.llm_output_parsers.abc import LLMOutputParserPlugin
from genie_tooling.prompts.llm_output_parsers.manager import LLMOutputParserManager
from genie_tooling.token_usage.manager import TokenUsageManager


@pytest.fixture
def mock_llm_provider_manager() -> MagicMock:
    """Mocks the LLMProviderManager."""
    return MagicMock(spec=LLMProviderManager)

@pytest.fixture
def mock_llm_output_parser_manager() -> MagicMock:
    """Mocks the LLMOutputParserManager."""
    mgr = MagicMock(spec=LLMOutputParserManager)
    mgr.parse = AsyncMock(return_value={"parsed": "data_from_parser_manager_mock"}) # Ensure parse is AsyncMock
    return mgr

@pytest.fixture
def mock_tracing_manager_for_llm_if() -> AsyncMock: # Changed to AsyncMock
    """Mocks the InteractionTracingManager for LLMInterface tests."""
    mgr = AsyncMock(spec=InteractionTracingManager) # Use AsyncMock for the manager
    mgr.trace_event = AsyncMock()
    return mgr

@pytest.fixture
def mock_guardrail_manager_for_llm_if() -> AsyncMock: # Changed to AsyncMock
    """Mocks the GuardrailManager for LLMInterface tests."""
    mgr = AsyncMock(spec=GuardrailManager) # Use AsyncMock for the manager
    mgr.check_input_guardrails = AsyncMock(return_value=GuardrailViolation(action="allow", reason=""))
    mgr.check_output_guardrails = AsyncMock(return_value=GuardrailViolation(action="allow", reason=""))
    return mgr

@pytest.fixture
def mock_token_usage_manager_for_llm_if() -> MagicMock:
    """Mocks the TokenUsageManager for LLMInterface tests."""
    mgr = MagicMock(spec=TokenUsageManager)
    mgr.record_usage = AsyncMock()
    return mgr

@pytest.fixture
def llm_interface(
    mock_llm_provider_manager: MagicMock,
    mock_llm_output_parser_manager: MagicMock,
    mock_tracing_manager_for_llm_if: AsyncMock, # Updated type hint
    mock_guardrail_manager_for_llm_if: AsyncMock, # Updated type hint
    mock_token_usage_manager_for_llm_if: MagicMock,
) -> LLMInterface:
    """Provides an LLMInterface instance with mocked dependencies."""
    return LLMInterface(
        llm_provider_manager=mock_llm_provider_manager,
        default_provider_id="default_llm_test_id",
        output_parser_manager=mock_llm_output_parser_manager,
        tracing_manager=mock_tracing_manager_for_llm_if,
        guardrail_manager=mock_guardrail_manager_for_llm_if,
        token_usage_manager=mock_token_usage_manager_for_llm_if,
    )

@pytest.fixture
def mock_llm_provider_plugin() -> AsyncMock:
    """Mocks an LLMProviderPlugin instance."""
    plugin = AsyncMock(spec=LLMProviderPlugin)
    plugin.plugin_id = "test_llm_plugin_v1"
    # Ensure _model_name attribute exists for token usage recording
    plugin._model_name = "test_model_from_plugin"
    # Default generate response
    plugin.generate = AsyncMock(return_value=LLMCompletionResponse(
        text="Generated text", finish_reason="stop", usage={"total_tokens": 10}, raw_response={}
    ))
    # Default chat response
    plugin.chat = AsyncMock(return_value=LLMChatResponse(
        message=ChatMessage(role="assistant", content="Chat response"),
        finish_reason="stop", usage={"total_tokens": 20}, raw_response={}
    ))
    return plugin


@pytest.mark.asyncio
class TestLLMInterfaceGenerate:
    """Tests for LLMInterface.generate() method."""

    async def test_generate_success_default_provider(
        self, llm_interface: LLMInterface, mock_llm_provider_manager: MagicMock, mock_llm_provider_plugin: AsyncMock
    ):
        """Test successful generation using the default provider."""
        mock_llm_provider_manager.get_llm_provider = AsyncMock(return_value=mock_llm_provider_plugin)
        prompt = "Test prompt"
        response = await llm_interface.generate(prompt)

        assert response["text"] == "Generated text"
        mock_llm_provider_manager.get_llm_provider.assert_awaited_once_with("default_llm_test_id")
        mock_llm_provider_plugin.generate.assert_awaited_once_with(prompt, stream=False)
        llm_interface._tracing_manager.trace_event.assert_any_call("llm.generate.start", ANY, "LLMInterface", ANY) # type: ignore
        llm_interface._tracing_manager.trace_event.assert_any_call("llm.generate.success", ANY, "LLMInterface", ANY) # type: ignore
        llm_interface._token_usage_manager.record_usage.assert_awaited_once() # type: ignore

    async def test_generate_success_specific_provider_and_kwargs(
        self, llm_interface: LLMInterface, mock_llm_provider_manager: MagicMock, mock_llm_provider_plugin: AsyncMock
    ):
        """Test successful generation with a specific provider and kwargs."""
        mock_llm_provider_manager.get_llm_provider = AsyncMock(return_value=mock_llm_provider_plugin)
        await llm_interface.generate("Another prompt", provider_id="custom_provider", temperature=0.5, model="override_model")

        mock_llm_provider_manager.get_llm_provider.assert_awaited_once_with("custom_provider")
        mock_llm_provider_plugin.generate.assert_awaited_once_with("Another prompt", stream=False, temperature=0.5, model="override_model")
        # Check token usage recorded with overridden model name
        usage_call_args = llm_interface._token_usage_manager.record_usage.call_args[0][0] # type: ignore
        assert usage_call_args["model_name"] == "override_model"


    async def test_generate_streaming(
        self, llm_interface: LLMInterface, mock_llm_provider_manager: MagicMock, mock_llm_provider_plugin: AsyncMock
    ):
        """Test successful streaming generation."""
        async def mock_stream():
            yield LLMCompletionChunk(text_delta="Hello ")
            yield LLMCompletionChunk(text_delta="World", finish_reason="stop", usage_delta={"total_tokens": 2})
        mock_llm_provider_plugin.generate = AsyncMock(return_value=mock_stream())
        mock_llm_provider_manager.get_llm_provider = AsyncMock(return_value=mock_llm_provider_plugin)

        result_stream = await llm_interface.generate("Test stream", stream=True)
        chunks = [chunk async for chunk in result_stream] # type: ignore

        assert len(chunks) == 2
        assert chunks[0]["text_delta"] == "Hello "
        assert chunks[1]["text_delta"] == "World"
        assert chunks[1]["finish_reason"] == "stop"
        assert chunks[1]["usage_delta"]["total_tokens"] == 2 # type: ignore
        llm_interface._tracing_manager.trace_event.assert_any_call("llm.generate.stream_end", ANY, "LLMInterface", ANY) # type: ignore
        llm_interface._token_usage_manager.record_usage.assert_awaited_once() # type: ignore

    async def test_generate_provider_not_found(self, llm_interface: LLMInterface, mock_llm_provider_manager: MagicMock):
        """Test error when the specified LLM provider is not found."""
        mock_llm_provider_manager.get_llm_provider = AsyncMock(return_value=None)
        with pytest.raises(RuntimeError, match="LLM Provider 'default_llm_test_id' not found or failed to load."):
            await llm_interface.generate("Test")
        llm_interface._tracing_manager.trace_event.assert_any_call("llm.generate.error", ANY, "LLMInterface", ANY) # type: ignore

    async def test_generate_input_guardrail_block(
        self, llm_interface: LLMInterface, mock_guardrail_manager_for_llm_if: AsyncMock # Updated type hint
    ):
        """Test input guardrail blocking generation."""
        mock_guardrail_manager_for_llm_if.check_input_guardrails = AsyncMock(
            return_value=GuardrailViolation(action="block", reason="Blocked by input guardrail")
        )
        with pytest.raises(PermissionError, match="LLM generate blocked by input guardrail: Blocked by input guardrail"):
            await llm_interface.generate("Risky prompt")
        llm_interface._tracing_manager.trace_event.assert_any_call("llm.generate.blocked_by_input_guardrail", ANY, "LLMInterface", ANY) # type: ignore

    async def test_generate_output_guardrail_block(
        self, llm_interface: LLMInterface, mock_llm_provider_manager: MagicMock, mock_llm_provider_plugin: AsyncMock, mock_guardrail_manager_for_llm_if: AsyncMock # Updated type hint
    ):
        """Test output guardrail blocking generation response."""
        mock_llm_provider_manager.get_llm_provider = AsyncMock(return_value=mock_llm_provider_plugin)
        mock_guardrail_manager_for_llm_if.check_output_guardrails = AsyncMock(
            return_value=GuardrailViolation(action="block", reason="Blocked by output guardrail")
        )
        response = await llm_interface.generate("Generate risky output")
        assert response["text"] == "[RESPONSE BLOCKED: Blocked by output guardrail]"
        assert response["finish_reason"] == "blocked_by_guardrail"
        llm_interface._tracing_manager.trace_event.assert_any_call("llm.generate.blocked_by_output_guardrail", ANY, "LLMInterface", ANY) # type: ignore

    async def test_generate_no_default_provider_id_error(self, llm_interface: LLMInterface):
        """Test error when no default provider ID is set and none is specified."""
        llm_interface._default_provider_id = None
        with pytest.raises(ValueError, match="No LLM provider ID specified and no default is set for generate."):
            await llm_interface.generate("Test prompt")


@pytest.mark.asyncio
class TestLLMInterfaceChat:
    """Tests for LLMInterface.chat() method."""

    async def test_chat_success(
        self, llm_interface: LLMInterface, mock_llm_provider_manager: MagicMock, mock_llm_provider_plugin: AsyncMock
    ):
        """Test successful chat completion."""
        mock_llm_provider_manager.get_llm_provider = AsyncMock(return_value=mock_llm_provider_plugin)
        messages: List[ChatMessage] = [{"role": "user", "content": "Hello"}]
        response = await llm_interface.chat(messages)

        assert response["message"]["content"] == "Chat response"
        mock_llm_provider_plugin.chat.assert_awaited_once_with(messages, stream=False)
        llm_interface._token_usage_manager.record_usage.assert_awaited_once() # type: ignore

    async def test_chat_streaming(
        self, llm_interface: LLMInterface, mock_llm_provider_manager: MagicMock, mock_llm_provider_plugin: AsyncMock
    ):
        """Test successful streaming chat completion."""
        async def mock_chat_stream():
            yield LLMChatChunk(message_delta=LLMChatChunkDeltaMessage(role="assistant", content="First part. "))
            yield LLMChatChunk(message_delta=LLMChatChunkDeltaMessage(content="Second part."), finish_reason="stop", usage_delta={"total_tokens": 5})
        mock_llm_provider_plugin.chat = AsyncMock(return_value=mock_chat_stream())
        mock_llm_provider_manager.get_llm_provider = AsyncMock(return_value=mock_llm_provider_plugin)

        result_stream = await llm_interface.chat([{"role": "user", "content": "Stream chat"}], stream=True)
        chunks = [chunk async for chunk in result_stream] # type: ignore

        assert len(chunks) == 2
        assert chunks[0]["message_delta"]["content"] == "First part. " # type: ignore
        assert chunks[1]["message_delta"]["content"] == "Second part." # type: ignore
        assert chunks[1]["finish_reason"] == "stop"
        assert chunks[1]["usage_delta"]["total_tokens"] == 5 # type: ignore
        llm_interface._token_usage_manager.record_usage.assert_awaited_once() # type: ignore

    async def test_chat_input_guardrail_block_with_message_list(
        self, llm_interface: LLMInterface, mock_guardrail_manager_for_llm_if: AsyncMock # Updated type hint
    ):
        """Test input guardrail blocking chat based on message content."""
        mock_guardrail_manager_for_llm_if.check_input_guardrails = AsyncMock(
            return_value=GuardrailViolation(action="block", reason="Blocked chat message")
        )
        messages: List[ChatMessage] = [{"role": "user", "content": "Risky chat message"}]
        with pytest.raises(PermissionError, match="LLM chat blocked by input guardrail: Blocked chat message"):
            await llm_interface.chat(messages)
        # Guardrail check should receive the last message
        mock_guardrail_manager_for_llm_if.check_input_guardrails.assert_awaited_with(messages[-1], ANY)

    async def test_chat_no_default_provider_id_error(self, llm_interface: LLMInterface):
        """Test error when no default provider ID is set and none is specified for chat."""
        llm_interface._default_provider_id = None
        with pytest.raises(ValueError, match="No LLM provider ID specified and no default is set for chat."):
            await llm_interface.chat([{"role": "user", "content": "Test"}])


@pytest.mark.asyncio
class TestLLMInterfaceParseOutput:
    """Tests for LLMInterface.parse_output() method."""

    async def test_parse_output_success(
        self, llm_interface: LLMInterface, mock_llm_output_parser_manager: MagicMock
    ):
        """Test successful parsing of LLM output."""
        # mock_parser_plugin = AsyncMock(spec=LLMOutputParserPlugin) # Not needed directly
        # mock_parser_plugin.parse = MagicMock(return_value={"parsed": True}) # parse is sync
        # Configure the manager's parse method to return the desired value
        # This was already done in the fixture for mock_llm_output_parser_manager
        # mock_llm_output_parser_manager.parse = AsyncMock(return_value={"parsed": True})

        response = LLMCompletionResponse(text='{"key":"value"}', finish_reason="stop", usage=None, raw_response={})
        parsed = await llm_interface.parse_output(response, parser_id="test_parser", schema={})

        assert parsed == {"parsed": "data_from_parser_manager_mock"} # Updated expected value
        mock_llm_output_parser_manager.parse.assert_awaited_once_with('{"key":"value"}', "test_parser", {})
        llm_interface._tracing_manager.trace_event.assert_any_call("llm.parse_output.success", ANY, "LLMInterface", ANY) # type: ignore

    async def test_parse_output_no_text_in_response(self, llm_interface: LLMInterface):
        """Test error when LLM response has no text content to parse."""
        # Test with LLMChatResponse missing content
        chat_response_no_content = LLMChatResponse(message=ChatMessage(role="assistant", content=None), finish_reason="stop", usage=None, raw_response={})
        with pytest.raises(ValueError, match="No text content found in LLM response to parse."):
            await llm_interface.parse_output(chat_response_no_content)
        llm_interface._tracing_manager.trace_event.assert_any_call("llm.parse_output.error", ANY, "LLMInterface", ANY) # type: ignore

    async def test_parse_output_parser_manager_fails(
        self, llm_interface: LLMInterface, mock_llm_output_parser_manager: MagicMock
    ):
        """Test error when the output parser manager itself fails."""
        mock_llm_output_parser_manager.parse = AsyncMock(side_effect=ValueError("Parser manager boom"))
        response = LLMCompletionResponse(text="data", finish_reason="stop", usage=None, raw_response={})
        with pytest.raises(ValueError, match="Parser manager boom"):
            await llm_interface.parse_output(response)
        llm_interface._tracing_manager.trace_event.assert_any_call("llm.parse_output.error", ANY, "LLMInterface", ANY) # type: ignore