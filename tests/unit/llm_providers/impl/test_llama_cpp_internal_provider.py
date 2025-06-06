import logging
from typing import Any, AsyncIterable, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.llm_providers.impl.llama_cpp_internal_provider import (
    LlamaCppInternalLLMProviderPlugin,
)
from genie_tooling.llm_providers.types import (
    ChatMessage,
)
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.token_usage.manager import TokenUsageManager
from pydantic import BaseModel as PydanticBaseModel

# Mock Llama and LlamaGrammar if llama-cpp-python is not installed
try:
    from llama_cpp import Llama, LlamaGrammar
    from llama_cpp.llama_chat_format import LlamaChatCompletionHandler
    LLAMA_CPP_PYTHON_AVAILABLE_FOR_TEST = True
except ImportError:
    Llama = MagicMock(name="MockLlamaClass") # type: ignore
    LlamaGrammar = MagicMock(name="MockLlamaGrammarClass") # type: ignore
    LlamaChatCompletionHandler = MagicMock(name="MockLlamaChatCompletionHandler") # type: ignore
    LLAMA_CPP_PYTHON_AVAILABLE_FOR_TEST = False

PROVIDER_LOGGER_NAME = "genie_tooling.llm_providers.impl.llama_cpp_internal_provider"

@pytest.fixture
def mock_llama_instance() -> MagicMock:
    llama_mock = MagicMock(spec=Llama)
    llama_mock.create_completion = MagicMock(name="MockCreateCompletion")
    llama_mock.create_chat_completion = MagicMock(name="MockCreateChatCompletion")
    #llama_mock.chat_handler = MagicMock(name="MockChatHandlerMethodOnLlamaInstance", return_value=MagicMock(spec=LlamaChatCompletionHandler))
    return llama_mock

@pytest.fixture
def mock_key_provider_for_llama_internal() -> AsyncMock:
    kp = AsyncMock(spec=KeyProvider)
    kp.get_key = AsyncMock(return_value=None)
    return kp

@pytest.fixture
def mock_token_usage_manager_for_llama_internal() -> AsyncMock:
    return AsyncMock(spec=TokenUsageManager)

@pytest.fixture
async def llama_cpp_internal_provider(
    mock_llama_instance: MagicMock,
    mock_key_provider_for_llama_internal: AsyncMock,
    mock_token_usage_manager_for_llama_internal: AsyncMock,
    tmp_path
) -> LlamaCppInternalLLMProviderPlugin:
    provider = LlamaCppInternalLLMProviderPlugin()

    dummy_model_file = tmp_path / "dummy_model.gguf"
    dummy_model_file.touch()

    with patch("genie_tooling.llm_providers.impl.llama_cpp_internal_provider.Llama", return_value=mock_llama_instance), \
         patch("genie_tooling.llm_providers.impl.llama_cpp_internal_provider.LlamaGrammar") as MockLlamaGrammarConstructor: # Patch LlamaGrammar too
        # Configure the mock LlamaGrammar.from_string
        mock_grammar_instance = MagicMock(spec=LlamaGrammar)
        MockLlamaGrammarConstructor.from_string = MagicMock(return_value=mock_grammar_instance)

        await provider.setup(
            config={
                "model_path": str(dummy_model_file),
                "key_provider": mock_key_provider_for_llama_internal,
                "token_usage_manager": mock_token_usage_manager_for_llama_internal,
                "model_name_for_logging": "test_internal_model"
            }
        )
    return provider

class SimpleOutputSchemaInternal(PydanticBaseModel):
    result: str

async def consume_async_iterable(iterable: AsyncIterable[Any]) -> List[Any]:
    return [item async for item in iterable]

@pytest.mark.skipif(not LLAMA_CPP_PYTHON_AVAILABLE_FOR_TEST, reason="llama-cpp-python not installed")
@pytest.mark.asyncio
class TestLlamaCppInternalProviderSetup:
    async def test_setup_success(self, tmp_path):
        provider = LlamaCppInternalLLMProviderPlugin()
        dummy_model_file = tmp_path / "model.gguf"
        dummy_model_file.touch()

        mock_llama_constructor = MagicMock(return_value=MagicMock(spec=Llama))

        with patch("genie_tooling.llm_providers.impl.llama_cpp_internal_provider.Llama", mock_llama_constructor):
            await provider.setup(config={"model_path": str(dummy_model_file), "n_gpu_layers": 10, "chat_format": "mistral"})

        mock_llama_constructor.assert_called_once()
        call_args = mock_llama_constructor.call_args[1]
        assert call_args["model_path"] == str(dummy_model_file)
        assert call_args["n_gpu_layers"] == 10
        assert call_args["chat_format"] == "mistral"
        assert provider._chat_format == "mistral"
        assert provider._model_client is not None

    async def test_setup_model_path_missing(self, caplog):
        caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
        provider = LlamaCppInternalLLMProviderPlugin()
        await provider.setup(config={})
        assert provider._model_client is None
        assert "'model_path' not provided" in caplog.text

    async def test_setup_llama_init_fails(self, tmp_path, caplog):
        caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
        provider = LlamaCppInternalLLMProviderPlugin()
        dummy_model_file = tmp_path / "fail_model.gguf"
        dummy_model_file.touch()

        with patch("genie_tooling.llm_providers.impl.llama_cpp_internal_provider.Llama", side_effect=RuntimeError("Llama init error")):
            await provider.setup(config={"model_path": str(dummy_model_file)})

        assert provider._model_client is None
        assert "Failed to initialize Llama model: Llama init error" in caplog.text

@pytest.mark.skipif(not LLAMA_CPP_PYTHON_AVAILABLE_FOR_TEST, reason="llama-cpp-python not installed")
@pytest.mark.asyncio
class TestLlamaCppInternalProviderGenerate:
    async def test_generate_success(self, llama_cpp_internal_provider: LlamaCppInternalLLMProviderPlugin, mock_llama_instance: MagicMock):
        provider = await llama_cpp_internal_provider
        prompt = "Test prompt"
        expected_text = "Generated text from Llama."
        mock_llama_instance.create_completion.return_value = {
            "choices": [{"text": expected_text, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
        }

        result = await provider.generate(prompt=prompt, temperature=0.6)

        assert isinstance(result, dict)
        assert result["text"] == expected_text
        assert result["finish_reason"] == "stop"
        assert result["usage"]["total_tokens"] == 12 # type: ignore
        mock_llama_instance.create_completion.assert_called_once()
        call_args = mock_llama_instance.create_completion.call_args[1]
        assert call_args["prompt"] == prompt
        assert call_args["temperature"] == 0.6
        assert call_args["stream"] is False

    @patch("genie_tooling.llm_providers.impl.llama_cpp_internal_provider.generate_gbnf_grammar_from_pydantic_models")
    async def test_generate_with_gbnf(self, mock_gen_gbnf: MagicMock, llama_cpp_internal_provider: LlamaCppInternalLLMProviderPlugin, mock_llama_instance: MagicMock):
        provider = await llama_cpp_internal_provider
        mock_gen_gbnf.return_value = "root ::= test_gbnf_rule"

        # Mock LlamaGrammar.from_string to return a mock grammar object
        mock_grammar_obj = MagicMock(spec=LlamaGrammar)
        with patch("genie_tooling.llm_providers.impl.llama_cpp_internal_provider.LlamaGrammar.from_string", return_value=mock_grammar_obj) as mock_from_string:
            mock_llama_instance.create_completion.return_value = {"choices": [{"text": "structured_output"}]}
            await provider.generate(prompt="Generate structured", output_schema=SimpleOutputSchemaInternal)
            mock_from_string.assert_called_once_with("root ::= test_gbnf_rule")

        mock_gen_gbnf.assert_called_once_with([SimpleOutputSchemaInternal])
        mock_llama_instance.create_completion.assert_called_once()
        call_args = mock_llama_instance.create_completion.call_args[1]
        assert call_args["grammar"] is mock_grammar_obj # Check that the grammar object was passed


    async def test_generate_streaming_success(self, llama_cpp_internal_provider: LlamaCppInternalLLMProviderPlugin, mock_llama_instance: MagicMock):
        provider = await llama_cpp_internal_provider

        def mock_stream_response_gen(*args, **kwargs):
            yield {"choices": [{"text": "Once ", "finish_reason": None}], "usage": None}
            yield {"choices": [{"text": "upon ", "finish_reason": None}], "usage": None}
            yield {"choices": [{"text": "a time.", "finish_reason": "stop"}], "usage": {"prompt_tokens":1, "completion_tokens":3, "total_tokens":4}}

        mock_llama_instance.create_completion.return_value = mock_stream_response_gen()

        result_stream = await provider.generate(prompt="Stream story", stream=True)
        chunks = await consume_async_iterable(result_stream)

        assert len(chunks) == 3
        assert chunks[0]["text_delta"] == "Once "
        assert chunks[1]["text_delta"] == "upon "
        assert chunks[2]["text_delta"] == "a time."
        assert chunks[2]["finish_reason"] == "stop"
        assert chunks[2]["usage_delta"]["total_tokens"] == 4 # type: ignore
        provider._token_usage_manager.record_usage.assert_awaited_once() # type: ignore

@pytest.mark.skipif(not LLAMA_CPP_PYTHON_AVAILABLE_FOR_TEST, reason="llama-cpp-python not installed")
@pytest.mark.asyncio
class TestLlamaCppInternalProviderChat:
    async def test_chat_success(self, llama_cpp_internal_provider: LlamaCppInternalLLMProviderPlugin, mock_llama_instance: MagicMock):
        provider = await llama_cpp_internal_provider
        messages: List[ChatMessage] = [{"role": "user", "content": "Hi Llama"}]
        expected_content = "Hello from internal Llama!"
        mock_llama_instance.create_chat_completion.return_value = {
            "choices": [{"message": {"role": "assistant", "content": expected_content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 6, "total_tokens": 9}
        }

        result = await provider.chat(messages=messages)

        assert isinstance(result, dict)
        assert result["message"]["role"] == "assistant"
        assert result["message"]["content"] == expected_content
        assert result["finish_reason"] == "stop"
        assert result["usage"]["total_tokens"] == 9 # type: ignore
        mock_llama_instance.create_chat_completion.assert_called_once()
        call_args = mock_llama_instance.create_chat_completion.call_args[1]
        assert call_args["messages"] == messages
        assert call_args["stream"] is False

    # In tests/unit/llm_providers/impl/test_llama_cpp_internal_provider.py
    # Modify TestLlamaCppInternalProviderSetup::test_setup_success
    # or add a new test for chat_format initialization.

    async def test_setup_with_chat_format_config(self, tmp_path):
        provider = LlamaCppInternalLLMProviderPlugin()
        dummy_model_file = tmp_path / "model.gguf"
        dummy_model_file.touch()

        mock_llama_constructor = MagicMock(return_value=MagicMock(spec=Llama))

        with patch("genie_tooling.llm_providers.impl.llama_cpp_internal_provider.Llama", mock_llama_constructor):
            await provider.setup(config={
                "model_path": str(dummy_model_file),
                "chat_format": "mistral-instruct" # A valid format
            })

        mock_llama_constructor.assert_called_once()
        # Check that 'chat_format' was passed to Llama constructor
        call_kwargs = mock_llama_constructor.call_args[1]
        assert call_kwargs["chat_format"] == "mistral-instruct"
        assert provider._chat_format == "mistral-instruct"
        assert provider._model_client is not None

    async def test_chat_streaming_success(self, llama_cpp_internal_provider: LlamaCppInternalLLMProviderPlugin, mock_llama_instance: MagicMock):
        provider = await llama_cpp_internal_provider

        def mock_stream_response_chat(*args, **kwargs):
            yield {"choices": [{"delta": {"role": "assistant", "content": "Streaming "}}], "usage": None}
            yield {"choices": [{"delta": {"content": "chat..."}}], "usage": None}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens":2, "completion_tokens":2, "total_tokens":4}}

        mock_llama_instance.create_chat_completion.return_value = mock_stream_response_chat()

        result_stream = await provider.chat(messages=[{"role":"user", "content":"Stream chat"}], stream=True)
        chunks = await consume_async_iterable(result_stream)

        assert len(chunks) == 3
        assert chunks[0]["message_delta"]["content"] == "Streaming " # type: ignore
        assert chunks[1]["message_delta"]["content"] == "chat..." # type: ignore
        assert chunks[2]["finish_reason"] == "stop"
        assert chunks[2]["usage_delta"]["total_tokens"] == 4 # type: ignore
        provider._token_usage_manager.record_usage.assert_awaited_once() # type: ignore

@pytest.mark.skipif(not LLAMA_CPP_PYTHON_AVAILABLE_FOR_TEST, reason="llama-cpp-python not installed")
@pytest.mark.asyncio
class TestLlamaCppInternalProviderErrorsAndInfo:
    async def test_client_not_initialized_raises_error(self):
        provider = LlamaCppInternalLLMProviderPlugin()
        provider._model_client = None
        with pytest.raises(RuntimeError, match="Model client not initialized"):
            await provider.generate(prompt="test")
        with pytest.raises(RuntimeError, match="Model client not initialized"):
            await provider.chat(messages=[])

    async def test_get_model_info(self, llama_cpp_internal_provider: LlamaCppInternalLLMProviderPlugin):
        provider = await llama_cpp_internal_provider
        info = await provider.get_model_info()
        assert info["provider"] == "llama.cpp (internal)"
        assert info["model_path"] == provider._model_path
        assert info["n_ctx"] == provider._n_ctx
        assert info["n_gpu_layers"] == provider._n_gpu_layers

    async def test_teardown(self, llama_cpp_internal_provider: LlamaCppInternalLLMProviderPlugin, mock_llama_instance: MagicMock):
        provider = await llama_cpp_internal_provider
        assert provider._model_client is mock_llama_instance

        await provider.teardown()
        assert provider._model_client is None
