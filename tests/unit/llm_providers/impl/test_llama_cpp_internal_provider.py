### tests/unit/llm_providers/impl/test_llama_cpp_internal_provider.py
import logging
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.llm_providers.impl.llama_cpp_internal_provider import (
    LLAMA_CPP_PYTHON_AVAILABLE,
    LlamaCppInternalLLMProviderPlugin,
)
from genie_tooling.llm_providers.types import (
    ChatMessage,
    LLMChatChunk,
    LLMCompletionChunk,
)
from genie_tooling.token_usage.manager import TokenUsageManager

# Mock Llama and LlamaGrammar if not available
if not LLAMA_CPP_PYTHON_AVAILABLE:
    LlamaMock = MagicMock(name="MockLlamaClass")
    LlamaGrammarMock = MagicMock(name="MockLlamaGrammarClass")
    LlamaChatCompletionHandlerMock = MagicMock(name="MockLlamaChatCompletionHandlerClass")
else:
    from llama_cpp import Llama, LlamaGrammar  # type: ignore
    from llama_cpp.llama_chat_format import LlamaChatCompletionHandler  # type: ignore
    LlamaMock = Llama
    LlamaGrammarMock = LlamaGrammar
    LlamaChatCompletionHandlerMock = LlamaChatCompletionHandler


PROVIDER_LOGGER_NAME = "genie_tooling.llm_providers.impl.llama_cpp_internal_provider"


@pytest.fixture()
def mock_llama_instance() -> MagicMock:
    instance = MagicMock(spec=LlamaMock)
    instance.create_completion = MagicMock()
    instance.create_chat_completion = MagicMock()
    instance.model_params = MagicMock()
    instance.model_params.n_ctx_train = 2048
    return instance


@pytest.fixture()
def mock_token_usage_manager_for_llama() -> AsyncMock:
    tum = AsyncMock(spec=TokenUsageManager)
    tum.record_usage = AsyncMock()
    return tum


@pytest.fixture()
async def llama_internal_provider(
    mock_llama_instance: MagicMock,
    mock_token_usage_manager_for_llama: AsyncMock,
    tmp_path: Any,
) -> LlamaCppInternalLLMProviderPlugin:
    provider = LlamaCppInternalLLMProviderPlugin()
    dummy_model_file = tmp_path / "dummy.gguf"
    dummy_model_file.touch()

    with patch(
        "genie_tooling.llm_providers.impl.llama_cpp_internal_provider.Llama",
        return_value=mock_llama_instance,
    ):
        await provider.setup(
            config={
                "model_path": str(dummy_model_file),
                "token_usage_manager": mock_token_usage_manager_for_llama,
            }
        )
    return provider


@pytest.mark.asyncio()
async def test_setup_model_path_missing(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger=PROVIDER_LOGGER_NAME)
    provider = LlamaCppInternalLLMProviderPlugin()
    await provider.setup(config={})
    assert provider._model_client is None
    assert (
        "'model_path' not provided in configuration. Plugin will be disabled"
        in caplog.text
    )


@pytest.mark.asyncio()
async def test_setup_llama_constructor_fails(
    tmp_path: Any, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = LlamaCppInternalLLMProviderPlugin()
    dummy_model_file = tmp_path / "dummy_fail.gguf"
    dummy_model_file.touch()

    with patch(
        "genie_tooling.llm_providers.impl.llama_cpp_internal_provider.Llama",
        side_effect=RuntimeError("Llama init failed"),
    ):
        await provider.setup(config={"model_path": str(dummy_model_file)})
    assert provider._model_client is None
    assert "Failed to initialize Llama model: Llama init failed" in caplog.text


@pytest.mark.asyncio()
async def test_generate_success_non_streaming(
    llama_internal_provider: LlamaCppInternalLLMProviderPlugin,
    mock_llama_instance: MagicMock,
    mock_token_usage_manager_for_llama: AsyncMock,
):
    provider = await llama_internal_provider
    mock_llama_instance.create_completion.return_value = {
        "choices": [{"text": "Generated text", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    response = await provider.generate("Test prompt")

    assert response["text"] == "Generated text"
    assert response["finish_reason"] == "stop"
    assert response["usage"]["total_tokens"] == 15 # type: ignore
    mock_llama_instance.create_completion.assert_called_once()
    mock_token_usage_manager_for_llama.record_usage.assert_awaited_once()


@pytest.mark.asyncio()
async def test_generate_success_streaming(
    llama_internal_provider: LlamaCppInternalLLMProviderPlugin,
    mock_llama_instance: MagicMock,
    mock_token_usage_manager_for_llama: AsyncMock,
):
    provider = await llama_internal_provider

    def stream_mock_generate():
        yield {"choices": [{"text": "Hello ", "finish_reason": None}], "usage": None}
        yield {
            "choices": [{"text": "World!", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }

    mock_llama_instance.create_completion.return_value = stream_mock_generate()

    stream_response = await provider.generate("Stream test", stream=True)
    chunks: List[LLMCompletionChunk] = []
    async for chunk in stream_response: # type: ignore
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0]["text_delta"] == "Hello "
    assert chunks[1]["text_delta"] == "World!"
    assert chunks[1]["finish_reason"] == "stop"
    assert chunks[1]["usage_delta"]["total_tokens"] == 5 # type: ignore
    mock_token_usage_manager_for_llama.record_usage.assert_awaited_once()


@pytest.mark.asyncio()
async def test_chat_success_non_streaming(
    llama_internal_provider: LlamaCppInternalLLMProviderPlugin,
    mock_llama_instance: MagicMock,
    mock_token_usage_manager_for_llama: AsyncMock,
):
    provider = await llama_internal_provider
    messages: List[ChatMessage] = [{"role": "user", "content": "Hi"}]
    mock_llama_instance.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Chat response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
    }

    response = await provider.chat(messages)

    assert response["message"]["content"] == "Chat response"
    assert response["finish_reason"] == "stop"
    assert response["usage"]["total_tokens"] == 11 # type: ignore
    mock_llama_instance.create_chat_completion.assert_called_once()
    mock_token_usage_manager_for_llama.record_usage.assert_awaited_once()


@pytest.mark.asyncio()
async def test_chat_success_streaming(
    llama_internal_provider: LlamaCppInternalLLMProviderPlugin,
    mock_llama_instance: MagicMock,
    mock_token_usage_manager_for_llama: AsyncMock,
):
    provider = await llama_internal_provider
    messages: List[ChatMessage] = [{"role": "user", "content": "Stream hi"}]

    def stream_mock_chat():
        yield {
            "choices": [{"delta": {"role": "assistant", "content": "Streaming "}}],
            "usage": None,
        }
        yield {
            "choices": [{"delta": {"content": "response."}}],
            "usage": None,
        }
        yield {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }

    mock_llama_instance.create_chat_completion.return_value = stream_mock_chat()

    stream_response = await provider.chat(messages, stream=True)
    chunks: List[LLMChatChunk] = []
    async for chunk in stream_response: # type: ignore
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0]["message_delta"]["content"] == "Streaming " # type: ignore
    assert chunks[1]["message_delta"]["content"] == "response." # type: ignore
    assert chunks[2]["finish_reason"] == "stop"
    assert chunks[2]["usage_delta"]["total_tokens"] == 7 # type: ignore
    mock_token_usage_manager_for_llama.record_usage.assert_awaited_once()


@pytest.mark.asyncio()
async def test_generate_client_not_initialized(
    llama_internal_provider: LlamaCppInternalLLMProviderPlugin,
):
    provider = await llama_internal_provider
    provider._model_client = None
    with pytest.raises(RuntimeError, match="Model client not initialized"):
        await provider.generate("test")


@pytest.mark.asyncio()
async def test_chat_client_not_initialized(
    llama_internal_provider: LlamaCppInternalLLMProviderPlugin,
):
    provider = await llama_internal_provider
    provider._model_client = None
    with pytest.raises(RuntimeError, match="Model client not initialized"):
        await provider.chat([{"role": "user", "content": "test"}])


@pytest.mark.asyncio()
async def test_get_model_info(
    llama_internal_provider: LlamaCppInternalLLMProviderPlugin,
):
    provider = await llama_internal_provider
    info = await provider.get_model_info()
    assert info["provider"] == "llama.cpp (internal)"
    assert "dummy.gguf" in info["model_path"]
    assert info["llama_cpp_model_params_available"] is True


@pytest.mark.asyncio()
async def test_teardown(llama_internal_provider: LlamaCppInternalLLMProviderPlugin):
    provider = await llama_internal_provider
    assert provider._model_client is not None
    await provider.teardown()
    assert provider._model_client is None
    assert provider._token_usage_manager is None

@pytest.mark.asyncio()
async def test_generate_with_gbnf_schema_pydantic(
    llama_internal_provider: LlamaCppInternalLLMProviderPlugin,
    mock_llama_instance: MagicMock,
    mock_token_usage_manager_for_llama: AsyncMock,
):
    provider = await llama_internal_provider
    from pydantic import BaseModel as PydanticBaseModel  # Local import for test

    class TestSchema(PydanticBaseModel):
        key: str

    mock_llama_instance.create_completion.return_value = {
        "choices": [{"text": '{"key":"value"}', "finish_reason": "stop"}], "usage": {}
    }
    with patch("genie_tooling.llm_providers.impl.llama_cpp_internal_provider.LlamaGrammar.from_string") as mock_from_string:
        await provider.generate("test", output_schema=TestSchema)
    mock_from_string.assert_called_once()
    mock_llama_instance.create_completion.assert_called_once()
    assert "grammar" in mock_llama_instance.create_completion.call_args.kwargs

@pytest.mark.asyncio()
async def test_chat_with_gbnf_schema_dict(
    llama_internal_provider: LlamaCppInternalLLMProviderPlugin,
    mock_llama_instance: MagicMock,
    mock_token_usage_manager_for_llama: AsyncMock,
):
    provider = await llama_internal_provider
    dict_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    mock_llama_instance.create_chat_completion.return_value = {
        "choices": [{"message": {"role": "assistant", "content": '{"name":"Test"}'}, "finish_reason": "stop"}], "usage": {}
    }
    with patch("genie_tooling.llm_providers.impl.llama_cpp_internal_provider.LlamaGrammar.from_string") as mock_from_string:
        await provider.chat([{"role":"user", "content":"test"}], output_schema=dict_schema)
    mock_from_string.assert_called_once()
    mock_llama_instance.create_chat_completion.assert_called_once()
    assert "grammar" in mock_llama_instance.create_chat_completion.call_args.kwargs

@pytest.mark.asyncio()
async def test_record_usage_no_manager(
    llama_internal_provider: LlamaCppInternalLLMProviderPlugin,
):
    provider = await llama_internal_provider
    provider._token_usage_manager = None # Simulate no manager
    # This call should not raise an error
    await provider._record_usage_if_manager({"prompt_tokens":1}, "generate")
    # No assertion needed other than it doesn't crash
