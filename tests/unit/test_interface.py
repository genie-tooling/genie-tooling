### tests/unit/test_interfaces.py
import uuid
from typing import Any as TypingAny
from typing import Dict
from typing import List as TypingList
from typing import Optional as TypingOptional
from unittest.mock import ANY, AsyncMock, MagicMock
import logging
import pytest
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.conversation.impl.manager import ConversationStateManager
from genie_tooling.conversation.types import ConversationState
from genie_tooling.core.types import RetrievedChunk
from genie_tooling.guardrails.manager import GuardrailManager
from genie_tooling.guardrails.types import GuardrailViolation
from genie_tooling.hitl.manager import HITLManager
from genie_tooling.hitl.types import ApprovalRequest, ApprovalResponse
from genie_tooling.interfaces import (
    ConversationInterface,
    HITLInterface,
    LLMInterface,
    ObservabilityInterface,
    PromptInterface,
    RAGInterface,
    TaskQueueInterface,
    UsageTrackingInterface,
)
from genie_tooling.llm_providers.manager import LLMProviderManager
from genie_tooling.llm_providers.types import (
    ChatMessage,
    LLMChatChunk,
    LLMChatResponse,
    LLMCompletionChunk,
    LLMCompletionResponse,
    LLMUsageInfo,
)
from genie_tooling.observability.manager import InteractionTracingManager
from genie_tooling.prompts.llm_output_parsers.manager import LLMOutputParserManager
from genie_tooling.prompts.manager import PromptManager
from genie_tooling.prompts.types import PromptData, PromptIdentifier
from genie_tooling.rag.manager import RAGManager
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.task_queues.manager import DistributedTaskQueueManager
from genie_tooling.token_usage.manager import TokenUsageManager
from genie_tooling.token_usage.types import TokenUsageRecord


# Helper concrete class for RAG tests
class _ConcreteRetrievedChunk(RetrievedChunk):
    def __init__(
        self,
        id: TypingOptional[str],
        content: str,
        score: float,
        metadata: TypingOptional[Dict[str, TypingAny]] = None,
        rank: TypingOptional[int] = None,
    ):
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata or {}
        self.rank = rank


# --- LLMInterface Tests ---
@pytest.fixture()
def mock_llm_provider_manager() -> MagicMock:
    return MagicMock(spec=LLMProviderManager)


@pytest.fixture()
def mock_llm_output_parser_manager() -> MagicMock:
    return MagicMock(spec=LLMOutputParserManager)


@pytest.fixture()
def mock_tracing_manager_for_llm() -> MagicMock:
    mgr = MagicMock(spec=InteractionTracingManager)
    mgr.trace_event = AsyncMock()
    return mgr


@pytest.fixture()
def mock_guardrail_manager_for_llm() -> MagicMock:
    mgr = MagicMock(spec=GuardrailManager)
    mgr.check_input_guardrails = AsyncMock(return_value=GuardrailViolation(action="allow", reason=""))
    mgr.check_output_guardrails = AsyncMock(return_value=GuardrailViolation(action="allow", reason=""))
    return mgr


@pytest.fixture()
def mock_token_usage_manager_for_llm() -> MagicMock:
    mgr = MagicMock(spec=TokenUsageManager)
    mgr.record_usage = AsyncMock()
    return mgr


@pytest.fixture()
def llm_interface(
    mock_llm_provider_manager: MagicMock,
    mock_llm_output_parser_manager: MagicMock,
    mock_tracing_manager_for_llm: MagicMock,
    mock_guardrail_manager_for_llm: MagicMock,
    mock_token_usage_manager_for_llm: MagicMock,
) -> LLMInterface:
    return LLMInterface(
        llm_provider_manager=mock_llm_provider_manager,
        default_provider_id="default_llm",
        output_parser_manager=mock_llm_output_parser_manager,
        tracing_manager=mock_tracing_manager_for_llm,
        guardrail_manager=mock_guardrail_manager_for_llm,
        token_usage_manager=mock_token_usage_manager_for_llm,
    )


@pytest.mark.asyncio()
async def test_llm_interface_generate_success(llm_interface: LLMInterface, mock_llm_provider_manager: MagicMock):
    mock_provider = AsyncMock()
    dummy_usage: LLMUsageInfo = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    mock_provider.generate = AsyncMock(
        return_value=LLMCompletionResponse(text="Generated text", finish_reason="stop", usage=dummy_usage, raw_response={})
    )
    mock_llm_provider_manager.get_llm_provider = AsyncMock(return_value=mock_provider)

    response = await llm_interface.generate("Test prompt")
    assert response["text"] == "Generated text"
    mock_llm_provider_manager.get_llm_provider.assert_awaited_once_with("default_llm")
    mock_provider.generate.assert_awaited_once_with("Test prompt", stream=False)
    llm_interface._tracing_manager.trace_event.assert_any_call("llm.generate.start", ANY, "LLMInterface", ANY)  # type: ignore
    llm_interface._token_usage_manager.record_usage.assert_awaited_once()  # type: ignore


@pytest.mark.asyncio()
async def test_llm_interface_chat_success(llm_interface: LLMInterface, mock_llm_provider_manager: MagicMock):
    mock_provider = AsyncMock()
    dummy_usage: LLMUsageInfo = {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 20}
    mock_provider.chat = AsyncMock(
        return_value=LLMChatResponse(
            message=ChatMessage(role="assistant", content="Chat response"),
            finish_reason="stop",
            usage=dummy_usage,
            raw_response={},
        )
    )
    mock_llm_provider_manager.get_llm_provider = AsyncMock(return_value=mock_provider)
    messages: TypingList[ChatMessage] = [{"role": "user", "content": "Hello"}]

    response = await llm_interface.chat(messages, provider_id="custom_llm")
    assert response["message"]["content"] == "Chat response"
    mock_llm_provider_manager.get_llm_provider.assert_awaited_once_with("custom_llm")
    mock_provider.chat.assert_awaited_once_with(messages, stream=False)
    llm_interface._tracing_manager.trace_event.assert_any_call("llm.chat.start", ANY, "LLMInterface", ANY)  # type: ignore
    llm_interface._token_usage_manager.record_usage.assert_awaited_once()  # type: ignore


@pytest.mark.asyncio()
async def test_llm_interface_generate_no_provider_id_error(llm_interface: LLMInterface):
    llm_interface._default_provider_id = None
    with pytest.raises(ValueError, match="No LLM provider ID specified"):
        await llm_interface.generate("Test prompt")


@pytest.mark.asyncio()
async def test_llm_interface_generate_provider_not_found_error(
    llm_interface: LLMInterface, mock_llm_provider_manager: MagicMock
):
    mock_llm_provider_manager.get_llm_provider = AsyncMock(return_value=None)
    with pytest.raises(RuntimeError, match="LLM Provider 'default_llm' not found"):
        await llm_interface.generate("Test prompt")


@pytest.mark.asyncio()
async def test_llm_interface_input_guardrail_block_generate(
    llm_interface: LLMInterface, mock_guardrail_manager_for_llm: MagicMock
):
    mock_guardrail_manager_for_llm.check_input_guardrails = AsyncMock(
        return_value=GuardrailViolation(action="block", reason="Blocked by input guardrail")
    )
    with pytest.raises(PermissionError, match="LLM generate blocked by input guardrail: Blocked by input guardrail"):
        await llm_interface.generate("Risky prompt")
    llm_interface._tracing_manager.trace_event.assert_any_call(  # type: ignore
        "llm.generate.blocked_by_input_guardrail", ANY, "LLMInterface", ANY
    )


@pytest.mark.asyncio()
async def test_llm_interface_output_guardrail_block_generate(
    llm_interface: LLMInterface, mock_llm_provider_manager: MagicMock, mock_guardrail_manager_for_llm: MagicMock
):
    mock_provider = AsyncMock()
    dummy_usage: LLMUsageInfo = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    mock_provider.generate = AsyncMock(
        return_value=LLMCompletionResponse(text="Risky output", finish_reason="stop", usage=dummy_usage, raw_response={})
    )
    mock_llm_provider_manager.get_llm_provider = AsyncMock(return_value=mock_provider)
    mock_guardrail_manager_for_llm.check_output_guardrails = AsyncMock(
        return_value=GuardrailViolation(action="block", reason="Blocked by output guardrail")
    )

    response = await llm_interface.generate("Generate risky output")
    assert response["text"] == "[RESPONSE BLOCKED: Blocked by output guardrail]"
    assert response["finish_reason"] == "blocked_by_guardrail"
    llm_interface._tracing_manager.trace_event.assert_any_call(  # type: ignore
        "llm.generate.blocked_by_output_guardrail", ANY, "LLMInterface", ANY
    )


# --- LLMInterface Streaming Tests ---
async def mock_generate_stream():
    yield LLMCompletionChunk(text_delta="Hello ")
    yield LLMCompletionChunk(text_delta="World", finish_reason="stop", usage_delta={"total_tokens": 2})


async def mock_chat_stream():
    yield LLMChatChunk(message_delta={"role": "assistant", "content": "First part. "})
    yield LLMChatChunk(message_delta={"content": "Second part."}, finish_reason="stop", usage_delta={"total_tokens": 5})


@pytest.mark.asyncio()
async def test_llm_interface_generate_streaming(llm_interface: LLMInterface, mock_llm_provider_manager: MagicMock):
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=mock_generate_stream())
    mock_llm_provider_manager.get_llm_provider = AsyncMock(return_value=mock_provider)

    result_stream = await llm_interface.generate("Test stream", stream=True)
    chunks = [chunk async for chunk in result_stream]  # type: ignore

    assert len(chunks) == 2
    assert chunks[0]["text_delta"] == "Hello "
    assert chunks[1]["text_delta"] == "World"
    assert chunks[1]["finish_reason"] == "stop"
    assert chunks[1]["usage_delta"] == {"total_tokens": 2}
    llm_interface._token_usage_manager.record_usage.assert_awaited_once()  # type: ignore
    llm_interface._tracing_manager.trace_event.assert_any_call("llm.generate.stream_end", ANY, "LLMInterface", ANY)  # type: ignore


@pytest.mark.asyncio()
async def test_llm_interface_chat_streaming(llm_interface: LLMInterface, mock_llm_provider_manager: MagicMock):
    mock_provider = AsyncMock()
    mock_provider.chat = AsyncMock(return_value=mock_chat_stream())
    mock_llm_provider_manager.get_llm_provider = AsyncMock(return_value=mock_provider)
    messages: TypingList[ChatMessage] = [{"role": "user", "content": "Stream chat"}]

    result_stream = await llm_interface.chat(messages, stream=True)
    chunks = [chunk async for chunk in result_stream]  # type: ignore

    assert len(chunks) == 2
    assert chunks[0]["message_delta"]["content"] == "First part. "  # type: ignore
    assert chunks[1]["message_delta"]["content"] == "Second part."  # type: ignore
    assert chunks[1]["finish_reason"] == "stop"
    assert chunks[1]["usage_delta"] == {"total_tokens": 5}
    llm_interface._token_usage_manager.record_usage.assert_awaited_once()  # type: ignore
    llm_interface._tracing_manager.trace_event.assert_any_call("llm.chat.stream_end", ANY, "LLMInterface", ANY)  # type: ignore


@pytest.mark.asyncio()
async def test_llm_interface_generate_streaming_output_guardrail_block(
    llm_interface: LLMInterface, mock_llm_provider_manager: MagicMock, mock_guardrail_manager_for_llm: MagicMock
):
    mock_provider = AsyncMock()

    async def guarded_stream():
        yield LLMCompletionChunk(text_delta="Allowed text. ")
        yield LLMCompletionChunk(text_delta="Risky text part.")  # This one will be blocked
        yield LLMCompletionChunk(text_delta="More text.")  # Should not be reached

    mock_provider.generate = AsyncMock(return_value=guarded_stream())
    mock_llm_provider_manager.get_llm_provider = AsyncMock(return_value=mock_provider)

    async def output_guardrail_check_side_effect(data, context):
        if "Risky text part" in data:
            return GuardrailViolation(action="block", reason="Risky content detected")
        return GuardrailViolation(action="allow", reason="")

    mock_guardrail_manager_for_llm.check_output_guardrails.side_effect = output_guardrail_check_side_effect

    result_stream = await llm_interface.generate("Test stream guardrail", stream=True)
    chunks = [chunk async for chunk in result_stream]  # type: ignore

    assert len(chunks) == 2
    assert chunks[0]["text_delta"] == "Allowed text. "
    assert chunks[1]["text_delta"] == "[STREAM BLOCKED: Risky content detected]"
    assert chunks[1]["finish_reason"] == "blocked_by_guardrail"


# --- LLMInterface parse_output error tests ---
@pytest.mark.asyncio()
async def test_llm_interface_parse_output_no_text_content(llm_interface: LLMInterface):
    response_no_text = LLMChatResponse(
        message=ChatMessage(role="assistant", content=None), finish_reason="stop", usage=None, raw_response={}
    )
    with pytest.raises(ValueError, match="No text content found in LLM response to parse."):
        await llm_interface.parse_output(response_no_text)
    llm_interface._tracing_manager.trace_event.assert_any_call("llm.parse_output.error", ANY, "LLMInterface", ANY)  # type: ignore


@pytest.mark.asyncio()
async def test_llm_interface_parse_output_parser_not_found(
    llm_interface: LLMInterface, mock_llm_output_parser_manager: MagicMock
):
    mock_llm_output_parser_manager.parse = AsyncMock(side_effect=RuntimeError("Parser not found simulation"))

    response_to_parse = LLMCompletionResponse(text='{"data":1}', finish_reason="stop", usage=None, raw_response={})
    with pytest.raises(RuntimeError, match="Parser not found simulation"):
        await llm_interface.parse_output(response_to_parse, parser_id="non_existent_parser")
    llm_interface._tracing_manager.trace_event.assert_any_call("llm.parse_output.error", ANY, "LLMInterface", ANY)  # type: ignore


@pytest.mark.asyncio()
async def test_llm_interface_parse_output_parser_raises_error(
    llm_interface: LLMInterface, mock_llm_output_parser_manager: MagicMock
):
    mock_llm_output_parser_manager.parse = AsyncMock(side_effect=ValueError("Parsing failed badly"))
    response_to_parse = LLMCompletionResponse(text="bad data", finish_reason="stop", usage=None, raw_response={})
    with pytest.raises(ValueError, match="Parsing failed badly"):
        await llm_interface.parse_output(response_to_parse)
    llm_interface._tracing_manager.trace_event.assert_any_call("llm.parse_output.error", ANY, "LLMInterface", ANY)  # type: ignore


# --- RAGInterface Tests ---
@pytest.fixture()
def mock_rag_manager() -> MagicMock:
    return MagicMock(spec=RAGManager)


@pytest.fixture()
def mock_middleware_config_for_rag() -> MagicMock:
    cfg = MagicMock(spec=MiddlewareConfig)
    cfg.default_rag_loader_id = "default_loader"
    cfg.default_rag_splitter_id = "default_splitter"
    cfg.default_rag_embedder_id = "default_embedder"
    cfg.default_rag_vector_store_id = "default_vector_store"
    cfg.default_rag_retriever_id = "default_retriever"
    cfg.document_loader_configurations = {}
    cfg.text_splitter_configurations = {}
    cfg.embedding_generator_configurations = {}
    cfg.vector_store_configurations = {}
    cfg.retriever_configurations = {}
    return cfg


@pytest.fixture()
def mock_key_provider_for_rag() -> MagicMock:
    return MagicMock(spec=KeyProvider)


@pytest.fixture()
def mock_tracing_manager_for_rag() -> MagicMock:
    mgr = MagicMock(spec=InteractionTracingManager)
    mgr.trace_event = AsyncMock()
    return mgr


@pytest.fixture()
def rag_interface(
    mock_rag_manager: MagicMock,
    mock_middleware_config_for_rag: MagicMock,
    mock_key_provider_for_rag: MagicMock,
    mock_tracing_manager_for_rag: MagicMock,
) -> RAGInterface:
    return RAGInterface(
        rag_manager=mock_rag_manager,
        config=mock_middleware_config_for_rag,
        key_provider=mock_key_provider_for_rag,
        tracing_manager=mock_tracing_manager_for_rag,
    )


@pytest.mark.asyncio()
async def test_rag_interface_index_directory_success(rag_interface: RAGInterface, mock_rag_manager: MagicMock):
    mock_rag_manager.index_data_source = AsyncMock(return_value={"status": "success", "added_count": 10})
    path = "./test_docs"
    collection = "my_collection"

    result = await rag_interface.index_directory(path, collection_name=collection)

    assert result["status"] == "success"
    mock_rag_manager.index_data_source.assert_awaited_once()
    call_kwargs = mock_rag_manager.index_data_source.call_args.kwargs
    assert call_kwargs["loader_source_uri"] == path
    assert call_kwargs["vector_store_config"]["collection_name"] == collection
    assert call_kwargs["embedder_config"]["key_provider"] is rag_interface._key_provider
    rag_interface._tracing_manager.trace_event.assert_any_call("rag.index_directory.start", ANY, "RAGInterface", ANY)  # type: ignore


@pytest.mark.asyncio()
async def test_rag_interface_search_success(rag_interface: RAGInterface, mock_rag_manager: MagicMock):
    mock_rag_manager.retrieve_from_query = AsyncMock(
        return_value=[_ConcreteRetrievedChunk(id="c1", content="text", score=0.9, metadata={})]
    )
    query = "What is RAG?"

    results = await rag_interface.search(query, top_k=3)

    assert len(results) == 1
    mock_rag_manager.retrieve_from_query.assert_awaited_once_with(
        query_text=query, retriever_id="default_retriever", retriever_config=ANY, top_k=3
    )
    rag_interface._tracing_manager.trace_event.assert_any_call("rag.search.start", ANY, "RAGInterface", ANY)  # type: ignore


@pytest.mark.asyncio()
async def test_rag_interface_index_directory_missing_embedder_id(
    rag_interface: RAGInterface, mock_middleware_config_for_rag: MagicMock
):
    mock_middleware_config_for_rag.default_rag_embedder_id = None  # Simulate missing config
    with pytest.raises(ValueError, match="RAG embedder ID not resolved for index_directory."):
        await rag_interface.index_directory("./docs")


@pytest.mark.asyncio()
async def test_rag_interface_index_web_page_missing_vector_store_id(
    rag_interface: RAGInterface, mock_middleware_config_for_rag: MagicMock
):
    mock_middleware_config_for_rag.default_rag_vector_store_id = None  # Simulate missing config
    with pytest.raises(ValueError, match="RAG vector store ID not resolved for index_web_page."):
        await rag_interface.index_web_page("http://example.com")


# --- ObservabilityInterface Tests ---
@pytest.fixture()
def mock_tracing_manager() -> MagicMock:
    mgr = MagicMock(spec=InteractionTracingManager)
    mgr.trace_event = AsyncMock()
    return mgr


@pytest.fixture()
def observability_interface(mock_tracing_manager: MagicMock) -> ObservabilityInterface:
    return ObservabilityInterface(tracing_manager=mock_tracing_manager)


@pytest.mark.asyncio()
async def test_observability_interface_trace_event(
    observability_interface: ObservabilityInterface, mock_tracing_manager: MagicMock
):
    event_name = "test.event"
    data = {"key": "value"}
    component = "TestComponent"
    corr_id = "test-corr-id"

    await observability_interface.trace_event(event_name, data, component, corr_id)
    mock_tracing_manager.trace_event.assert_awaited_once_with(event_name, data, component, corr_id)


@pytest.mark.asyncio()
async def test_observability_interface_no_manager(observability_interface: ObservabilityInterface):
    observability_interface._tracing_manager = None  # type: ignore
    # Should not raise an error, just be a no-op
    await observability_interface.trace_event("test.event", {}, "Comp", "id")


# --- HITLInterface Tests ---
@pytest.fixture()
def mock_hitl_manager() -> MagicMock:
    mgr = MagicMock(spec=HITLManager)
    mgr.request_approval = AsyncMock()
    return mgr


@pytest.fixture()
def hitl_interface(mock_hitl_manager: MagicMock) -> HITLInterface:
    return HITLInterface(hitl_manager=mock_hitl_manager)


@pytest.mark.asyncio()
async def test_hitl_interface_request_approval(hitl_interface: HITLInterface, mock_hitl_manager: MagicMock):
    mock_hitl_manager.request_approval.return_value = ApprovalResponse(
        request_id="req1", status="approved", approver_id=None, reason=None, timestamp=None
    )
    req = ApprovalRequest(request_id="req1", prompt="Approve?", data_to_approve={})

    response = await hitl_interface.request_approval(req, approver_id="custom_approver")

    assert response["status"] == "approved"
    mock_hitl_manager.request_approval.assert_awaited_once_with(req, "custom_approver")


@pytest.mark.asyncio()
async def test_hitl_interface_no_manager(hitl_interface: HITLInterface):
    hitl_interface._hitl_manager = None  # type: ignore
    req_id_val = "req_no_mgr_" + str(uuid.uuid4())
    req = ApprovalRequest(request_id=req_id_val, prompt="Test", data_to_approve={})
    response = await hitl_interface.request_approval(req)
    assert response["status"] == "error"
    assert response["reason"] == "HITL system unavailable."
    assert response["request_id"] == req_id_val


# --- UsageTrackingInterface Tests ---
@pytest.fixture()
def mock_token_usage_manager() -> MagicMock:
    mgr = MagicMock(spec=TokenUsageManager)
    mgr.record_usage = AsyncMock()
    mgr.get_summary = AsyncMock()
    return mgr


@pytest.fixture()
def usage_tracking_interface(mock_token_usage_manager: MagicMock) -> UsageTrackingInterface:
    return UsageTrackingInterface(token_usage_manager=mock_token_usage_manager)


@pytest.mark.asyncio()
async def test_usage_tracking_interface_record_usage(
    usage_tracking_interface: UsageTrackingInterface, mock_token_usage_manager: MagicMock
):
    record = TokenUsageRecord(provider_id="p1", model_name="m1", total_tokens=100, timestamp=123.0)

    await usage_tracking_interface.record_usage(record)
    mock_token_usage_manager.record_usage.assert_awaited_once_with(record)


@pytest.mark.asyncio()
async def test_usage_tracking_interface_get_summary(
    usage_tracking_interface: UsageTrackingInterface, mock_token_usage_manager: MagicMock
):
    mock_token_usage_manager.get_summary.return_value = {"total": 1000}

    summary = await usage_tracking_interface.get_summary(recorder_id="rec1", filter_criteria={"user": "u1"})

    assert summary == {"total": 1000}
    mock_token_usage_manager.get_summary.assert_awaited_once_with("rec1", {"user": "u1"})


@pytest.mark.asyncio()
async def test_usage_tracking_interface_no_manager(usage_tracking_interface: UsageTrackingInterface):
    usage_tracking_interface._token_usage_manager = None  # type: ignore
    record = TokenUsageRecord(provider_id="p1", model_name="m1", total_tokens=100, timestamp=123.0)
    await usage_tracking_interface.record_usage(record)  # Should not error
    summary = await usage_tracking_interface.get_summary()
    assert summary == {"error": "TokenUsageManager unavailable."}


# --- PromptInterface Tests ---
@pytest.fixture()
def mock_prompt_manager() -> MagicMock:
    mgr = MagicMock(spec=PromptManager)
    mgr.get_raw_template = AsyncMock()
    mgr.render_prompt = AsyncMock()
    mgr.render_chat_prompt = AsyncMock()
    mgr.list_available_templates = AsyncMock()
    return mgr


@pytest.fixture()
def prompt_interface(mock_prompt_manager: MagicMock) -> PromptInterface:
    return PromptInterface(prompt_manager=mock_prompt_manager)


@pytest.mark.asyncio()
async def test_prompt_interface_get_template_content(prompt_interface: PromptInterface, mock_prompt_manager: MagicMock):
    mock_prompt_manager.get_raw_template.return_value = "Template content"
    content = await prompt_interface.get_prompt_template_content("name", "v1", "reg1")
    assert content == "Template content"
    mock_prompt_manager.get_raw_template.assert_awaited_once_with("name", "v1", "reg1")


@pytest.mark.asyncio()
async def test_prompt_interface_render_prompt(prompt_interface: PromptInterface, mock_prompt_manager: MagicMock):
    mock_prompt_manager.render_prompt.return_value = "Rendered prompt"
    data: PromptData = {"var": "val"}
    # FIX: Use keyword arguments to avoid misinterpreting positional arguments
    rendered = await prompt_interface.render_prompt(
        name="name", data=data, version="v1", registry_id="reg1", template_engine_id="eng1"
    )
    assert rendered == "Rendered prompt"
    # FIX: Update assertion to use keyword arguments
    mock_prompt_manager.render_prompt.assert_awaited_once_with(
        "name", data, None, "v1", "reg1", "eng1"
    )


@pytest.mark.asyncio()
async def test_prompt_interface_list_templates(prompt_interface: PromptInterface, mock_prompt_manager: MagicMock):
    mock_prompt_manager.list_available_templates.return_value = [
        PromptIdentifier(name="p1", version="v1", description="d1")
    ]
    templates = await prompt_interface.list_templates("reg1")
    assert len(templates) == 1
    mock_prompt_manager.list_available_templates.assert_awaited_once_with("reg1")


# --- ConversationInterface Tests ---
@pytest.fixture()
def mock_conversation_manager() -> MagicMock:
    mgr = MagicMock(spec=ConversationStateManager)
    mgr.load_state = AsyncMock()
    mgr.save_state = AsyncMock()
    mgr.add_message = AsyncMock()
    mgr.delete_state = AsyncMock()
    return mgr


@pytest.fixture()
def conversation_interface(mock_conversation_manager: MagicMock) -> ConversationInterface:
    return ConversationInterface(conversation_manager=mock_conversation_manager)


@pytest.mark.asyncio()
async def test_conversation_interface_load_state(
    conversation_interface: ConversationInterface, mock_conversation_manager: MagicMock
):
    mock_conversation_manager.load_state.return_value = ConversationState(session_id="s1", history=[])
    state = await conversation_interface.load_state("s1", "p1")
    assert state is not None
    assert state["session_id"] == "s1"
    mock_conversation_manager.load_state.assert_awaited_once_with("s1", "p1")


@pytest.mark.asyncio()
async def test_conversation_interface_save_state(
    conversation_interface: ConversationInterface, mock_conversation_manager: MagicMock
):
    state = ConversationState(session_id="s1", history=[])
    await conversation_interface.save_state(state, "p1")
    mock_conversation_manager.save_state.assert_awaited_once_with(state, "p1")


@pytest.mark.asyncio()
async def test_conversation_interface_add_message(
    conversation_interface: ConversationInterface, mock_conversation_manager: MagicMock
):
    msg: ChatMessage = {"role": "user", "content": "Hi"}
    await conversation_interface.add_message("s1", msg, "p1")
    mock_conversation_manager.add_message.assert_awaited_once_with("s1", msg, "p1")


@pytest.mark.asyncio()
async def test_conversation_interface_delete_state(
    conversation_interface: ConversationInterface, mock_conversation_manager: MagicMock
):
    mock_conversation_manager.delete_state.return_value = True
    result = await conversation_interface.delete_state("s1", "p1")
    assert result is True
    mock_conversation_manager.delete_state.assert_awaited_once_with("s1", "p1")


@pytest.mark.asyncio()
async def test_conversation_interface_no_manager(conversation_interface: ConversationInterface):
    conversation_interface._conversation_manager = None  # type: ignore
    assert await conversation_interface.load_state("s1") is None
    await conversation_interface.save_state(ConversationState(session_id="s1", history=[]))
    await conversation_interface.add_message("s1", {"role": "user", "content": "test"})
    assert await conversation_interface.delete_state("s1") is False


# --- TaskQueueInterface Tests (New) ---
@pytest.fixture()
def mock_task_queue_manager() -> MagicMock:
    mgr = MagicMock(spec=DistributedTaskQueueManager)
    mgr.submit_task = AsyncMock(return_value="task_abc_123")
    mgr.get_task_status = AsyncMock(return_value="success")
    mgr.get_task_result = AsyncMock(return_value={"data": "task done"})
    mgr.revoke_task = AsyncMock(return_value=True)
    return mgr


@pytest.fixture()
def task_queue_interface(mock_task_queue_manager: MagicMock) -> TaskQueueInterface:
    return TaskQueueInterface(task_queue_manager=mock_task_queue_manager)


@pytest.mark.asyncio()
async def test_task_queue_interface_submit_task(
    task_queue_interface: TaskQueueInterface, mock_task_queue_manager: MagicMock
):
    task_id = await task_queue_interface.submit_task(
        "my_task", args=(1,), kwargs={"op": "add"}, queue_id="q1", task_options={"countdown": 10}
    )
    assert task_id == "task_abc_123"
    mock_task_queue_manager.submit_task.assert_awaited_once_with(
        "my_task", (1,), {"op": "add"}, "q1", {"countdown": 10}
    )


@pytest.mark.asyncio()
async def test_task_queue_interface_get_task_status(
    task_queue_interface: TaskQueueInterface, mock_task_queue_manager: MagicMock
):
    status = await task_queue_interface.get_task_status("task_abc_123", "q1")
    assert status == "success"
    mock_task_queue_manager.get_task_status.assert_awaited_once_with("task_abc_123", "q1")


@pytest.mark.asyncio()
async def test_task_queue_interface_get_task_result(
    task_queue_interface: TaskQueueInterface, mock_task_queue_manager: MagicMock
):
    result = await task_queue_interface.get_task_result("task_abc_123", "q1", timeout_seconds=5.0)
    assert result == {"data": "task done"}
    mock_task_queue_manager.get_task_result.assert_awaited_once_with("task_abc_123", "q1", 5.0)


@pytest.mark.asyncio()
async def test_task_queue_interface_revoke_task(
    task_queue_interface: TaskQueueInterface, mock_task_queue_manager: MagicMock
):
    revoked = await task_queue_interface.revoke_task("task_abc_123", "q1", terminate=True)
    assert revoked is True
    mock_task_queue_manager.revoke_task.assert_awaited_once_with("task_abc_123", "q1", True)


@pytest.mark.asyncio()
async def test_task_queue_interface_no_manager(task_queue_interface: TaskQueueInterface, caplog: pytest.LogCaptureFixture):
    task_queue_interface._task_queue_manager = None  # type: ignore
    # FIX: Capture logs to verify the error is logged as expected.
    with caplog.at_level(logging.ERROR):
        assert await task_queue_interface.submit_task("t") is None
        assert "DistributedTaskQueueManager not available" in caplog.text
        caplog.clear()

        # FIX: The implementation returns the literal "unknown", not None.
        assert await task_queue_interface.get_task_status("t_id") == "unknown"
        assert "DistributedTaskQueueManager not available" in caplog.text
        caplog.clear()

        # FIX: The implementation returns None, not a RuntimeError.
        assert await task_queue_interface.get_task_result("id") is None
        assert "DistributedTaskQueueManager not available" in caplog.text
        caplog.clear()

        assert await task_queue_interface.revoke_task("t_id") is False
        assert "DistributedTaskQueueManager not available" in caplog.text