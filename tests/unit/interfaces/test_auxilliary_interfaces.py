### tests/unit/interfaces/test_auxilliary_interfaces.py
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.conversation.impl.manager import ConversationStateManager
from genie_tooling.conversation.types import ConversationState
from genie_tooling.hitl.manager import HITLManager
from genie_tooling.hitl.types import ApprovalRequest, ApprovalResponse
from genie_tooling.interfaces import (
    ConversationInterface,
    HITLInterface,
    ObservabilityInterface,
    PromptInterface,
    TaskQueueInterface,
    UsageTrackingInterface,
)
from genie_tooling.llm_providers.types import (
    ChatMessage,
)
from genie_tooling.observability.manager import InteractionTracingManager
from genie_tooling.prompts.manager import PromptManager
from genie_tooling.prompts.types import PromptData, PromptIdentifier
from genie_tooling.task_queues.manager import DistributedTaskQueueManager
from genie_tooling.token_usage.manager import TokenUsageManager
from genie_tooling.token_usage.types import TokenUsageRecord

logger = logging.getLogger(__name__)

# --- ObservabilityInterface Tests ---
@pytest.fixture()
def mock_tracing_manager_for_aux() -> MagicMock:
    mgr = MagicMock(spec=InteractionTracingManager)
    mgr.trace_event = AsyncMock()
    return mgr

@pytest.fixture()
def observability_interface(mock_tracing_manager_for_aux: MagicMock) -> ObservabilityInterface:
    return ObservabilityInterface(tracing_manager=mock_tracing_manager_for_aux)

@pytest.mark.asyncio()
async def test_observability_interface_trace_event(
    observability_interface: ObservabilityInterface, mock_tracing_manager_for_aux: MagicMock
):
    await observability_interface.trace_event("test.event", {"data": "value"}, "TestComponent", "corr-id-1")
    mock_tracing_manager_for_aux.trace_event.assert_awaited_once_with(
        "test.event", {"data": "value"}, "TestComponent", "corr-id-1"
    )

@pytest.mark.asyncio()
async def test_observability_interface_no_manager(observability_interface: ObservabilityInterface):
    observability_interface._tracing_manager = None # type: ignore
    # Should not raise, just be a no-op
    await observability_interface.trace_event("test.event", {}, "Comp", "id")


# --- HITLInterface Tests ---
@pytest.fixture()
def mock_hitl_manager_for_aux() -> MagicMock:
    mgr = MagicMock(spec=HITLManager)
    mgr.request_approval = AsyncMock(return_value=ApprovalResponse(request_id="req_aux", status="approved"))
    return mgr

@pytest.fixture()
def hitl_interface(mock_hitl_manager_for_aux: MagicMock) -> HITLInterface:
    return HITLInterface(hitl_manager=mock_hitl_manager_for_aux)

@pytest.mark.asyncio()
async def test_hitl_interface_request_approval(
    hitl_interface: HITLInterface, mock_hitl_manager_for_aux: MagicMock
):
    req_id_val = "req_aux_test_" + str(uuid.uuid4())
    request = ApprovalRequest(request_id=req_id_val, prompt="Approve aux?", data_to_approve={})
    response = await hitl_interface.request_approval(request, approver_id="custom_aux_approver")
    assert response["status"] == "approved"
    mock_hitl_manager_for_aux.request_approval.assert_awaited_once_with(request, "custom_aux_approver")

@pytest.mark.asyncio()
async def test_hitl_interface_no_manager(hitl_interface: HITLInterface):
    hitl_interface._hitl_manager = None # type: ignore
    req_id_val = "req_aux_no_mgr_" + str(uuid.uuid4())
    request = ApprovalRequest(request_id=req_id_val, prompt="Test", data_to_approve={})
    response = await hitl_interface.request_approval(request)
    assert response["status"] == "error"
    assert response["reason"] == "HITL system unavailable."
    assert response["request_id"] == req_id_val


# --- UsageTrackingInterface Tests ---
@pytest.fixture()
def mock_token_usage_manager_for_aux() -> MagicMock:
    mgr = MagicMock(spec=TokenUsageManager)
    mgr.record_usage = AsyncMock()
    mgr.get_summary = AsyncMock(return_value={"total_tokens_aux": 100})
    return mgr

@pytest.fixture()
def usage_tracking_interface(mock_token_usage_manager_for_aux: MagicMock) -> UsageTrackingInterface:
    return UsageTrackingInterface(token_usage_manager=mock_token_usage_manager_for_aux)

@pytest.mark.asyncio()
async def test_usage_interface_record_usage(
    usage_tracking_interface: UsageTrackingInterface, mock_token_usage_manager_for_aux: MagicMock
):
    record = TokenUsageRecord(provider_id="p_aux", model_name="m_aux", total_tokens=50)
    await usage_tracking_interface.record_usage(record)
    mock_token_usage_manager_for_aux.record_usage.assert_awaited_once_with(record)

@pytest.mark.asyncio()
async def test_usage_interface_get_summary(
    usage_tracking_interface: UsageTrackingInterface, mock_token_usage_manager_for_aux: MagicMock
):
    summary = await usage_tracking_interface.get_summary(recorder_id="rec_aux", filter_criteria={"user": "u_aux"})
    assert summary == {"total_tokens_aux": 100}
    mock_token_usage_manager_for_aux.get_summary.assert_awaited_once_with("rec_aux", {"user": "u_aux"})

@pytest.mark.asyncio()
async def test_usage_interface_no_manager(usage_tracking_interface: UsageTrackingInterface):
    usage_tracking_interface._token_usage_manager = None # type: ignore
    await usage_tracking_interface.record_usage(TokenUsageRecord(provider_id="p",model_name="m",total_tokens=1)) # No error
    summary = await usage_tracking_interface.get_summary()
    assert summary == {"error": "TokenUsageManager unavailable."}


# --- PromptInterface Tests ---
@pytest.fixture()
def mock_prompt_manager_for_aux() -> MagicMock:

    mgr = AsyncMock(spec=PromptManager)

    mgr.get_raw_template = AsyncMock(return_value="Raw template content aux")
    mgr.render_prompt = AsyncMock(return_value="Rendered prompt aux")
    mgr.render_chat_prompt = AsyncMock(return_value=[{"role": "user", "content": "Rendered chat aux"}])
    mgr.list_available_templates = AsyncMock(return_value=[PromptIdentifier(name="aux_prompt", version="v1")])
    return mgr

@pytest.fixture()
def prompt_interface(mock_prompt_manager_for_aux: MagicMock) -> PromptInterface:
    return PromptInterface(prompt_manager=mock_prompt_manager_for_aux)

@pytest.mark.asyncio()
async def test_prompt_interface_get_content(prompt_interface: PromptInterface, mock_prompt_manager_for_aux: MagicMock):
    content = await prompt_interface.get_prompt_template_content("name_aux", "v_aux", "reg_aux")
    assert content == "Raw template content aux"
    mock_prompt_manager_for_aux.get_raw_template.assert_awaited_once_with("name_aux", "v_aux", "reg_aux")

@pytest.mark.asyncio()
async def test_prompt_interface_render_prompt(prompt_interface: PromptInterface, mock_prompt_manager_for_aux: MagicMock):
    data: PromptData = {"key": "val_aux"}
    rendered = await prompt_interface.render_prompt("name_aux", data, template_content=None, version="v_aux", registry_id="reg_aux", template_engine_id="eng_aux")
    assert rendered == "Rendered prompt aux"
    mock_prompt_manager_for_aux.render_prompt.assert_awaited_once_with("name_aux", data, None, "v_aux", "reg_aux", "eng_aux")

@pytest.mark.asyncio()
async def test_prompt_interface_render_chat_prompt(prompt_interface: PromptInterface, mock_prompt_manager_for_aux: MagicMock):
    data: PromptData = {"key": "val_chat_aux"}
    chat_messages = await prompt_interface.render_chat_prompt("name_chat_aux", data, template_content=None, version="v_chat_aux", registry_id="reg_chat_aux", template_engine_id="eng_chat_aux")
    assert chat_messages == [{"role": "user", "content": "Rendered chat aux"}]
    mock_prompt_manager_for_aux.render_chat_prompt.assert_awaited_once_with("name_chat_aux", data, None, "v_chat_aux", "reg_chat_aux", "eng_chat_aux")

@pytest.mark.asyncio()
async def test_prompt_interface_list_templates(prompt_interface: PromptInterface, mock_prompt_manager_for_aux: MagicMock):
    templates = await prompt_interface.list_templates("reg_list_aux")
    assert len(templates) == 1
    assert templates[0]["name"] == "aux_prompt"
    mock_prompt_manager_for_aux.list_available_templates.assert_awaited_once_with("reg_list_aux")


# --- ConversationInterface Tests ---
@pytest.fixture()
def mock_conversation_manager_for_aux() -> MagicMock:
    mgr = MagicMock(spec=ConversationStateManager)
    mgr.load_state = AsyncMock(return_value=ConversationState(session_id="s_aux", history=[]))
    mgr.save_state = AsyncMock()
    mgr.add_message = AsyncMock()
    mgr.delete_state = AsyncMock(return_value=True)
    return mgr

@pytest.fixture()
def conversation_interface(mock_conversation_manager_for_aux: MagicMock) -> ConversationInterface:
    return ConversationInterface(conversation_manager=mock_conversation_manager_for_aux)

@pytest.mark.asyncio()
async def test_conversation_interface_load_state(
    conversation_interface: ConversationInterface, mock_conversation_manager_for_aux: MagicMock
):
    state = await conversation_interface.load_state("s_aux", "p_aux")
    assert state is not None
    assert state["session_id"] == "s_aux"
    mock_conversation_manager_for_aux.load_state.assert_awaited_once_with("s_aux", "p_aux")

@pytest.mark.asyncio()
async def test_conversation_interface_save_state(
    conversation_interface: ConversationInterface, mock_conversation_manager_for_aux: MagicMock
):
    state_to_save = ConversationState(session_id="s_save_aux", history=[])
    await conversation_interface.save_state(state_to_save, "p_save_aux")
    mock_conversation_manager_for_aux.save_state.assert_awaited_once_with(state_to_save, "p_save_aux")

@pytest.mark.asyncio()
async def test_conversation_interface_add_message(
    conversation_interface: ConversationInterface, mock_conversation_manager_for_aux: MagicMock
):
    msg: ChatMessage = {"role": "user", "content": "Hi aux"}
    await conversation_interface.add_message("s_add_aux", msg, "p_add_aux")
    mock_conversation_manager_for_aux.add_message.assert_awaited_once_with("s_add_aux", msg, "p_add_aux")

@pytest.mark.asyncio()
async def test_conversation_interface_delete_state(
    conversation_interface: ConversationInterface, mock_conversation_manager_for_aux: MagicMock
):
    result = await conversation_interface.delete_state("s_del_aux", "p_del_aux")
    assert result is True
    mock_conversation_manager_for_aux.delete_state.assert_awaited_once_with("s_del_aux", "p_del_aux")

@pytest.mark.asyncio()
async def test_conversation_interface_no_manager(conversation_interface: ConversationInterface):
    conversation_interface._conversation_manager = None # type: ignore
    assert await conversation_interface.load_state("s1") is None
    await conversation_interface.save_state(ConversationState(session_id="s1", history=[])) # No error
    await conversation_interface.add_message("s1", {"role": "user", "content": "test"}) # No error
    assert await conversation_interface.delete_state("s1") is False


# --- TaskQueueInterface Tests ---
@pytest.fixture()
def mock_task_queue_manager_for_aux() -> MagicMock:
    mgr = MagicMock(spec=DistributedTaskQueueManager)
    mgr.submit_task = AsyncMock(return_value="task_id_aux")
    mgr.get_task_status = AsyncMock(return_value="success")
    mgr.get_task_result = AsyncMock(return_value={"result": "done_aux"})
    mgr.revoke_task = AsyncMock(return_value=True)
    return mgr

@pytest.fixture()
def task_queue_interface(mock_task_queue_manager_for_aux: MagicMock) -> TaskQueueInterface:
    return TaskQueueInterface(task_queue_manager=mock_task_queue_manager_for_aux)

@pytest.mark.asyncio()
async def test_task_queue_interface_submit_task(
    task_queue_interface: TaskQueueInterface, mock_task_queue_manager_for_aux: MagicMock
):
    task_id = await task_queue_interface.submit_task("task_aux", args=(1,), kwargs={"k": "v_aux"}, queue_id="q_aux", task_options={"opt": True})
    assert task_id == "task_id_aux"
    mock_task_queue_manager_for_aux.submit_task.assert_awaited_once_with("task_aux", (1,), {"k": "v_aux"}, "q_aux", {"opt": True})

@pytest.mark.asyncio()
async def test_task_queue_interface_get_status(
    task_queue_interface: TaskQueueInterface, mock_task_queue_manager_for_aux: MagicMock
):
    status = await task_queue_interface.get_task_status("id_aux", "q_stat_aux")
    assert status == "success"
    mock_task_queue_manager_for_aux.get_task_status.assert_awaited_once_with("id_aux", "q_stat_aux")

@pytest.mark.asyncio()
async def test_task_queue_interface_get_result(
    task_queue_interface: TaskQueueInterface, mock_task_queue_manager_for_aux: MagicMock
):
    result = await task_queue_interface.get_task_result("id_res_aux", "q_res_aux", timeout_seconds=10.0)
    assert result == {"result": "done_aux"}
    mock_task_queue_manager_for_aux.get_task_result.assert_awaited_once_with("id_res_aux", "q_res_aux", 10.0)

@pytest.mark.asyncio()
async def test_task_queue_interface_revoke_task(
    task_queue_interface: TaskQueueInterface, mock_task_queue_manager_for_aux: MagicMock
):
    revoked = await task_queue_interface.revoke_task("id_rev_aux", "q_rev_aux", terminate=True)
    assert revoked is True
    mock_task_queue_manager_for_aux.revoke_task.assert_awaited_once_with("id_rev_aux", "q_rev_aux", True)

@pytest.mark.asyncio()
async def test_task_queue_interface_no_manager(task_queue_interface: TaskQueueInterface, caplog: pytest.LogCaptureFixture):
    task_queue_interface._task_queue_manager = None # type: ignore
    caplog.set_level(logging.ERROR) # To capture the error log

    assert await task_queue_interface.submit_task("t") is None
    assert "DistributedTaskQueueManager not available" in caplog.text
    caplog.clear()

    # The original test checked for "unknown", but the implementation returns None.
    # The fix is to make the implementation return the correct TaskStatus literal.
    # For now, I'll update the test to expect the literal, and correct the implementation.
    assert await task_queue_interface.get_task_status("t_id") == "unknown"
    assert "DistributedTaskQueueManager not available" in caplog.text
    caplog.clear()

    assert await task_queue_interface.get_task_result("id") is None
    assert "DistributedTaskQueueManager not available" in caplog.text
    caplog.clear()

    assert await task_queue_interface.revoke_task("id") is False
    assert "DistributedTaskQueueManager not available" in caplog.text
