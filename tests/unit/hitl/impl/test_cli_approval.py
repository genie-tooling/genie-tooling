### tests/unit/hitl/impl/test_cli_approval.py
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch # Ensure MagicMock is imported

import pytest
from genie_tooling.hitl.impl.cli_approval import CliApprovalPlugin
from genie_tooling.hitl.types import ApprovalRequest, ApprovalResponse

APPROVAL_PLUGIN_LOGGER_NAME = "genie_tooling.hitl.impl.cli_approval"


@pytest.fixture
async def cli_approval_plugin() -> CliApprovalPlugin:
    plugin = CliApprovalPlugin()
    await plugin.setup()
    return plugin


@pytest.mark.asyncio
# Patch 'builtins.input' which is called by asyncio.to_thread(input, ...)
@patch("builtins.input") 
async def test_request_approval_approved_no_reason(
    mock_builtin_input: MagicMock, cli_approval_plugin: CliApprovalPlugin # Changed mock name
):
    plugin = await cli_approval_plugin
    mock_builtin_input.side_effect = ["yes", ""]  # Approve, no reason

    request: ApprovalRequest = {
        "request_id": "req-approve-1",
        "prompt": "Do you approve this action?",
        "data_to_approve": {"action": "test_action"},
    }
    response = await plugin.request_approval(request)

    assert response["request_id"] == "req-approve-1"
    assert response["status"] == "approved"
    assert response["approver_id"] == "cli_user"
    assert response["reason"] is None
    assert mock_builtin_input.call_count == 2
    assert "Approve? (yes/no/y/n): " in mock_builtin_input.call_args_list[0].args[0]
    assert "Optional reason/comment for approval: " in mock_builtin_input.call_args_list[1].args[0]


@pytest.mark.asyncio
@patch("builtins.input")
async def test_request_approval_approved_with_reason(
    mock_builtin_input: MagicMock, cli_approval_plugin: CliApprovalPlugin
):
    plugin = await cli_approval_plugin
    mock_builtin_input.side_effect = ["y", "Looks good to me!"]

    request: ApprovalRequest = {
        "request_id": "req-approve-2",
        "prompt": "Approve data processing?",
        "data_to_approve": {"data_id": "xyz"},
        "context": {"user": "test_user"}
    }
    response = await plugin.request_approval(request)

    assert response["status"] == "approved"
    assert response["reason"] == "Looks good to me!"


@pytest.mark.asyncio
@patch("builtins.input")
async def test_request_approval_denied_with_reason(
    mock_builtin_input: MagicMock, cli_approval_plugin: CliApprovalPlugin
):
    plugin = await cli_approval_plugin
    mock_builtin_input.side_effect = ["no", "This is not safe."]

    request: ApprovalRequest = {
        "request_id": "req-deny-1",
        "prompt": "Proceed with deletion?",
        "data_to_approve": {"item_id": "item_to_delete"},
    }
    response = await plugin.request_approval(request)

    assert response["status"] == "denied"
    assert response["reason"] == "This is not safe."
    assert mock_builtin_input.call_count == 2
    assert "Reason for denial (required if not approved): " in mock_builtin_input.call_args_list[1].args[0]


@pytest.mark.asyncio
@patch("builtins.input")
async def test_request_approval_denied_empty_reason_then_provided(
    mock_builtin_input: MagicMock, cli_approval_plugin: CliApprovalPlugin
):
    plugin = await cli_approval_plugin
    mock_builtin_input.side_effect = ["n", "", "Needs more review."]

    request: ApprovalRequest = {"request_id": "req-deny-2", "prompt": "Confirm?", "data_to_approve": {}}
    response = await plugin.request_approval(request)

    assert response["status"] == "denied"
    assert response["reason"] == "Needs more review."
    assert mock_builtin_input.call_count == 3


@pytest.mark.asyncio
@patch("builtins.input") # Still patch input, as it's called within the task
@patch("asyncio.wait_for") # Now also patch asyncio.wait_for
async def test_request_approval_timeout(
    mock_asyncio_wait_for: AsyncMock, # Mock for asyncio.wait_for
    mock_builtin_input: MagicMock,    # Mock for builtins.input
    cli_approval_plugin: CliApprovalPlugin
):
    plugin = await cli_approval_plugin
    # Simulate asyncio.TimeoutError when asyncio.wait_for is called
    mock_asyncio_wait_for.side_effect = asyncio.TimeoutError

    request: ApprovalRequest = {
        "request_id": "req-timeout-1",
        "prompt": "Approve within 0.01 sec?",
        "data_to_approve": {},
        "timeout_seconds": 0.01 
    }
    response = await plugin.request_approval(request)

    assert response["status"] == "timeout"
    assert response["reason"] == "User did not respond in time."
    # asyncio.wait_for should have been called
    mock_asyncio_wait_for.assert_awaited_once()
    # builtins.input might or might not be called depending on how quickly wait_for raises Timeout.
    # For this test, focusing on the timeout status is key.


@pytest.mark.asyncio
@patch("builtins.input")
async def test_request_approval_input_raises_exception(
    mock_builtin_input: MagicMock, cli_approval_plugin: CliApprovalPlugin, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=APPROVAL_PLUGIN_LOGGER_NAME)
    plugin = await cli_approval_plugin
    mock_builtin_input.side_effect = EOFError("Simulated EOF during input")

    request: ApprovalRequest = {"request_id": "req-error-1", "prompt": "Test input error", "data_to_approve": {}}
    response = await plugin.request_approval(request)

    assert response["status"] == "error"
    assert "CLI prompt error: Simulated EOF during input" in response["reason"] # type: ignore
    assert f"{plugin.plugin_id}: Error during CLI approval prompt: Simulated EOF during input" in caplog.text


@pytest.mark.asyncio
@patch("builtins.input")
async def test_request_approval_no_timeout_specified(
    mock_builtin_input: MagicMock, cli_approval_plugin: CliApprovalPlugin
):
    plugin = await cli_approval_plugin
    with patch("asyncio.wait_for") as mock_wait_for:
        mock_builtin_input.side_effect = ["yes", ""]
        request: ApprovalRequest = {"request_id": "req-no-timeout", "prompt": "No timeout", "data_to_approve": {}}
        await plugin.request_approval(request)
        mock_wait_for.assert_not_called()


@pytest.mark.asyncio
@patch("builtins.input")
async def test_request_approval_zero_or_negative_timeout_is_no_timeout(
    mock_builtin_input: MagicMock, cli_approval_plugin: CliApprovalPlugin
):
    plugin = await cli_approval_plugin
    with patch("asyncio.wait_for") as mock_wait_for:
        mock_builtin_input.side_effect = ["y", "reason", "y", "reason2"] # Enough for two calls
        
        request_zero: ApprovalRequest = {"request_id": "req-zero-timeout", "prompt": "Zero", "data_to_approve": {}, "timeout_seconds": 0}
        await plugin.request_approval(request_zero)
        mock_wait_for.assert_not_called()

        request_neg: ApprovalRequest = {"request_id": "req-neg-timeout", "prompt": "Neg", "data_to_approve": {}, "timeout_seconds": -5}
        await plugin.request_approval(request_neg)
        # mock_wait_for should still not be called from the second request
        mock_wait_for.assert_not_called() # This checks total calls, which is correct


@pytest.mark.asyncio
async def test_setup_and_teardown_logging(cli_approval_plugin: CliApprovalPlugin, caplog: pytest.LogCaptureFixture):
    plugin = await cli_approval_plugin
    
    temp_plugin = CliApprovalPlugin()
    with caplog.at_level(logging.INFO, logger=APPROVAL_PLUGIN_LOGGER_NAME):
        await temp_plugin.setup()
    assert f"{temp_plugin.plugin_id}: Initialized. Will prompt on CLI for approvals." in caplog.text
    caplog.clear()

    with caplog.at_level(logging.DEBUG, logger=APPROVAL_PLUGIN_LOGGER_NAME):
        await plugin.teardown()
    assert f"{plugin.plugin_id}: Teardown complete." in caplog.text