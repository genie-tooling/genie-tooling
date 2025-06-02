### tests/unit/hitl/test_hitl_manager.py
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.hitl.abc import HumanApprovalRequestPlugin
from genie_tooling.hitl.manager import HITLManager
from genie_tooling.hitl.types import ApprovalRequest, ApprovalResponse

MANAGER_LOGGER_NAME = "genie_tooling.hitl.manager"


@pytest.fixture
def mock_plugin_manager_for_hitl_mgr() -> MagicMock:
    pm = MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm


@pytest.fixture
def mock_default_approver_plugin() -> MagicMock:
    approver = AsyncMock(spec=HumanApprovalRequestPlugin)
    approver.plugin_id = "default_approver"
    approver.request_approval = AsyncMock(return_value=ApprovalResponse(request_id="default_req", status="approved"))
    approver.teardown = AsyncMock()
    return approver

@pytest.fixture
def mock_specific_approver_plugin() -> MagicMock:
    approver = AsyncMock(spec=HumanApprovalRequestPlugin)
    approver.plugin_id = "specific_approver"
    approver.request_approval = AsyncMock(return_value=ApprovalResponse(request_id="spec_req", status="denied", reason="Specific deny"))
    approver.teardown = AsyncMock()
    return approver


@pytest.fixture
def hitl_manager(
    mock_plugin_manager_for_hitl_mgr: MagicMock,
    mock_default_approver_plugin: MagicMock,
    mock_specific_approver_plugin: MagicMock,
) -> HITLManager:
    async def get_instance_side_effect(plugin_id, config=None):
        if plugin_id == "default_approver":
            return mock_default_approver_plugin
        if plugin_id == "specific_approver":
            return mock_specific_approver_plugin
        return None

    mock_plugin_manager_for_hitl_mgr.get_plugin_instance.side_effect = get_instance_side_effect
    return HITLManager(
        plugin_manager=mock_plugin_manager_for_hitl_mgr,
        default_approver_id="default_approver",
        approver_configurations={"default_approver": {"timeout": 30}, "specific_approver": {"custom_setting": "val"}},
    )


@pytest.mark.asyncio
async def test_get_default_approver_success(
    hitl_manager: HITLManager,
    mock_default_approver_plugin: MagicMock,
    mock_plugin_manager_for_hitl_mgr: MagicMock,
):
    approver = await hitl_manager._get_default_approver()
    assert approver is mock_default_approver_plugin
    mock_plugin_manager_for_hitl_mgr.get_plugin_instance.assert_any_call(
        "default_approver", config={"timeout": 30}
    )
    assert hitl_manager._initialized_default is True

    mock_plugin_manager_for_hitl_mgr.get_plugin_instance.reset_mock()
    approver2 = await hitl_manager._get_default_approver()
    assert approver2 is approver
    mock_plugin_manager_for_hitl_mgr.get_plugin_instance.assert_not_called()


@pytest.mark.asyncio
async def test_get_default_approver_not_configured(
    mock_plugin_manager_for_hitl_mgr: MagicMock, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger=MANAGER_LOGGER_NAME)
    manager = HITLManager(plugin_manager=mock_plugin_manager_for_hitl_mgr)
    approver = await manager._get_default_approver()
    assert approver is None
    assert "No default HITL approver configured." in caplog.text


@pytest.mark.asyncio
async def test_get_default_approver_load_fails(
    mock_plugin_manager_for_hitl_mgr: MagicMock, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=MANAGER_LOGGER_NAME)
    mock_plugin_manager_for_hitl_mgr.get_plugin_instance.side_effect = RuntimeError("Load failed")
    manager = HITLManager(
        plugin_manager=mock_plugin_manager_for_hitl_mgr,
        default_approver_id="error_approver"
    )
    approver = await manager._get_default_approver()
    assert approver is None
    assert "Error loading default HITL approver 'error_approver': Load failed" in caplog.text


@pytest.mark.asyncio
async def test_request_approval_uses_default(
    hitl_manager: HITLManager, mock_default_approver_plugin: MagicMock
):
    request = ApprovalRequest(request_id="req1", prompt="Approve?", data_to_approve={})
    response = await hitl_manager.request_approval(request)
    assert response["status"] == "approved"
    mock_default_approver_plugin.request_approval.assert_awaited_once_with(request)


@pytest.mark.asyncio
async def test_request_approval_uses_specific_id(
    hitl_manager: HITLManager, mock_specific_approver_plugin: MagicMock
):
    request = ApprovalRequest(request_id="req_spec", prompt="Specific approve?", data_to_approve={})
    response = await hitl_manager.request_approval(request, approver_id="specific_approver")

    assert response["status"] == "denied"
    assert response["reason"] == "Specific deny"

    mock_specific_approver_plugin.request_approval.assert_awaited_once_with(request)


@pytest.mark.asyncio
async def test_request_approval_no_approver_configured(
    mock_plugin_manager_for_hitl_mgr: MagicMock, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING, logger=MANAGER_LOGGER_NAME)
    manager = HITLManager(plugin_manager=mock_plugin_manager_for_hitl_mgr)
    request_id_val = "req_no_cfg_" + str(uuid.uuid4())
    request = ApprovalRequest(request_id=request_id_val, prompt="Test", data_to_approve={})
    response = await manager.request_approval(request)
    assert response["status"] == "denied"
    assert response["reason"] == "No HITL approver configured."
    assert response["request_id"] == request_id_val
    assert "HITL approval requested, but no approver ID specified and no default configured." in caplog.text


@pytest.mark.asyncio
async def test_request_approval_specific_approver_not_found(
    hitl_manager: HITLManager, caplog: pytest.LogCaptureFixture
):
    # The specific log we are looking for is a WARNING
    caplog.set_level(logging.WARNING, logger=MANAGER_LOGGER_NAME)
    await hitl_manager._get_default_approver()

    original_side_effect = hitl_manager._plugin_manager.get_plugin_instance.side_effect
    async def side_effect_not_found(plugin_id, config=None):
        if plugin_id == "non_existent_approver":
            return None
        if callable(original_side_effect):
            return await original_side_effect(plugin_id, config)
        return None
    hitl_manager._plugin_manager.get_plugin_instance.side_effect = side_effect_not_found

    request_id_val = "req_not_found_" + str(uuid.uuid4())
    request = ApprovalRequest(request_id=request_id_val, prompt="Test", data_to_approve={})
    response = await hitl_manager.request_approval(request, approver_id="non_existent_approver")

    assert response["status"] == "denied"
    assert response["reason"] == "HITL approver 'non_existent_approver' unavailable."
    assert response["request_id"] == request_id_val
    # Assert the WARNING log that explains why target_approver became None
    assert "Specified HITL approver 'non_existent_approver' not found or failed to load." in caplog.text
    hitl_manager._plugin_manager.get_plugin_instance.side_effect = original_side_effect


@pytest.mark.asyncio
async def test_request_approval_approver_raises_exception(
    hitl_manager: HITLManager, mock_default_approver_plugin: MagicMock, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=MANAGER_LOGGER_NAME)
    mock_default_approver_plugin.request_approval.side_effect = RuntimeError("Approval plugin crashed")
    request_id_val = "req_crash_" + str(uuid.uuid4())
    request = ApprovalRequest(request_id=request_id_val, prompt="Test crash", data_to_approve={})

    response = await hitl_manager.request_approval(request)

    assert response["status"] == "error"
    assert "Error in approval process: Approval plugin crashed" in response["reason"] # type: ignore
    assert response["request_id"] == request_id_val
    assert f"Error during HITL approval request with '{mock_default_approver_plugin.plugin_id}': Approval plugin crashed" in caplog.text


@pytest.mark.asyncio
async def test_teardown_calls_default_approver_teardown(
    hitl_manager: HITLManager, mock_default_approver_plugin: MagicMock
):
    await hitl_manager._get_default_approver()
    await hitl_manager.teardown()
    mock_default_approver_plugin.teardown.assert_awaited_once()
    assert hitl_manager._default_approver_instance is None
    assert hitl_manager._initialized_default is False


@pytest.mark.asyncio
async def test_teardown_default_approver_teardown_error(
    hitl_manager: HITLManager,
    mock_default_approver_plugin: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR, logger=MANAGER_LOGGER_NAME)
    await hitl_manager._get_default_approver()
    mock_default_approver_plugin.teardown.side_effect = RuntimeError("Approver teardown error")

    await hitl_manager.teardown()
    assert f"Error tearing down default HITL approver '{mock_default_approver_plugin.plugin_id}': Approver teardown error" in caplog.text
    assert hitl_manager._default_approver_instance is None
