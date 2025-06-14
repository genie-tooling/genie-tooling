### tests/unit/token_usage/test_token_usage_manager.py
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin  # Import base Plugin
from genie_tooling.token_usage.abc import TokenUsageRecorderPlugin
from genie_tooling.token_usage.manager import TokenUsageManager
from genie_tooling.token_usage.types import TokenUsageRecord

MANAGER_LOGGER_NAME = "genie_tooling.token_usage.manager"


@pytest.fixture()
def mock_plugin_manager_for_token_mgr() -> MagicMock:
    pm = MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm


@pytest.fixture()
def mock_recorder_plugin() -> MagicMock:
    recorder = AsyncMock(spec=TokenUsageRecorderPlugin)
    recorder.plugin_id = "mock_recorder_v1"
    recorder.record_usage = AsyncMock()
    recorder.get_summary = AsyncMock(return_value={"total_tokens": 0})
    recorder.teardown = AsyncMock()
    return recorder


@pytest.fixture()
def token_usage_manager(
    mock_plugin_manager_for_token_mgr: MagicMock,
    mock_recorder_plugin: MagicMock,
) -> TokenUsageManager:
    async def get_instance_side_effect(plugin_id, config=None):
        if plugin_id == "mock_recorder_v1":
            return mock_recorder_plugin
        return None

    mock_plugin_manager_for_token_mgr.get_plugin_instance.side_effect = get_instance_side_effect
    return TokenUsageManager(
        plugin_manager=mock_plugin_manager_for_token_mgr,
        default_recorder_ids=["mock_recorder_v1"],
        recorder_configurations={"mock_recorder_v1": {"some_config": "value"}},
    )


@pytest.mark.asyncio()
async def test_initialize_recorders_success(
    token_usage_manager: TokenUsageManager,
    mock_recorder_plugin: MagicMock,
    mock_plugin_manager_for_token_mgr: MagicMock,
):
    await token_usage_manager._initialize_recorders()
    assert len(token_usage_manager._active_recorders) == 1
    assert token_usage_manager._active_recorders[0] is mock_recorder_plugin
    mock_plugin_manager_for_token_mgr.get_plugin_instance.assert_awaited_once_with(
        "mock_recorder_v1", config={"some_config": "value"}
    )
    assert token_usage_manager._initialized is True

    mock_plugin_manager_for_token_mgr.get_plugin_instance.reset_mock()
    await token_usage_manager._initialize_recorders()
    mock_plugin_manager_for_token_mgr.get_plugin_instance.assert_not_called()


@pytest.mark.asyncio()
async def test_initialize_recorders_plugin_not_found(
    mock_plugin_manager_for_token_mgr: MagicMock, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING, logger=MANAGER_LOGGER_NAME)
    mock_plugin_manager_for_token_mgr.get_plugin_instance.return_value = None
    manager = TokenUsageManager(
        plugin_manager=mock_plugin_manager_for_token_mgr,
        default_recorder_ids=["non_existent_recorder"],
    )
    await manager._initialize_recorders()
    assert len(manager._active_recorders) == 0
    assert "TokenUsageRecorderPlugin 'non_existent_recorder' not found or failed to load." in caplog.text


@pytest.mark.asyncio()
async def test_initialize_recorders_plugin_wrong_type(
    mock_plugin_manager_for_token_mgr: MagicMock, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING, logger=MANAGER_LOGGER_NAME)

    class WrongTypeRecorder(Plugin): # Is a Plugin, but not a TokenUsageRecorderPlugin
        plugin_id = "wrong_type_recorder"
        description = "Not a token recorder"
        async def setup(self, config=None): pass
        async def teardown(self): pass

    wrong_type_plugin_instance = WrongTypeRecorder()
    mock_plugin_manager_for_token_mgr.get_plugin_instance.return_value = wrong_type_plugin_instance

    manager = TokenUsageManager(
        plugin_manager=mock_plugin_manager_for_token_mgr,
        default_recorder_ids=["wrong_type_recorder"],
    )
    await manager._initialize_recorders()
    assert len(manager._active_recorders) == 0
    assert "Plugin 'wrong_type_recorder' loaded but is not a valid TokenUsageRecorderPlugin." in caplog.text


@pytest.mark.asyncio()
async def test_initialize_recorders_load_error(
    mock_plugin_manager_for_token_mgr: MagicMock, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=MANAGER_LOGGER_NAME)
    mock_plugin_manager_for_token_mgr.get_plugin_instance.side_effect = RuntimeError("Load failed")
    manager = TokenUsageManager(
        plugin_manager=mock_plugin_manager_for_token_mgr,
        default_recorder_ids=["error_recorder"],
    )
    await manager._initialize_recorders()
    assert len(manager._active_recorders) == 0
    assert "Error loading TokenUsageRecorderPlugin 'error_recorder': Load failed" in caplog.text


@pytest.mark.asyncio()
async def test_record_usage_no_active_recorders(
    mock_plugin_manager_for_token_mgr: MagicMock,
):
    mock_plugin_manager_for_token_mgr.get_plugin_instance.return_value = None
    manager = TokenUsageManager(plugin_manager=mock_plugin_manager_for_token_mgr)
    record = TokenUsageRecord(provider_id="p", model_name="m", total_tokens=10)
    await manager.record_usage(record)


@pytest.mark.asyncio()
async def test_record_usage_recorder_error(
    token_usage_manager: TokenUsageManager,
    mock_recorder_plugin: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR, logger=MANAGER_LOGGER_NAME)
    await token_usage_manager._initialize_recorders()
    mock_recorder_plugin.record_usage.side_effect = RuntimeError("Record failed")
    record = TokenUsageRecord(provider_id="p", model_name="m", total_tokens=10)
    await token_usage_manager.record_usage(record)
    assert f"Error recording token usage with recorder '{mock_recorder_plugin.plugin_id}': Record failed" in caplog.text


@pytest.mark.asyncio()
async def test_get_summary_specific_recorder(
    token_usage_manager: TokenUsageManager, mock_recorder_plugin: MagicMock
):
    await token_usage_manager._initialize_recorders()
    mock_recorder_plugin.get_summary.return_value = {"specific_total": 100}

    summary = await token_usage_manager.get_summary(recorder_id="mock_recorder_v1")
    assert summary == {"specific_total": 100}
    mock_recorder_plugin.get_summary.assert_awaited_once_with(None)


@pytest.mark.asyncio()
async def test_get_summary_all_recorders(
    token_usage_manager: TokenUsageManager, mock_recorder_plugin: MagicMock
):
    await token_usage_manager._initialize_recorders()
    mock_recorder_plugin.get_summary.return_value = {"total_from_mock": 50}

    mock_recorder2 = AsyncMock(spec=TokenUsageRecorderPlugin)
    mock_recorder2.plugin_id = "mock_recorder_v2"
    mock_recorder2.get_summary = AsyncMock(return_value={"total_from_mock2": 75})

    async def get_instance_side_effect_multi(plugin_id, config=None):
        if plugin_id == "mock_recorder_v1":
            return mock_recorder_plugin
        if plugin_id == "mock_recorder_v2":
            return mock_recorder2
        return None
    token_usage_manager._plugin_manager.get_plugin_instance.side_effect = get_instance_side_effect_multi # type: ignore
    token_usage_manager._default_recorder_ids = ["mock_recorder_v1", "mock_recorder_v2"]
    token_usage_manager._initialized = False

    summary_all = await token_usage_manager.get_summary(filter_criteria={"user": "test"})

    assert summary_all["mock_recorder_v1"] == {"total_from_mock": 50}
    assert summary_all["mock_recorder_v2"] == {"total_from_mock2": 75}
    mock_recorder_plugin.get_summary.assert_awaited_with({"user": "test"})
    mock_recorder2.get_summary.assert_awaited_with({"user": "test"})


@pytest.mark.asyncio()
async def test_get_summary_recorder_not_found(token_usage_manager: TokenUsageManager):
    await token_usage_manager._initialize_recorders()
    summary = await token_usage_manager.get_summary(recorder_id="non_existent_recorder")
    assert summary == {"error": "Recorder 'non_existent_recorder' not active or found."}


@pytest.mark.asyncio()
async def test_get_summary_no_active_recorders(mock_plugin_manager_for_token_mgr: MagicMock):
    mock_plugin_manager_for_token_mgr.get_plugin_instance.return_value = None
    manager = TokenUsageManager(plugin_manager=mock_plugin_manager_for_token_mgr)
    summary = await manager.get_summary()
    assert summary == {"error": "No active token usage recorders to query."}


@pytest.mark.asyncio()
async def test_get_summary_recorder_get_summary_error(
    token_usage_manager: TokenUsageManager,
    mock_recorder_plugin: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR, logger=MANAGER_LOGGER_NAME)
    await token_usage_manager._initialize_recorders()
    mock_recorder_plugin.get_summary.side_effect = RuntimeError("Summary retrieval failed")

    summary = await token_usage_manager.get_summary(recorder_id="mock_recorder_v1")
    assert summary == {"error": "Failed to get summary: Summary retrieval failed"}
    assert f"Error getting summary from recorder '{mock_recorder_plugin.plugin_id}': Summary retrieval failed" in caplog.text


@pytest.mark.asyncio()
async def test_teardown_calls_recorder_teardown(
    token_usage_manager: TokenUsageManager, mock_recorder_plugin: MagicMock
):
    await token_usage_manager._initialize_recorders()
    await token_usage_manager.teardown()
    mock_recorder_plugin.teardown.assert_awaited_once()
    assert len(token_usage_manager._active_recorders) == 0
    assert token_usage_manager._initialized is False


@pytest.mark.asyncio()
async def test_teardown_recorder_teardown_error(
    token_usage_manager: TokenUsageManager,
    mock_recorder_plugin: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR, logger=MANAGER_LOGGER_NAME)
    await token_usage_manager._initialize_recorders()
    mock_recorder_plugin.teardown.side_effect = RuntimeError("Teardown error")

    await token_usage_manager.teardown()
    assert f"Error tearing down token usage recorder '{mock_recorder_plugin.plugin_id}': Teardown error" in caplog.text
    assert len(token_usage_manager._active_recorders) == 0
