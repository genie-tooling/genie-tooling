### tests/unit/observability/test_observability_manager.py
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin  # Import base Plugin
from genie_tooling.log_adapters.abc import LogAdapter as LogAdapterPlugin # ADDED
from genie_tooling.observability.abc import InteractionTracerPlugin
from genie_tooling.observability.manager import InteractionTracingManager
from genie_tooling.observability.types import TraceEvent

MANAGER_LOGGER_NAME = "genie_tooling.observability.manager"


@pytest.fixture
def mock_plugin_manager_for_obs_mgr() -> MagicMock:
    pm = MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm


@pytest.fixture
def mock_tracer_plugin() -> MagicMock:
    tracer = AsyncMock(spec=InteractionTracerPlugin)
    tracer.plugin_id = "mock_tracer_v1"
    tracer.record_trace = AsyncMock()
    tracer.teardown = AsyncMock()
    return tracer

@pytest.fixture
def mock_log_adapter_instance() -> MagicMock: # ADDED
    adapter = AsyncMock(spec=LogAdapterPlugin)
    adapter.plugin_id = "mock_log_adapter_for_obs_mgr_v1"
    adapter.process_event = AsyncMock()
    return adapter


@pytest.fixture
def tracing_manager(
    mock_plugin_manager_for_obs_mgr: MagicMock,
    mock_tracer_plugin: MagicMock,
    mock_log_adapter_instance: MagicMock, # ADDED
) -> InteractionTracingManager:
    async def get_instance_side_effect(plugin_id, config=None):
        if plugin_id == "mock_tracer_v1":
            # If ConsoleTracer is being tested, its config might now include log_adapter_instance
            if plugin_id == "console_tracer_plugin_v1" and config: # Example
                assert config.get("log_adapter_instance_for_console_tracer") is mock_log_adapter_instance
            return mock_tracer_plugin
        return None

    mock_plugin_manager_for_obs_mgr.get_plugin_instance.side_effect = get_instance_side_effect
    return InteractionTracingManager(
        plugin_manager=mock_plugin_manager_for_obs_mgr,
        default_tracer_ids=["mock_tracer_v1"],
        tracer_configurations={"mock_tracer_v1": {"endpoint": "http://localhost"}},
        log_adapter_instance=mock_log_adapter_instance # ADDED
    )


@pytest.mark.asyncio
async def test_initialize_tracers_success(
    tracing_manager: InteractionTracingManager,
    mock_tracer_plugin: MagicMock,
    mock_plugin_manager_for_obs_mgr: MagicMock,
    mock_log_adapter_instance: MagicMock, # ADDED
):
    # Simulate ConsoleTracer being the one configured to check log_adapter passing
    console_tracer_id = "console_tracer_plugin_v1"
    mock_console_tracer_plugin_instance = AsyncMock(spec=InteractionTracerPlugin)
    mock_console_tracer_plugin_instance.plugin_id = console_tracer_id

    async def get_instance_side_effect_console(plugin_id, config=None):
        if plugin_id == console_tracer_id:
            assert config is not None
            assert config.get("log_adapter_instance_for_console_tracer") is mock_log_adapter_instance
            # Ensure plugin_manager is also passed if ConsoleTracer needs to load its own fallback LogAdapter
            assert isinstance(config.get("plugin_manager_for_console_tracer"), PluginManager)
            return mock_console_tracer_plugin_instance
        return None
    mock_plugin_manager_for_obs_mgr.get_plugin_instance.side_effect = get_instance_side_effect_console
    tracing_manager_for_console_test = InteractionTracingManager(
        plugin_manager=mock_plugin_manager_for_obs_mgr,
        default_tracer_ids=[console_tracer_id],
        tracer_configurations={console_tracer_id: {"some_tracer_config": "val"}},
        log_adapter_instance=mock_log_adapter_instance
    )

    await tracing_manager_for_console_test._initialize_tracers()
    assert len(tracing_manager_for_console_test._active_tracers) == 1
    assert tracing_manager_for_console_test._active_tracers[0] is mock_console_tracer_plugin_instance
    mock_plugin_manager_for_obs_mgr.get_plugin_instance.assert_awaited_once_with(
        console_tracer_id, config={
            "some_tracer_config": "val",
            "log_adapter_instance_for_console_tracer": mock_log_adapter_instance,
            "plugin_manager_for_console_tracer": mock_plugin_manager_for_obs_mgr
            }
    )
    assert tracing_manager_for_console_test._initialized is True

    # Test re-initialization doesn't re-call get_plugin_instance
    mock_plugin_manager_for_obs_mgr.get_plugin_instance.reset_mock()
    await tracing_manager_for_console_test._initialize_tracers()
    mock_plugin_manager_for_obs_mgr.get_plugin_instance.assert_not_called()


@pytest.mark.asyncio
async def test_initialize_tracers_plugin_not_found(
    mock_plugin_manager_for_obs_mgr: MagicMock,
    mock_log_adapter_instance: MagicMock, # ADDED
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING, logger=MANAGER_LOGGER_NAME)
    mock_plugin_manager_for_obs_mgr.get_plugin_instance.return_value = None
    manager = InteractionTracingManager(
        plugin_manager=mock_plugin_manager_for_obs_mgr,
        default_tracer_ids=["non_existent_tracer"],
        log_adapter_instance=mock_log_adapter_instance # ADDED
    )
    await manager._initialize_tracers()
    assert len(manager._active_tracers) == 0
    assert "InteractionTracerPlugin 'non_existent_tracer' not found or failed to load." in caplog.text


@pytest.mark.asyncio
async def test_initialize_tracers_plugin_wrong_type(
    mock_plugin_manager_for_obs_mgr: MagicMock,
    mock_log_adapter_instance: MagicMock, # ADDED
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING, logger=MANAGER_LOGGER_NAME)

    class WrongTypeTracer(Plugin): # Is a Plugin, but not an InteractionTracerPlugin
        plugin_id = "wrong_type_tracer"
        description = "Not a tracer"
        async def setup(self, config=None): pass
        async def teardown(self): pass

    wrong_type_plugin_instance = WrongTypeTracer()
    mock_plugin_manager_for_obs_mgr.get_plugin_instance.return_value = wrong_type_plugin_instance

    manager = InteractionTracingManager(
        plugin_manager=mock_plugin_manager_for_obs_mgr,
        default_tracer_ids=["wrong_type_tracer"],
        log_adapter_instance=mock_log_adapter_instance # ADDED
    )
    await manager._initialize_tracers()
    assert len(manager._active_tracers) == 0
    assert "Plugin 'wrong_type_tracer' loaded but is not a valid InteractionTracerPlugin." in caplog.text


@pytest.mark.asyncio
async def test_initialize_tracers_load_error(
    mock_plugin_manager_for_obs_mgr: MagicMock,
    mock_log_adapter_instance: MagicMock, # ADDED
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=MANAGER_LOGGER_NAME)
    mock_plugin_manager_for_obs_mgr.get_plugin_instance.side_effect = RuntimeError("Load failed")
    manager = InteractionTracingManager(
        plugin_manager=mock_plugin_manager_for_obs_mgr,
        default_tracer_ids=["error_tracer"],
        log_adapter_instance=mock_log_adapter_instance # ADDED
    )
    await manager._initialize_tracers()
    assert len(manager._active_tracers) == 0
    assert "Error loading InteractionTracerPlugin 'error_tracer': Load failed" in caplog.text


@pytest.mark.asyncio
async def test_trace_event_no_active_tracers(
    mock_plugin_manager_for_obs_mgr: MagicMock,
    mock_log_adapter_instance: MagicMock, # ADDED
):
    mock_plugin_manager_for_obs_mgr.get_plugin_instance.return_value = None
    manager = InteractionTracingManager(
        plugin_manager=mock_plugin_manager_for_obs_mgr,
        log_adapter_instance=mock_log_adapter_instance # ADDED
    )
    await manager.trace_event(event_name="test", data={}, component="comp", correlation_id="cid")


@pytest.mark.asyncio
async def test_trace_event_success(
    tracing_manager: InteractionTracingManager, mock_tracer_plugin: MagicMock
):
    await tracing_manager._initialize_tracers()
    event_data = {"key": "value"}
    await tracing_manager.trace_event("my.event", event_data, "MyComponent", "corr-123")

    mock_tracer_plugin.record_trace.assert_awaited_once()
    called_event: TraceEvent = mock_tracer_plugin.record_trace.call_args[0][0]
    assert called_event["event_name"] == "my.event"
    assert called_event["data"] == event_data
    assert called_event["component"] == "MyComponent"
    assert called_event["correlation_id"] == "corr-123"
    assert "timestamp" in called_event


@pytest.mark.asyncio
async def test_trace_event_tracer_error(
    tracing_manager: InteractionTracingManager,
    mock_tracer_plugin: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR, logger=MANAGER_LOGGER_NAME)
    await tracing_manager._initialize_tracers()
    mock_tracer_plugin.record_trace.side_effect = RuntimeError("Trace recording failed")

    await tracing_manager.trace_event("error.event", {}, "ErrorComp", "err-cid")
    assert f"Error recording trace with tracer '{mock_tracer_plugin.plugin_id}': Trace recording failed" in caplog.text


@pytest.mark.asyncio
async def test_teardown_calls_tracer_teardown(
    tracing_manager: InteractionTracingManager, mock_tracer_plugin: MagicMock
):
    await tracing_manager._initialize_tracers()
    await tracing_manager.teardown()
    mock_tracer_plugin.teardown.assert_awaited_once()
    assert len(tracing_manager._active_tracers) == 0
    assert tracing_manager._initialized is False
    assert tracing_manager._log_adapter_instance is None # Check log adapter is cleared


@pytest.mark.asyncio
async def test_teardown_tracer_teardown_error(
    tracing_manager: InteractionTracingManager,
    mock_tracer_plugin: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR, logger=MANAGER_LOGGER_NAME)
    await tracing_manager._initialize_tracers()
    mock_tracer_plugin.teardown.side_effect = RuntimeError("Teardown error")

    await tracing_manager.teardown()
    assert f"Error tearing down tracer '{mock_tracer_plugin.plugin_id}': Teardown error" in caplog.text
    assert len(tracing_manager._active_tracers) == 0
