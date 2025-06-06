### tests/unit/observability/impl/test_console_tracer.py
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from genie_tooling.core.plugin_manager import PluginManager # For mocking PM if ConsoleTracer loads its own LogAdapter
from genie_tooling.log_adapters.abc import LogAdapter as LogAdapterPlugin
from genie_tooling.log_adapters.impl.default_adapter import DefaultLogAdapter
from genie_tooling.observability.impl.console_tracer import ConsoleTracerPlugin
from genie_tooling.observability.types import TraceEvent

TRACER_LOGGER_NAME = "genie_tooling.observability.impl.console_tracer"
# Get the actual logger instance that the module uses
module_logger_instance = logging.getLogger(TRACER_LOGGER_NAME)


@pytest.fixture
def mock_log_adapter_for_console_tracer() -> MagicMock:
    adapter = AsyncMock(spec=LogAdapterPlugin)
    adapter.plugin_id = "mock_log_adapter_for_console_tracer_v1"
    adapter.process_event = AsyncMock()
    return adapter

@pytest.fixture
def mock_plugin_manager_for_console_tracer_fallback(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    # Simulate DefaultLogAdapter being loadable
    async def get_instance_side_effect(plugin_id, config=None):
        if plugin_id == DefaultLogAdapter.plugin_id:
            # DefaultLogAdapter needs a PluginManager to load its redactor
            # For this test, we can assume it gets a NoOpRedactor or mock that part too.
            # Here, we provide a basic DefaultLogAdapter instance.
            dl_adapter = DefaultLogAdapter()
            # Its setup needs a PM. If the ConsoleTracer's PM is passed, it could create a loop.
            # So, the PM passed to DefaultLogAdapter should be a fresh one or carefully managed.
            # For simplicity, assume DefaultLogAdapter's setup handles PM absence for redactor.
            await dl_adapter.setup_logging(config={"plugin_manager": PluginManager()}) # Minimal setup
            return dl_adapter
        return None
    pm.get_plugin_instance = AsyncMock(side_effect=get_instance_side_effect)
    return pm


@pytest.fixture
async def console_tracer_with_mock_adapter(
    mock_log_adapter_for_console_tracer: MagicMock
) -> ConsoleTracerPlugin:
    tracer = ConsoleTracerPlugin()
    await tracer.setup(config={"log_adapter_instance_for_console_tracer": mock_log_adapter_for_console_tracer})
    return tracer

@pytest.fixture
async def console_tracer_fallback_to_default_adapter(
    mock_plugin_manager_for_console_tracer_fallback: PluginManager
) -> ConsoleTracerPlugin:
    tracer = ConsoleTracerPlugin()
    # Pass the PM that can load DefaultLogAdapter
    await tracer.setup(config={"plugin_manager_for_console_tracer": mock_plugin_manager_for_console_tracer_fallback})
    return tracer


@pytest.mark.asyncio
async def test_setup_with_provided_log_adapter(
    mock_log_adapter_for_console_tracer: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    tracer = ConsoleTracerPlugin()
    with caplog.at_level(logging.INFO, logger=TRACER_LOGGER_NAME):
        await tracer.setup(config={"log_adapter_instance_for_console_tracer": mock_log_adapter_for_console_tracer})

    assert tracer._log_adapter_to_use is mock_log_adapter_for_console_tracer
    assert f"{tracer.plugin_id}: Initialized. Will use LogAdapter '{mock_log_adapter_for_console_tracer.plugin_id}'" in caplog.text

@pytest.mark.asyncio
async def test_setup_fallback_to_default_log_adapter(
    console_tracer_fallback_to_default_adapter: ConsoleTracerPlugin, # Uses the fixture that sets up fallback
    caplog: pytest.LogCaptureFixture
):
    tracer = await console_tracer_fallback_to_default_adapter
    with caplog.at_level(logging.INFO, logger=TRACER_LOGGER_NAME):
        # Fixture already calls setup, so we check the state or re-log for assertion
        # For this test, we'll check the state and assume setup log was correct.
        # To re-log, we'd need to re-initialize and setup a new instance.
        pass # Setup is done by fixture

    assert isinstance(tracer._log_adapter_to_use, DefaultLogAdapter)
    # The log message about fallback success is now INFO level
    assert f"{tracer.plugin_id}: Successfully loaded fallback DefaultLogAdapter." in caplog.text


@pytest.mark.asyncio
async def test_setup_no_adapter_and_no_fallback_pm(caplog: pytest.LogCaptureFixture):
    tracer = ConsoleTracerPlugin()
    # Capture multiple levels to see the sequence
    with caplog.at_level(logging.DEBUG, logger=TRACER_LOGGER_NAME):
        await tracer.setup(config={}) # No adapter, no PM to load fallback

    assert tracer._log_adapter_to_use is None
    # Check for the sequence of logs
    assert any(
        rec.name == TRACER_LOGGER_NAME and
        rec.levelno == logging.WARNING and # First warning
        f"{tracer.plugin_id}: LogAdapter not provided via 'log_adapter_instance_for_console_tracer' in config." in rec.message
        for rec in caplog.records
    )
    assert any(
        rec.name == TRACER_LOGGER_NAME and
        rec.levelno == logging.ERROR and # Second is error
        f"{tracer.plugin_id}: PluginManager not available in config. Cannot load fallback LogAdapter." in rec.message
        for rec in caplog.records
    )
    assert any(
        rec.name == TRACER_LOGGER_NAME and
        rec.levelno == logging.WARNING and # Final fallback message
        f"{tracer.plugin_id}: No LogAdapter available. Falling back to direct logging" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_record_trace_uses_log_adapter(
    console_tracer_with_mock_adapter: ConsoleTracerPlugin,
    mock_log_adapter_for_console_tracer: MagicMock
):
    tracer = await console_tracer_with_mock_adapter
    event: TraceEvent = {
        "event_name": "user_login",
        "data": {"user_id": "123", "status": "success"},
        "timestamp": 1678886400.0,
        "component": "AuthService",
        "correlation_id": "corr-abc"
    }
    await tracer.record_trace(event)
    mock_log_adapter_for_console_tracer.process_event.assert_awaited_once_with(
        event_type="user_login",
        data=dict(event), # Ensure it's a plain dict
        schema_for_data=None
    )

@pytest.mark.asyncio
async def test_record_trace_fallback_direct_logging_if_no_adapter(caplog: pytest.LogCaptureFixture):
    tracer_no_adapter = ConsoleTracerPlugin()
    await tracer_no_adapter.setup(config={"log_level": "DEBUG"}) # Setup without providing adapter or PM
    assert tracer_no_adapter._log_adapter_to_use is None

    event: TraceEvent = {"event_name": "direct_log_test", "data": {"info": "test"}, "timestamp": 0.0}

    # Capture logs from the tracer's own logger at the configured level
    with caplog.at_level(logging.DEBUG, logger=TRACER_LOGGER_NAME):
        await tracer_no_adapter.record_trace(event)

    # Check if the fallback direct logging occurred
    direct_log_found = False
    for record in caplog.records:
        if record.name == TRACER_LOGGER_NAME and record.levelno == logging.DEBUG:
            if "CONSOLE_TRACE (direct) :: Event: direct_log_test" in record.message:
                direct_log_found = True
                break
    assert direct_log_found, f"Direct fallback log not found. Caplog: {caplog.text}"


@pytest.mark.asyncio
async def test_teardown(console_tracer_with_mock_adapter: ConsoleTracerPlugin, caplog: pytest.LogCaptureFixture):
    tracer = await console_tracer_with_mock_adapter
    with caplog.at_level(logging.DEBUG, logger=TRACER_LOGGER_NAME):
        await tracer.teardown()
    assert tracer._log_adapter_to_use is None
    assert f"{tracer.plugin_id}: Teardown complete." in caplog.text
