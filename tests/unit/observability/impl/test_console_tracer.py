import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.log_adapters.abc import LogAdapter as LogAdapterPlugin
from genie_tooling.log_adapters.impl.default_adapter import DefaultLogAdapter
from genie_tooling.observability.impl.console_tracer import ConsoleTracerPlugin

TRACER_LOGGER_NAME = "genie_tooling.observability.impl.console_tracer"

@pytest.fixture
def mock_log_adapter_for_console_tracer() -> MagicMock:
    adapter = AsyncMock(spec=LogAdapterPlugin)
    adapter.plugin_id = "mock_log_adapter_for_console_tracer_v1"
    adapter.process_event = AsyncMock()
    return adapter

@pytest.mark.asyncio
async def test_setup_no_adapter_and_no_fallback_pm(caplog: pytest.LogCaptureFixture):
    tracer = ConsoleTracerPlugin()
    with caplog.at_level(logging.WARNING, logger=TRACER_LOGGER_NAME):
        await tracer.setup(config={})
    assert tracer._log_adapter_to_use is None
    assert "LogAdapter not provided" in caplog.text
    assert "PluginManager not available in config" in caplog.text
    assert "No LogAdapter available. Falling back to direct logging" in caplog.text

@pytest.mark.asyncio
async def test_setup_fallback_to_default_log_adapter(caplog: pytest.LogCaptureFixture):
    tracer = ConsoleTracerPlugin()
    mock_pm = MagicMock(spec=PluginManager)
    mock_default_adapter = AsyncMock(spec=DefaultLogAdapter)
    mock_default_adapter.plugin_id = "default_log_adapter_v1"
    mock_pm.get_plugin_instance = AsyncMock(return_value=mock_default_adapter)

    with caplog.at_level(logging.INFO, logger=TRACER_LOGGER_NAME):
        await tracer.setup(config={"plugin_manager_for_console_tracer": mock_pm})

    assert tracer._log_adapter_to_use is mock_default_adapter
    assert "Successfully loaded fallback DefaultLogAdapter" in caplog.text
