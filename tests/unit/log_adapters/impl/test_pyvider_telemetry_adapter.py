import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.log_adapters.impl.pyvider_telemetry_adapter import (
    PYVIDER_AVAILABLE,
    PyviderTelemetryLogAdapter,
)
from genie_tooling.redactors.abc import Redactor
from genie_tooling.redactors.impl.noop_redactor import NoOpRedactorPlugin

ADAPTER_LOGGER_NAME = "genie_tooling.log_adapters.impl.pyvider_telemetry_adapter"
PYVIDER_LOGGER_MODULE_PATH = "genie_tooling.log_adapters.impl.pyvider_telemetry_adapter.pyvider_global_logger"


@pytest.fixture()
def mock_pyvider_setup_telemetry():
    with patch("genie_tooling.log_adapters.impl.pyvider_telemetry_adapter.pyvider_setup_telemetry") as mock_setup:
        yield mock_setup

@pytest.fixture()
def mock_pyvider_shutdown_telemetry():
    with patch("genie_tooling.log_adapters.impl.pyvider_telemetry_adapter.pyvider_shutdown_telemetry", new_callable=AsyncMock) as mock_shutdown:
        yield mock_shutdown

@pytest.fixture()
def mock_pyvider_global_logger_instance():
    logger_instance = MagicMock(name="MockPyviderGlobalLoggerInstance")
    logger_instance.info = MagicMock()
    logger_instance.error = MagicMock()
    logger_instance.warning = MagicMock()
    logger_instance.debug = MagicMock()
    logger_instance.trace = MagicMock()
    return logger_instance

@pytest.fixture()
def mock_pyvider_logger_module(mock_pyvider_global_logger_instance):
    with patch(PYVIDER_LOGGER_MODULE_PATH, mock_pyvider_global_logger_instance) as mock_logger_obj:
        yield mock_logger_obj


@pytest.fixture()
def mock_plugin_manager_for_pyvider_adapter(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm

@pytest.fixture()
async def pyvider_adapter(
    mock_pyvider_setup_telemetry: MagicMock,
    mock_pyvider_logger_module: MagicMock,
    mock_plugin_manager_for_pyvider_adapter: PluginManager
) -> PyviderTelemetryLogAdapter:
    adapter = PyviderTelemetryLogAdapter()
    mock_noop_redactor = NoOpRedactorPlugin()
    await mock_noop_redactor.setup()
    mock_plugin_manager_for_pyvider_adapter.get_plugin_instance.return_value = mock_noop_redactor

    await adapter.setup({"plugin_manager": mock_plugin_manager_for_pyvider_adapter})
    return adapter


@pytest.mark.skipif(not PYVIDER_AVAILABLE, reason="Pyvider Telemetry not installed")
@pytest.mark.asyncio()
class TestPyviderTelemetryLogAdapter:
    async def test_process_event_redaction_error_logged(
        self,
        pyvider_adapter: PyviderTelemetryLogAdapter,
        mock_pyvider_global_logger_instance: MagicMock,
        caplog: pytest.LogCaptureFixture
    ):
        adapter = await pyvider_adapter
        caplog.set_level(logging.ERROR, logger=ADAPTER_LOGGER_NAME)


        adapter._enable_schema_redaction = True
        adapter._redactor = MagicMock(spec=Redactor)
        adapter._redactor.sanitize = MagicMock(side_effect=ValueError("Redaction boom!"))

        await adapter.process_event("event_redact_fail", {"data": "value"})

        assert "Error during custom redactor" in caplog.text
        assert "Redaction boom!" in caplog.text
        mock_pyvider_global_logger_instance.info.assert_called_once()
