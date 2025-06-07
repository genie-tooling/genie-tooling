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


@pytest.fixture
def mock_pyvider_setup_telemetry():
    with patch("genie_tooling.log_adapters.impl.pyvider_telemetry_adapter.pyvider_setup_telemetry") as mock_setup:
        yield mock_setup

@pytest.fixture
def mock_pyvider_shutdown_telemetry():
    with patch("genie_tooling.log_adapters.impl.pyvider_telemetry_adapter.pyvider_shutdown_telemetry", new_callable=AsyncMock) as mock_shutdown:
        yield mock_shutdown

@pytest.fixture
def mock_pyvider_global_logger_instance():
    logger_instance = MagicMock(name="MockPyviderGlobalLoggerInstance")
    logger_instance.info = MagicMock()
    logger_instance.error = MagicMock()
    logger_instance.warning = MagicMock()
    logger_instance.debug = MagicMock()
    return logger_instance

@pytest.fixture
def mock_pyvider_logger_module(mock_pyvider_global_logger_instance):
    # This fixture patches the 'pyvider_global_logger' object *within the adapter's module*
    with patch(PYVIDER_LOGGER_MODULE_PATH, mock_pyvider_global_logger_instance) as mock_logger_obj:
        yield mock_logger_obj


@pytest.fixture
def mock_plugin_manager_for_pyvider_adapter(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm

@pytest.fixture
async def pyvider_adapter(
    mock_pyvider_setup_telemetry: MagicMock, # Ensure Pyvider setup is mocked
    mock_pyvider_logger_module: MagicMock, # Ensure logger is mocked
    mock_plugin_manager_for_pyvider_adapter: PluginManager
) -> PyviderTelemetryLogAdapter:
    adapter = PyviderTelemetryLogAdapter()
    # Simulate NoOpRedactor loading successfully if no specific redactor is configured
    mock_noop_redactor = NoOpRedactorPlugin()
    await mock_noop_redactor.setup()
    mock_plugin_manager_for_pyvider_adapter.get_plugin_instance.return_value = mock_noop_redactor

    await adapter.setup({"plugin_manager": mock_plugin_manager_for_pyvider_adapter})
    return adapter


@pytest.mark.skipif(not PYVIDER_AVAILABLE, reason="Pyvider Telemetry not installed")
@pytest.mark.asyncio
class TestPyviderTelemetryLogAdapter:
    async def test_setup_calls_pyvider_setup(
        self,
        mock_pyvider_setup_telemetry: MagicMock,
        mock_pyvider_logger_module: MagicMock,
        mock_plugin_manager_for_pyvider_adapter: PluginManager
    ):
        adapter = PyviderTelemetryLogAdapter()
        test_service_name = "TestGenieApp"
        test_default_level = "INFO"

        await adapter.setup({
            "plugin_manager": mock_plugin_manager_for_pyvider_adapter,
            "service_name": test_service_name,
            "default_level": test_default_level,
            "console_formatter": "json"
        })

        assert adapter._is_setup_successful is True
        assert adapter._pyvider_logger is mock_pyvider_logger_module
        mock_pyvider_setup_telemetry.assert_called_once()
        telemetry_config_arg = mock_pyvider_setup_telemetry.call_args[1]["config"]
        assert telemetry_config_arg.service_name == test_service_name
        assert telemetry_config_arg.logging.default_level == test_default_level
        assert telemetry_config_arg.logging.console_formatter == "json"

    async def test_setup_loads_default_redactor(
        self,
        pyvider_adapter: PyviderTelemetryLogAdapter # Uses the fixture that sets up NoOpRedactor
    ):
        adapter = await pyvider_adapter
        assert isinstance(adapter._redactor, NoOpRedactorPlugin)

    async def test_setup_loads_custom_redactor(
        self,
        mock_pyvider_setup_telemetry: MagicMock,
        mock_pyvider_logger_module: MagicMock,
        mock_plugin_manager_for_pyvider_adapter: PluginManager
    ):
        adapter = PyviderTelemetryLogAdapter()
        mock_custom_redactor = AsyncMock(spec=Redactor)
        mock_custom_redactor.plugin_id = "my_custom_redactor_v1"
        mock_custom_redactor.sanitize = MagicMock(side_effect=lambda data, schema_hints=None: {"redacted_by_custom": True, **data})

        async def get_redactor_side_effect(plugin_id, config=None):
            if plugin_id == "my_custom_redactor_v1":
                await mock_custom_redactor.setup(config)
                return mock_custom_redactor
            return None
        mock_plugin_manager_for_pyvider_adapter.get_plugin_instance.side_effect = get_redactor_side_effect

        await adapter.setup({
            "plugin_manager": mock_plugin_manager_for_pyvider_adapter,
            "redactor_plugin_id": "my_custom_redactor_v1",
            "redactor_config": {"custom_setting": "abc"}
        })
        assert adapter._redactor is mock_custom_redactor
        mock_custom_redactor.setup.assert_awaited_with({"custom_setting": "abc"})


    async def test_process_event_logs_with_pyvider(
        self,
        pyvider_adapter: PyviderTelemetryLogAdapter,
        mock_pyvider_global_logger_instance: MagicMock # Use the instance directly for assertions
    ):
        adapter = await pyvider_adapter # Adapter is already setup by fixture
        event_type = "test.event.info"
        data = {"key": "value", "sensitive": "secret_data"} # No "component" key
        schema = {"type": "object", "properties": {"sensitive": {"type": "string", "x-sensitive": True}}}

        await adapter.process_event(event_type, data, schema_for_data=schema)

        mock_pyvider_global_logger_instance.info.assert_called_once()
        call_args = mock_pyvider_global_logger_instance.info.call_args[1] # kwargs
        assert call_args["key"] == "value"
        assert call_args["sensitive"] == "[REDACTED]" # Assuming schema redaction works
        assert call_args.get("domain") is None # CORRECTED: domain should be None if no component

    async def test_process_event_maps_to_das_fields(
        self,
        pyvider_adapter: PyviderTelemetryLogAdapter,
        mock_pyvider_global_logger_instance: MagicMock
    ):
        adapter = await pyvider_adapter
        await adapter.process_event("tool.execute.success", {"component": "MyTool", "status_override": "complete"})
        mock_pyvider_global_logger_instance.info.assert_called_once()
        call_kwargs = mock_pyvider_global_logger_instance.info.call_args[1]
        assert call_kwargs["domain"] == "mytool"
        assert call_kwargs["action"] == "complete" # From status_override
        assert call_kwargs["status"] == "success" # From event_type

    async def test_process_event_redaction_error_logged(
        self,
        pyvider_adapter: PyviderTelemetryLogAdapter,
        mock_pyvider_global_logger_instance: MagicMock,
        caplog: pytest.LogCaptureFixture
    ):
        adapter = await pyvider_adapter
        caplog.set_level(logging.ERROR, logger=ADAPTER_LOGGER_NAME)
        adapter._redactor.sanitize = MagicMock(side_effect=ValueError("Redaction boom!")) # type: ignore

        await adapter.process_event("event_redact_fail", {"data": "value"})
        assert "Error during custom redactor" in caplog.text
        assert "Redaction boom!" in caplog.text
        # Ensure Pyvider logger was still called (with potentially unredacted data)
        mock_pyvider_global_logger_instance.info.assert_called_once()

    async def test_process_event_pyvider_log_fails(
        self,
        pyvider_adapter: PyviderTelemetryLogAdapter,
        mock_pyvider_global_logger_instance: MagicMock,
        caplog: pytest.LogCaptureFixture
    ):
        adapter = await pyvider_adapter
        caplog.set_level(logging.ERROR) # Capture ERROR from adapter's own logger
        mock_pyvider_global_logger_instance.info.side_effect = Exception("Pyvider logging crashed")

        await adapter.process_event("event_pyvider_fail", {"data": "value"})
        # Check the fallback log from PyviderTelemetryLogAdapter's own logger
        assert "Error logging event 'event_pyvider_fail' with Pyvider: Pyvider logging crashed" in caplog.text


    async def test_teardown_calls_pyvider_shutdown(
        self,
        pyvider_adapter: PyviderTelemetryLogAdapter,
        mock_pyvider_shutdown_telemetry: AsyncMock
    ):
        adapter = await pyvider_adapter
        await adapter.teardown()
        mock_pyvider_shutdown_telemetry.assert_awaited_once()
        assert adapter._pyvider_logger is None
        assert adapter._redactor is None # Assuming default NoOpRedactor was loaded

    async def test_process_event_not_setup(
        self,
        caplog: pytest.LogCaptureFixture
    ):
        adapter = PyviderTelemetryLogAdapter() # Not setup
        caplog.set_level(logging.ERROR, logger=ADAPTER_LOGGER_NAME)
        await adapter.process_event("test", {})
        assert f"{adapter.plugin_id}: Not properly set up or Pyvider logger unavailable." in caplog.text

    @pytest.mark.parametrize("pyvider_available_runtime", [True, False])
    async def test_setup_pyvider_availability_check(
        self,
        pyvider_available_runtime: bool,
        mock_plugin_manager_for_pyvider_adapter: PluginManager,
        caplog: pytest.LogCaptureFixture
    ):
        adapter = PyviderTelemetryLogAdapter()
        with patch("genie_tooling.log_adapters.impl.pyvider_telemetry_adapter.PYVIDER_AVAILABLE", pyvider_available_runtime), \
             patch("genie_tooling.log_adapters.impl.pyvider_telemetry_adapter.pyvider_setup_telemetry") as mock_setup_call:

            if not pyvider_available_runtime:
                caplog.set_level(logging.ERROR, logger=ADAPTER_LOGGER_NAME)
                await adapter.setup({"plugin_manager": mock_plugin_manager_for_pyvider_adapter})
                assert f"{adapter.plugin_id}: Pyvider telemetry library not available. Cannot setup." in caplog.text
                mock_setup_call.assert_not_called()
            else: # Pyvider is available
                caplog.set_level(logging.INFO, logger=ADAPTER_LOGGER_NAME)
                # Ensure NoOpRedactor is loaded if no redactor_plugin_id is specified
                mock_noop_redactor = NoOpRedactorPlugin()
                await mock_noop_redactor.setup()
                mock_plugin_manager_for_pyvider_adapter.get_plugin_instance.return_value = mock_noop_redactor

                await adapter.setup({"plugin_manager": mock_plugin_manager_for_pyvider_adapter})
                assert f"{adapter.plugin_id}: Pyvider telemetry setup complete." in caplog.text
                mock_setup_call.assert_called_once()
