import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.log_adapters.impl.default_adapter import (
    DEFAULT_LIBRARY_LOGGER_NAME,
    DefaultLogAdapter,
)
from genie_tooling.redactors.abc import Redactor
from genie_tooling.redactors.impl.noop_redactor import (
    NoOpRedactorPlugin,
)
from genie_tooling.redactors.impl.schema_aware import (
    REDACTION_PLACEHOLDER_VALUE,
    SchemaAwareRedactor,
)

ADAPTER_MODULE_LOGGER_NAME = "genie_tooling.log_adapters.impl.default_adapter"

@pytest.fixture
def mock_plugin_manager_for_adapter(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    pm.list_discovered_plugin_classes = MagicMock(return_value={})
    return pm

@pytest.fixture
async def default_log_adapter(
    mock_plugin_manager_for_adapter: PluginManager,
) -> DefaultLogAdapter:
    adapter = DefaultLogAdapter()
    mock_noop_redactor = NoOpRedactorPlugin()
    await mock_noop_redactor.setup()
    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = mock_noop_redactor
    await adapter.setup(
        config={"plugin_manager": mock_plugin_manager_for_adapter}
    )
    return adapter

@pytest.mark.asyncio
async def test_setup_default_config(
    mock_plugin_manager_for_adapter: PluginManager,
):
    adapter = DefaultLogAdapter()
    mock_noop_redactor = NoOpRedactorPlugin()
    await mock_noop_redactor.setup()
    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = mock_noop_redactor

    with patch(f"{ADAPTER_MODULE_LOGGER_NAME}.logging.getLogger") as mock_get_logger:
        mock_lib_logger_inst = MagicMock(spec=logging.Logger)
        mock_lib_logger_inst.handlers = []
        mock_get_logger.return_value = mock_lib_logger_inst

        await adapter.setup(
            config={"plugin_manager": mock_plugin_manager_for_adapter}
        )

        mock_get_logger.assert_any_call(DEFAULT_LIBRARY_LOGGER_NAME)
        mock_lib_logger_inst.setLevel.assert_called_with(logging.INFO)
        mock_lib_logger_inst.addHandler.assert_called_once()
        assert isinstance(adapter._redactor, NoOpRedactorPlugin)
        assert adapter._enable_schema_redaction is True
        assert adapter._enable_key_name_redaction is True

@pytest.mark.asyncio
async def test_setup_no_plugin_manager_uses_noop_redactor(
    caplog: pytest.LogCaptureFixture,
):
    adapter = DefaultLogAdapter()
    caplog.set_level(logging.WARNING, logger=ADAPTER_MODULE_LOGGER_NAME)
    await adapter.setup(config={})
    assert isinstance(adapter._redactor, NoOpRedactorPlugin)
    assert "PluginManager not provided in config" in caplog.text

@pytest.mark.asyncio
async def test_setup_custom_redactor_success(
    mock_plugin_manager_for_adapter: PluginManager,
):
    adapter = DefaultLogAdapter()
    mock_custom_redactor = SchemaAwareRedactor()
    await mock_custom_redactor.setup()
    mock_custom_redactor_id = "my_custom_redactor_v1"
    type(mock_custom_redactor).plugin_id = mock_custom_redactor_id # type: ignore

    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = mock_custom_redactor
    redactor_cfg = {"redact_matching_key_names": False}

    await adapter.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_adapter,
            "redactor_plugin_id": mock_custom_redactor_id,
            "redactor_config": redactor_cfg,
        }
    )
    assert adapter._redactor is mock_custom_redactor
    mock_plugin_manager_for_adapter.get_plugin_instance.assert_awaited_once_with(
        mock_custom_redactor_id, config={"plugin_manager": mock_plugin_manager_for_adapter, **redactor_cfg}
    )

@pytest.mark.asyncio
async def test_process_event_custom_redactor_called(
    mock_plugin_manager_for_adapter: PluginManager,
):
    adapter = DefaultLogAdapter()
    mock_custom_redactor_inst = SchemaAwareRedactor()
    await mock_custom_redactor_inst.setup()
    mock_custom_redactor_inst.sanitize = MagicMock(wraps=mock_custom_redactor_inst.sanitize)
    type(mock_custom_redactor_inst).plugin_id = "schema_aware_for_test" # type: ignore

    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = mock_custom_redactor_inst
    await adapter.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_adapter,
            "redactor_plugin_id": "schema_aware_for_test",
            "enable_schema_redaction": False,
        }
    )
    adapter._library_logger = MagicMock(spec=logging.Logger)

    data = {"api_token": "my_token"}
    schema = {"type": "object", "properties": {"api_token": {"type": "string", "format": "token"}}}
    await adapter.process_event("custom_redact_event", data, schema)

    mock_custom_redactor_inst.sanitize.assert_called_once_with(data, schema_hints=schema)
    logged_message = adapter._library_logger.info.call_args[0][0]
    logged_data = json.loads(logged_message.split("DATA: ")[1])
    assert logged_data["api_token"] == REDACTION_PLACEHOLDER_VALUE

@pytest.mark.asyncio
async def test_teardown_calls_redactor_teardown(
    mock_plugin_manager_for_adapter: PluginManager,
):
    adapter = DefaultLogAdapter()
    mock_custom_redactor_inst = AsyncMock(spec=Redactor)
    type(mock_custom_redactor_inst).plugin_id = "custom_teardown_redactor" # type: ignore

    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = mock_custom_redactor_inst
    await adapter.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_adapter,
            "redactor_plugin_id": "custom_teardown_redactor",
        }
    )
    assert adapter._redactor is mock_custom_redactor_inst

    await adapter.teardown()
    await mock_custom_redactor_inst.teardown()
    assert adapter._library_logger is None
    assert adapter._redactor is None
    assert adapter._plugin_manager is None

# Add other tests from the original file, ensuring they use the corrected setup and mocks.
# The key is to patch 'genie_tooling.log_adapters.impl.default_adapter.logging.getLogger'
# and to ensure the mocks are correctly configured and awaited.
