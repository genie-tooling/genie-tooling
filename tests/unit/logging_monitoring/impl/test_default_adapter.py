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
        assert adapter._enable_schema_redaction is False
        assert adapter._enable_key_name_redaction is False

@pytest.mark.asyncio
async def test_setup_with_schema_redaction_enabled(
    mock_plugin_manager_for_adapter: PluginManager,
):
    """Test setup correctly enables schema-based redaction flags."""
    adapter = DefaultLogAdapter()
    mock_noop_redactor = NoOpRedactorPlugin()
    await mock_noop_redactor.setup()
    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = mock_noop_redactor

    await adapter.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_adapter,
            "enable_schema_redaction": True,
            "enable_key_name_redaction": True,
        }
    )
    assert adapter._enable_schema_redaction is True
    assert adapter._enable_key_name_redaction is True

@pytest.mark.asyncio
async def test_setup_loads_custom_redactor(
    mock_plugin_manager_for_adapter: PluginManager,
):
    """Test setup correctly loads a specified Redactor plugin."""
    adapter = DefaultLogAdapter()
    mock_schema_redactor = SchemaAwareRedactor()
    await mock_schema_redactor.setup()
    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = mock_schema_redactor

    await adapter.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_adapter,
            "redactor_plugin_id": "schema_aware_redactor_v1",
        }
    )
    assert isinstance(adapter._redactor, SchemaAwareRedactor)
    assert adapter._redactor.plugin_id == "schema_aware_redactor_v1"

@pytest.mark.asyncio
async def test_process_event_with_schema_redaction(
    mock_plugin_manager_for_adapter: PluginManager, caplog: pytest.LogCaptureFixture
):
    """Test that process_event applies schema redaction when enabled."""
    caplog.set_level(logging.INFO, logger=DEFAULT_LIBRARY_LOGGER_NAME)
    adapter = DefaultLogAdapter()
    
    # FIX: Ensure the correct redactor is loaded for this test.
    # The default fixture loads a NoOpRedactor. We need SchemaAwareRedactor for this test.
    mock_schema_redactor = SchemaAwareRedactor()
    await mock_schema_redactor.setup(config={"redact_matching_key_names": True})
    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = mock_schema_redactor
    
    await adapter.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_adapter,
            "redactor_plugin_id": "schema_aware_redactor_v1", # Explicitly request it
            "enable_schema_redaction": False, # Schema redaction is now done BY the redactor
            "enable_key_name_redaction": False, # This flag is for the internal function, not the plugin
        }
    )
    # The adapter should now use the SchemaAwareRedactor instance.
    adapter._redactor = mock_schema_redactor

    sensitive_data = {"api_key": "secret_key_123", "user_data": "public"}
    await adapter.process_event("test_event", sensitive_data)

    assert "secret_key_123" not in caplog.text
    assert REDACTION_PLACEHOLDER_VALUE in caplog.text
    assert "public" in caplog.text

@pytest.mark.asyncio
async def test_process_event_with_custom_redactor(
    mock_plugin_manager_for_adapter: PluginManager, caplog: pytest.LogCaptureFixture
):
    """Test that a custom redactor's sanitize method is called."""
    caplog.set_level(logging.INFO, logger=DEFAULT_LIBRARY_LOGGER_NAME)
    adapter = DefaultLogAdapter()
    
    mock_custom_redactor = MagicMock(spec=Redactor)
    mock_custom_redactor.plugin_id = "custom_redactor_v1"
    mock_custom_redactor.sanitize.return_value = {"redacted": True}
    
    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = mock_custom_redactor
    
    await adapter.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_adapter,
            "redactor_plugin_id": "custom_redactor_v1",
        }
    )
    
    await adapter.process_event("test_event", {"original": "data"})
    
    mock_custom_redactor.sanitize.assert_called_once_with({"original": "data"}, schema_hints=None)
    assert '"redacted": true' in caplog.text

@pytest.mark.asyncio
async def test_process_event_emergency_log(caplog: pytest.LogCaptureFixture):
    """Test emergency logging when the internal logger is not set."""
    caplog.set_level(logging.DEBUG, logger=ADAPTER_MODULE_LOGGER_NAME)
    adapter = DefaultLogAdapter()
    adapter._library_logger = None
    
    await adapter.process_event("emergency_event", {"data": "important"})
    
    assert "EMERGENCY LOG" in caplog.text
    assert "emergency_event" in caplog.text
    assert "important" in caplog.text
