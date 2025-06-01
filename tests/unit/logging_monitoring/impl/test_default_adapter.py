### tests/unit/logging_monitoring/impl/test_default_adapter.py
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin
from genie_tooling.log_adapters.impl.default_adapter import (
    DEFAULT_LIBRARY_LOGGER_NAME,
    DefaultLogAdapter,
)
from genie_tooling.redactors.abc import Redactor
from genie_tooling.redactors.impl.noop_redactor import NoOpRedactorPlugin
from genie_tooling.redactors.impl.schema_aware import (
    REDACTION_PLACEHOLDER_VALUE,
    SchemaAwareRedactor,
)

# Logger for the DefaultLogAdapter module itself
ADAPTER_MODULE_LOGGER_NAME = "genie_tooling.log_adapters.impl.default_adapter"


@pytest.fixture
def mock_plugin_manager_for_adapter(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    # Simulate that list_discovered_plugin_classes returns a dict
    # where keys are plugin_ids and values are the plugin classes.
    # This is important for how PluginManager.get_plugin_instance might work.
    pm.list_discovered_plugin_classes = MagicMock(return_value={})
    return pm


@pytest.fixture
async def default_log_adapter(
    mock_plugin_manager_for_adapter: PluginManager,
) -> DefaultLogAdapter:
    adapter = DefaultLogAdapter()
    # Default setup: uses NoOpRedactor
    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = (
        NoOpRedactorPlugin()
    )
    await adapter.setup_logging(
        config={"plugin_manager": mock_plugin_manager_for_adapter}
    )
    return adapter


# --- Test setup_logging ---
@pytest.mark.asyncio
async def test_setup_logging_default_config(
    mock_plugin_manager_for_adapter: PluginManager,
):
    adapter = DefaultLogAdapter()
    mock_noop_redactor = NoOpRedactorPlugin()
    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = (
        mock_noop_redactor
    )

    # Patch logging.getLogger to inspect calls
    with patch("logging.getLogger") as mock_get_logger:
        mock_lib_logger_inst = MagicMock(spec=logging.Logger)
        mock_lib_logger_inst.handlers = []  # Simulate no handlers initially
        mock_get_logger.side_effect = lambda name: (
            mock_lib_logger_inst
            if name == DEFAULT_LIBRARY_LOGGER_NAME
            else logging.getLogger(name)
        )

        await adapter.setup_logging(
            config={"plugin_manager": mock_plugin_manager_for_adapter}
        )

        mock_get_logger.assert_any_call(DEFAULT_LIBRARY_LOGGER_NAME)
        mock_lib_logger_inst.setLevel.assert_called_with(logging.INFO)
        mock_lib_logger_inst.addHandler.assert_called_once() # Default console handler
        assert adapter._redactor is mock_noop_redactor
        assert adapter._enable_schema_redaction is True
        assert adapter._enable_key_name_redaction is True


@pytest.mark.asyncio
async def test_setup_logging_custom_level_and_name(
    mock_plugin_manager_for_adapter: PluginManager,
):
    adapter = DefaultLogAdapter()
    custom_logger_name = "my_app.genie"
    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = (
        NoOpRedactorPlugin()
    )

    with patch("logging.getLogger") as mock_get_logger:
        mock_custom_logger_inst = MagicMock(spec=logging.Logger)
        mock_custom_logger_inst.handlers = []
        mock_get_logger.return_value = mock_custom_logger_inst

        await adapter.setup_logging(
            config={
                "plugin_manager": mock_plugin_manager_for_adapter,
                "log_level": "DEBUG",
                "library_logger_name": custom_logger_name,
            }
        )
        mock_get_logger.assert_any_call(custom_logger_name)
        mock_custom_logger_inst.setLevel.assert_called_with(logging.DEBUG)


@pytest.mark.asyncio
async def test_setup_logging_no_console_handler_added(
    mock_plugin_manager_for_adapter: PluginManager,
):
    adapter = DefaultLogAdapter()
    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = (
        NoOpRedactorPlugin()
    )
    with patch("logging.getLogger") as mock_get_logger:
        mock_lib_logger_inst = MagicMock(spec=logging.Logger)
        mock_lib_logger_inst.handlers = [MagicMock()]  # Simulate existing handler
        mock_get_logger.return_value = mock_lib_logger_inst

        await adapter.setup_logging(
            config={
                "plugin_manager": mock_plugin_manager_for_adapter,
                "add_console_handler_if_no_handlers": True, # Even if true, shouldn't add if one exists
            }
        )
        mock_lib_logger_inst.addHandler.assert_not_called()

        mock_lib_logger_inst.handlers = [] # Reset for next part
        await adapter.setup_logging(
            config={
                "plugin_manager": mock_plugin_manager_for_adapter,
                "add_console_handler_if_no_handlers": False, # Explicitly false
            }
        )
        mock_lib_logger_inst.addHandler.assert_not_called()


@pytest.mark.asyncio
async def test_setup_logging_no_plugin_manager_uses_noop_redactor(
    caplog: pytest.LogCaptureFixture,
):
    adapter = DefaultLogAdapter()
    caplog.set_level(logging.WARNING, logger=ADAPTER_MODULE_LOGGER_NAME)
    await adapter.setup_logging(config={})  # No plugin_manager
    assert isinstance(adapter._redactor, NoOpRedactorPlugin)
    assert (
        f"{adapter.plugin_id}: PluginManager not provided in config. "
        "Custom Redactor plugin cannot be loaded. Using NoOpRedactor."
    ) in caplog.text


@pytest.mark.asyncio
async def test_setup_logging_custom_redactor_success(
    mock_plugin_manager_for_adapter: PluginManager,
):
    adapter = DefaultLogAdapter()
    mock_custom_redactor = SchemaAwareRedactor()
    mock_custom_redactor_id = "my_custom_redactor_v1"
    type(mock_custom_redactor).plugin_id = mock_custom_redactor_id # type: ignore

    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = (
        mock_custom_redactor
    )
    redactor_cfg = {"redact_matching_key_names": False}

    await adapter.setup_logging(
        config={
            "plugin_manager": mock_plugin_manager_for_adapter,
            "redactor_plugin_id": mock_custom_redactor_id,
            "redactor_config": redactor_cfg,
        }
    )
    assert adapter._redactor is mock_custom_redactor
    mock_plugin_manager_for_adapter.get_plugin_instance.assert_awaited_once_with(
        mock_custom_redactor_id, config=redactor_cfg
    )


@pytest.mark.asyncio
async def test_setup_logging_custom_redactor_load_fails(
    mock_plugin_manager_for_adapter: PluginManager, caplog: pytest.LogCaptureFixture
):
    adapter = DefaultLogAdapter()
    caplog.set_level(logging.WARNING, logger=ADAPTER_MODULE_LOGGER_NAME)
    failing_redactor_id = "failing_redactor_v1"
    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = None

    await adapter.setup_logging(
        config={
            "plugin_manager": mock_plugin_manager_for_adapter,
            "redactor_plugin_id": failing_redactor_id,
        }
    )
    assert isinstance(adapter._redactor, NoOpRedactorPlugin)
    assert (
        f"{adapter.plugin_id}: Redactor plugin '{failing_redactor_id}' not found or invalid. "
        "Falling back to NoOpRedactor."
    ) in caplog.text


@pytest.mark.asyncio
async def test_setup_logging_custom_redactor_wrong_type(
    mock_plugin_manager_for_adapter: PluginManager, caplog: pytest.LogCaptureFixture
):
    adapter = DefaultLogAdapter()
    caplog.set_level(logging.WARNING, logger=ADAPTER_MODULE_LOGGER_NAME)
    wrong_type_plugin_id = "not_a_redactor_v1"
    # Simulate loading a plugin that isn't a Redactor
    class NotARedactor(Plugin):
        plugin_id = wrong_type_plugin_id
        description = "Not a redactor"
        async def setup(self, config=None): pass
        async def teardown(self): pass

    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = NotARedactor()

    await adapter.setup_logging(
        config={
            "plugin_manager": mock_plugin_manager_for_adapter,
            "redactor_plugin_id": wrong_type_plugin_id,
        }
    )
    assert isinstance(adapter._redactor, NoOpRedactorPlugin)
    assert (
        f"{adapter.plugin_id}: Redactor plugin '{wrong_type_plugin_id}' not found or invalid. "
        "Falling back to NoOpRedactor."
    ) in caplog.text


@pytest.mark.asyncio
async def test_setup_logging_schema_redaction_disabled(
    mock_plugin_manager_for_adapter: PluginManager,
):
    adapter = DefaultLogAdapter()
    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = (
        NoOpRedactorPlugin()
    )
    await adapter.setup_logging(
        config={
            "plugin_manager": mock_plugin_manager_for_adapter,
            "enable_schema_redaction": False,
            "enable_key_name_redaction": False,
        }
    )
    assert adapter._enable_schema_redaction is False
    assert adapter._enable_key_name_redaction is False


# --- Test process_event ---
@pytest.mark.asyncio
async def test_process_event_logger_not_initialized(
    caplog: pytest.LogCaptureFixture,
):
    adapter = DefaultLogAdapter()  # No setup_logging called
    caplog.set_level(logging.DEBUG, logger=ADAPTER_MODULE_LOGGER_NAME)
    await adapter.process_event("test_event", {"data": "value"})
    assert "EMERGENCY LOG (logger not init): EVENT: test_event" in caplog.text


@pytest.mark.asyncio
async def test_process_event_schema_redaction_works(
    mock_plugin_manager_for_adapter: PluginManager,
):
    adapter = DefaultLogAdapter()
    # Setup with default NoOpRedactor, but schema redaction enabled
    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = (
        NoOpRedactorPlugin()
    )
    await adapter.setup_logging(
        config={
            "plugin_manager": mock_plugin_manager_for_adapter,
            "enable_schema_redaction": True,
            "enable_key_name_redaction": True, # Also test key name redaction
        }
    )

    # Mock the library logger to capture its output
    mock_lib_logger = MagicMock(spec=logging.Logger)
    adapter._library_logger = mock_lib_logger

    data_to_redact = {"secret_key": "sensitive_value", "public_info": "safe"}
    schema = {
        "type": "object",
        "properties": {
            "secret_key": {"type": "string", "x-sensitive": True}, # Schema hint
            "public_info": {"type": "string"},
        },
    }
    await adapter.process_event("sensitive_event", data_to_redact, schema)

    mock_lib_logger.info.assert_called_once()
    logged_message = mock_lib_logger.info.call_args[0][0]
    assert "EVENT: sensitive_event" in logged_message
    # Check that 'secret_key' was redacted by schema_aware.py's logic
    # The exact string depends on sanitize_data_with_schema_based_rules
    logged_data_str = logged_message.split("DATA: ")[1]
    logged_data = json.loads(logged_data_str)
    assert logged_data["secret_key"] == REDACTION_PLACEHOLDER_VALUE
    assert logged_data["public_info"] == "safe"


@pytest.mark.asyncio
async def test_process_event_custom_redactor_called(
    mock_plugin_manager_for_adapter: PluginManager,
):
    adapter = DefaultLogAdapter()
    mock_custom_redactor_inst = SchemaAwareRedactor() # Use a real redactor
    mock_custom_redactor_inst.sanitize = MagicMock(wraps=mock_custom_redactor_inst.sanitize) # Spy on sanitize
    type(mock_custom_redactor_inst).plugin_id = "schema_aware_for_test" # type: ignore

    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = (
        mock_custom_redactor_inst
    )
    await adapter.setup_logging(
        config={
            "plugin_manager": mock_plugin_manager_for_adapter,
            "redactor_plugin_id": "schema_aware_for_test",
            "enable_schema_redaction": False, # Disable built-in to isolate custom
        }
    )
    adapter._library_logger = MagicMock(spec=logging.Logger) # Mock logger

    data = {"api_token": "my_token"}
    schema = {"type": "object", "properties": {"api_token": {"type": "string", "format": "token"}}}
    await adapter.process_event("custom_redact_event", data, schema)

    mock_custom_redactor_inst.sanitize.assert_called_once_with(data, schema_hints=schema)
    logged_message = adapter._library_logger.info.call_args[0][0]
    logged_data = json.loads(logged_message.split("DATA: ")[1])
    assert logged_data["api_token"] == REDACTION_PLACEHOLDER_VALUE


@pytest.mark.asyncio
async def test_process_event_schema_redaction_error_logged(
    default_log_adapter: DefaultLogAdapter, caplog: pytest.LogCaptureFixture
):
    adapter = await default_log_adapter # Fixture is async
    caplog.set_level(logging.ERROR, logger=ADAPTER_MODULE_LOGGER_NAME)
    adapter._library_logger = MagicMock(spec=logging.Logger) # Mock logger

    with patch(
        "genie_tooling.log_adapters.impl.default_adapter.sanitize_data_with_schema_based_rules",
        side_effect=ValueError("Schema redaction boom!"),
    ) as mock_sanitize:
        await adapter.process_event("event_schema_fail", {"data": "value"}, {})

    mock_sanitize.assert_called_once()
    assert "Error during schema-based redaction for event 'event_schema_fail'" in caplog.text
    assert "Schema redaction boom!" in caplog.text
    # Ensure event is still logged, possibly with unredacted data
    adapter._library_logger.info.assert_called_once()


@pytest.mark.asyncio
async def test_process_event_custom_redactor_error_logged(
    mock_plugin_manager_for_adapter: PluginManager, caplog: pytest.LogCaptureFixture
):
    adapter = DefaultLogAdapter()
    caplog.set_level(logging.ERROR, logger=ADAPTER_MODULE_LOGGER_NAME)

    mock_failing_redactor = MagicMock(spec=Redactor)
    mock_failing_redactor.sanitize.side_effect = RuntimeError("Custom redactor boom!")
    type(mock_failing_redactor).plugin_id = "failing_custom_redactor" # type: ignore

    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = (
        mock_failing_redactor
    )
    await adapter.setup_logging(
        config={
            "plugin_manager": mock_plugin_manager_for_adapter,
            "redactor_plugin_id": "failing_custom_redactor",
            "enable_schema_redaction": False, # Isolate custom redactor error
        }
    )
    adapter._library_logger = MagicMock(spec=logging.Logger) # Mock logger

    await adapter.process_event("event_custom_fail", {"data": "value"})

    mock_failing_redactor.sanitize.assert_called_once()
    assert "Error during custom Redactor plugin 'failing_custom_redactor'" in caplog.text
    assert "Custom redactor boom!" in caplog.text
    adapter._library_logger.info.assert_called_once() # Still logs


@pytest.mark.asyncio
async def test_process_event_json_dump_fallback(
    default_log_adapter: DefaultLogAdapter,
):
    adapter = await default_log_adapter
    adapter._library_logger = MagicMock(spec=logging.Logger)

    class NonJsonSerializable:
        def __str__(self):
            # This __str__ will be used by json.dumps(..., default=str)
            return "NonJsonSerializableViaDefaultStr"
        def __repr__(self):
            # This __repr__ will be used by the str(dict_containing_this_object) fallback
            return "<NonJsonSerializableObjectRepr>"


    data_unserializable = {"key": NonJsonSerializable()}

    # Test the primary path: json.dumps with default=str
    await adapter.process_event("unserializable_event_default_str", data_unserializable)
    adapter._library_logger.info.assert_called_once()
    logged_message_default_str = adapter._library_logger.info.call_args[0][0]
    assert "NonJsonSerializableViaDefaultStr" in logged_message_default_str
    adapter._library_logger.info.reset_mock() # Reset for next part of test

    # Test the fallback path: json.dumps itself raises an error
    with patch("json.dumps", side_effect=TypeError("Cannot serialize at all")):
        await adapter.process_event("unserializable_event_fallback", data_unserializable)

    adapter._library_logger.info.assert_called_once()
    logged_message_fallback = adapter._library_logger.info.call_args[0][0]
    # The str(dict_containing_object) will use the object's __repr__
    assert "<NonJsonSerializableObjectRepr>" in logged_message_fallback
    assert "..." in logged_message_fallback # Check truncation


@pytest.mark.asyncio
async def test_process_event_long_data_truncation(
    default_log_adapter: DefaultLogAdapter,
):
    adapter = await default_log_adapter
    adapter._library_logger = MagicMock(spec=logging.Logger)
    long_string = "a" * 3000
    data_long = {"long_field": long_string}

    await adapter.process_event("long_data_event", data_long)
    adapter._library_logger.info.assert_called_once()
    logged_message = adapter._library_logger.info.call_args[0][0]
    assert len(logged_message.split("DATA: ")[1]) <= 2000 + 3 # 2000 chars + "..."
    assert logged_message.endswith("...")


# --- Test teardown ---
@pytest.mark.asyncio
async def test_teardown_calls_redactor_teardown(
    mock_plugin_manager_for_adapter: PluginManager,
):
    adapter = DefaultLogAdapter()
    mock_custom_redactor_inst = AsyncMock(spec=Redactor) # Use AsyncMock for teardown
    type(mock_custom_redactor_inst).plugin_id = "custom_teardown_redactor" # type: ignore

    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = (
        mock_custom_redactor_inst
    )
    await adapter.setup_logging(
        config={
            "plugin_manager": mock_plugin_manager_for_adapter,
            "redactor_plugin_id": "custom_teardown_redactor",
        }
    )
    assert adapter._redactor is mock_custom_redactor_inst

    await adapter.teardown()
    mock_custom_redactor_inst.teardown.assert_awaited_once()
    assert adapter._library_logger is None
    assert adapter._redactor is None
    assert adapter._plugin_manager is None


@pytest.mark.asyncio
async def test_teardown_redactor_teardown_fails(
    mock_plugin_manager_for_adapter: PluginManager, caplog: pytest.LogCaptureFixture
):
    adapter = DefaultLogAdapter()
    caplog.set_level(logging.ERROR, logger=ADAPTER_MODULE_LOGGER_NAME)

    mock_failing_teardown_redactor = AsyncMock(spec=Redactor)
    mock_failing_teardown_redactor.teardown.side_effect = ValueError("Redactor teardown boom!")
    type(mock_failing_teardown_redactor).plugin_id = "failing_td_redactor" # type: ignore

    mock_plugin_manager_for_adapter.get_plugin_instance.return_value = (
        mock_failing_teardown_redactor
    )
    await adapter.setup_logging(
        config={
            "plugin_manager": mock_plugin_manager_for_adapter,
            "redactor_plugin_id": "failing_td_redactor",
        }
    )
    await adapter.teardown()
    assert "Error tearing down redactor 'failing_td_redactor'" in caplog.text
    assert "Redactor teardown boom!" in caplog.text
    assert adapter._redactor is None # Should still be nulled
