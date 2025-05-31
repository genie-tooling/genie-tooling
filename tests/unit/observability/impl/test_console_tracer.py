### tests/unit/observability/impl/test_console_tracer.py
import json
import logging
from unittest.mock import patch

import pytest
from genie_tooling.observability.impl.console_tracer import ConsoleTracerPlugin
from genie_tooling.observability.types import TraceEvent

TRACER_LOGGER_NAME = "genie_tooling.observability.impl.console_tracer"
# Get the actual logger instance that the module uses
module_logger_instance = logging.getLogger(TRACER_LOGGER_NAME)


@pytest.fixture
async def console_tracer() -> ConsoleTracerPlugin:
    tracer = ConsoleTracerPlugin()
    # Default setup uses INFO level for trace messages
    await tracer.setup()
    return tracer


@pytest.mark.asyncio
async def test_setup_default_log_level(console_tracer: ConsoleTracerPlugin, caplog: pytest.LogCaptureFixture):
    tracer_for_setup_test = ConsoleTracerPlugin()
    with caplog.at_level(logging.INFO, logger=TRACER_LOGGER_NAME):
        await tracer_for_setup_test.setup() # Default config
    
    assert tracer_for_setup_test._log_level == logging.INFO
    assert f"{tracer_for_setup_test.plugin_id}: Initialized. Trace events will be logged at level INFO." in caplog.text


@pytest.mark.asyncio
async def test_setup_custom_log_level(caplog: pytest.LogCaptureFixture):
    tracer = ConsoleTracerPlugin()
    with caplog.at_level(logging.INFO, logger=TRACER_LOGGER_NAME): # Setup logs at INFO
        await tracer.setup(config={"log_level": "DEBUG"})
    assert tracer._log_level == logging.DEBUG
    # The setup log message itself is INFO, but it states what level traces will be logged at.
    assert f"{tracer.plugin_id}: Initialized. Trace events will be logged at level DEBUG." in caplog.text
    caplog.clear() # Clear for the next setup

    tracer_warn = ConsoleTracerPlugin()
    with caplog.at_level(logging.INFO, logger=TRACER_LOGGER_NAME): # Setup logs at INFO
        await tracer_warn.setup(config={"log_level": "WARNING"})
    assert tracer_warn._log_level == logging.WARNING
    assert f"{tracer_warn.plugin_id}: Initialized. Trace events will be logged at level WARNING." in caplog.text


@pytest.mark.asyncio
async def test_record_trace_logs_correctly(console_tracer: ConsoleTracerPlugin, caplog: pytest.LogCaptureFixture):
    tracer = await console_tracer
    # Set the tracer's internal log level for trace messages to DEBUG for this test
    tracer._log_level = logging.DEBUG 
    
    event: TraceEvent = {
        "event_name": "user_login",
        "data": {"user_id": "123", "status": "success"},
        "timestamp": 1678886400.0, 
        "component": "AuthService",
        "correlation_id": "corr-abc"
    }

    with caplog.at_level(logging.DEBUG, logger=TRACER_LOGGER_NAME): 
        caplog.clear() 
        await tracer.record_trace(event)

    assert len(caplog.records) == 1
    log_record = caplog.records[0]
    assert log_record.levelname == "DEBUG" 
    assert log_record.name == TRACER_LOGGER_NAME 

    assert "TRACE :: Event: user_login" in log_record.message
    assert "Component: AuthService" in log_record.message
    assert "CorrID: corr-abc" in log_record.message
    # Exact JSON string formatting can be tricky, check for key content
    assert '"user_id": "123"' in log_record.message
    assert '"status": "success"' in log_record.message


@pytest.mark.asyncio
async def test_record_trace_long_data_truncation(console_tracer: ConsoleTracerPlugin, caplog: pytest.LogCaptureFixture):
    tracer = await console_tracer
    tracer._log_level = logging.INFO 

    long_value = "a" * 1500
    event: TraceEvent = {"event_name": "long_data", "data": {"long": long_value}, "timestamp": 0.0}
    
    with caplog.at_level(logging.INFO, logger=TRACER_LOGGER_NAME):
        caplog.clear()
        await tracer.record_trace(event)
    
    assert len(caplog.records) == 1
    log_message = caplog.records[0].message
    
    data_part_prefix = "Data: "
    assert data_part_prefix in log_message
    
    # The json.dumps with indent=2 will format the long string.
    # The entire formatted JSON string is then truncated if it exceeds 1000 characters.
    # We check that the key "long" is present, and the overall data string ends with "..."
    # and its length is roughly 1000 + len("...").
    
    assert '"long": "' in log_message 
    assert log_message.endswith("...") 

    data_str_in_log = log_message.split(data_part_prefix, 1)[1]
    assert len(data_str_in_log) <= 1000 + 3 # 1000 chars + "..."


@pytest.mark.asyncio
async def test_record_trace_serialization_error(console_tracer: ConsoleTracerPlugin, caplog: pytest.LogCaptureFixture):
    tracer = await console_tracer
    tracer._log_level = logging.INFO

    class NonSerializable:
        def __repr__(self): # Add repr for consistent str conversion
            return "<NonSerializableTestObject>"
            
    event: TraceEvent = {"event_name": "bad_data", "data": {"obj": NonSerializable()}, "timestamp": 0.0}

    with caplog.at_level(logging.INFO, logger=TRACER_LOGGER_NAME):
        caplog.clear()
        await tracer.record_trace(event)

    assert len(caplog.records) == 1
    log_message = caplog.records[0].message
    # json.dumps(..., default=str) calls str() on NonSerializable, which uses __repr__
    # The output of json.dumps with indent=2 for {"obj": "<NonSerializableTestObject>"}
    # will be '{\n  "obj": "<NonSerializableTestObject>"\n}'
    expected_data_str_part = '"obj": "<NonSerializableTestObject>"'
    assert expected_data_str_part in log_message

    # Check for truncation if the full data string was too long
    full_expected_json_dump = json.dumps(event['data'], default=str, indent=2)
    if len(full_expected_json_dump) > 1000:
        assert log_message.endswith("...")
    else:
        assert log_message.strip().endswith("}") # Ensure the JSON structure is complete if not truncated


@pytest.mark.asyncio
async def test_teardown(console_tracer: ConsoleTracerPlugin, caplog: pytest.LogCaptureFixture):
    tracer = await console_tracer
    with caplog.at_level(logging.DEBUG, logger=TRACER_LOGGER_NAME):
        caplog.clear()
        await tracer.teardown()
    assert f"{tracer.plugin_id}: Teardown complete." in caplog.text