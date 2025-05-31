### tests/unit/observability/impl/test_otel_tracer.py
import logging
from unittest.mock import MagicMock, patch

import pytest
from genie_tooling.observability.impl.otel_tracer import (
    OpenTelemetryTracerPlugin,
)
from genie_tooling.observability.types import TraceEvent

TRACER_LOGGER_NAME = "genie_tooling.observability.impl.otel_tracer"


@pytest.fixture
def otel_tracer() -> OpenTelemetryTracerPlugin:
    return OpenTelemetryTracerPlugin()


@pytest.mark.asyncio
async def test_setup_otel_not_available(otel_tracer: OpenTelemetryTracerPlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger=TRACER_LOGGER_NAME)
    with patch("genie_tooling.observability.impl.otel_tracer.OTEL_AVAILABLE", False):
        await otel_tracer.setup()
    assert otel_tracer._tracer is None
    assert f"{otel_tracer.plugin_id}: OpenTelemetry libraries not found. This tracer will be a no-op." in caplog.text


@pytest.mark.asyncio
async def test_setup_otel_available_stub_behavior(otel_tracer: OpenTelemetryTracerPlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger=TRACER_LOGGER_NAME)
    # Even if OTEL_AVAILABLE is True, the current implementation is a stub.
    # We don't mock the OTel library itself here, as the plugin doesn't use it yet.
    with patch("genie_tooling.observability.impl.otel_tracer.OTEL_AVAILABLE", True):
        await otel_tracer.setup(config={"otel_service_name": "my_test_service"})

    assert otel_tracer._tracer is None # Stub doesn't set a real tracer
    assert f"{otel_tracer.plugin_id}: STUB Initialized. OTEL_AVAILABLE=True. Service Name: my_test_service" in caplog.text


@pytest.mark.asyncio
async def test_record_trace_tracer_not_available(otel_tracer: OpenTelemetryTracerPlugin, caplog: pytest.LogCaptureFixture):
    # Ensure tracer is None (e.g., OTEL_AVAILABLE was False during setup)
    await otel_tracer.setup() # Default setup where OTEL_AVAILABLE might be False
    otel_tracer._tracer = None # Explicitly ensure it's None for this test path

    # caplog.set_level(logging.DEBUG, logger=TRACER_LOGGER_NAME) # No debug log in current stub for this case
    event: TraceEvent = {"event_name": "test_event", "data": {"key": "val"}, "timestamp": 123.45}
    await otel_tracer.record_trace(event)
    # Current stub doesn't log if tracer is None, it just returns.
    # If it were to log, we'd check caplog.text here.
    assert True # Test passes if no error and no specific log expected


@pytest.mark.asyncio
async def test_record_trace_stub_behavior_with_tracer_placeholder(otel_tracer: OpenTelemetryTracerPlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=TRACER_LOGGER_NAME)
    # Simulate that setup somehow assigned a mock tracer (even though current stub doesn't)
    # This tests the path where self._tracer is not None.
    otel_tracer._tracer = MagicMock(name="MockOtelTracerInstance")

    event: TraceEvent = {"event_name": "another_event", "data": {"info": "detail"}, "timestamp": 678.90}
    await otel_tracer.record_trace(event)

    assert f"{otel_tracer.plugin_id}: STUB record_trace called for event: another_event" in caplog.text
    # If the stub were to interact with the mock tracer, we'd assert calls here:
    # otel_tracer._tracer.start_as_current_span.assert_called_once_with("another_event")


@pytest.mark.asyncio
async def test_teardown(otel_tracer: OpenTelemetryTracerPlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=TRACER_LOGGER_NAME)
    await otel_tracer.teardown()
    assert f"{otel_tracer.plugin_id}: STUB Teardown complete." in caplog.text
