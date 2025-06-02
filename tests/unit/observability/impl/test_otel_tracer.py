### tests/unit/observability/impl/test_otel_tracer.py
import asyncio
import json
import logging
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genie_tooling import __version__ as genie_version
from genie_tooling.observability.impl.otel_tracer import (
    OpenTelemetryTracerPlugin,
)
from genie_tooling.observability.types import TraceEvent

# Logger for the module under test
OTEL_TRACER_MODULE_LOGGER_NAME = "genie_tooling.observability.impl.otel_tracer"
OTEL_TRACER_PLUGIN_ID = OpenTelemetryTracerPlugin.plugin_id


@pytest.fixture
def mock_otel_sdk_components():
    """Mocks core OpenTelemetry SDK components used by the plugin."""
    mock_trace_module = MagicMock(name="MockOtelTraceModule")
    mock_tracer_provider_instance = MagicMock(name="MockTracerProviderInstance")
    mock_tracer_instance = MagicMock(name="MockTracerInstance")
    mock_span_instance = MagicMock(name="MockSpanInstance")

    # Configure trace module mocks
    mock_trace_module.get_tracer_provider.return_value = mock_tracer_provider_instance
    mock_trace_module.set_tracer_provider = MagicMock()
    mock_trace_module.get_tracer.return_value = mock_tracer_instance
    mock_trace_module.Status = MagicMock(name="MockOtelStatus")
    mock_trace_module.StatusCode = MagicMock(name="MockOtelStatusCode", ERROR="otel_status_code_error")

    # Configure TracerProvider instance mocks
    mock_tracer_provider_instance.add_span_processor = MagicMock()
    mock_tracer_provider_instance.shutdown = MagicMock()

    # Configure Tracer instance mocks
    mock_tracer_instance.start_as_current_span.return_value.__enter__.return_value = mock_span_instance

    # Configure Span instance mocks
    mock_span_instance.set_attribute = MagicMock()
    mock_span_instance.set_status = MagicMock()

    # Mock classes that are instantiated
    MockTracerProviderClass = MagicMock(name="MockTracerProviderClass", return_value=mock_tracer_provider_instance)
    MockResourceClass = MagicMock(name="MockResourceClass")
    MockResourceClass.create = MagicMock(return_value=MagicMock(name="MockResourceInstance"))
    MockBatchSpanProcessorClass = MagicMock(name="MockBatchSpanProcessorClass")
    MockConsoleSpanExporterClass = MagicMock(name="MockConsoleSpanExporterClass")
    MockOTLPHttpSpanExporterClass = MagicMock(name="MockOTLPHttpSpanExporterClass")
    MockOTLPGrpcSpanExporterClass = MagicMock(name="MockOTLPGrpcSpanExporterClass")

    patches = {
        "trace": patch("genie_tooling.observability.impl.otel_tracer.trace", mock_trace_module),
        "TracerProvider": patch("genie_tooling.observability.impl.otel_tracer.TracerProvider", MockTracerProviderClass),
        "Resource": patch("genie_tooling.observability.impl.otel_tracer.Resource", MockResourceClass),
        "BatchSpanProcessor": patch("genie_tooling.observability.impl.otel_tracer.BatchSpanProcessor", MockBatchSpanProcessorClass),
        "ConsoleSpanExporter": patch("genie_tooling.observability.impl.otel_tracer.ConsoleSpanExporter", MockConsoleSpanExporterClass),
        "OTLPHttpSpanExporter": patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter", MockOTLPHttpSpanExporterClass, create=True),
        "OTLPGrpcSpanExporter": patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter", MockOTLPGrpcSpanExporterClass, create=True),
    }

    # Store mocks for assertions
    return {
        "patches": patches,
        "trace_module": mock_trace_module,
        "TracerProviderClass": MockTracerProviderClass,
        "tracer_provider_instance": mock_tracer_provider_instance,
        "tracer_instance": mock_tracer_instance,
        "span_instance": mock_span_instance,
        "ResourceClass": MockResourceClass,
        "BatchSpanProcessorClass": MockBatchSpanProcessorClass,
        "ConsoleSpanExporterClass": MockConsoleSpanExporterClass,
        "OTLPHttpSpanExporterClass": MockOTLPHttpSpanExporterClass,
        "OTLPGrpcSpanExporterClass": MockOTLPGrpcSpanExporterClass,
    }


@pytest.fixture
async def otel_tracer(mock_otel_sdk_components) -> OpenTelemetryTracerPlugin: # Still async def
    """Provides an OpenTelemetryTracerPlugin instance with OTel SDK mocked."""
    # Activate patches
    for p in mock_otel_sdk_components["patches"].values():
        p.start()

    plugin = OpenTelemetryTracerPlugin()
    # yield plugin # This makes it an async generator
    # For a single yield, we can just return it and make the fixture sync if setup is sync
    # However, since setup is async, we keep the fixture async and use anext() in tests.
    # If setup were synchronous, we could do:
    # plugin.setup_sync_version()
    # yield plugin
    # plugin.teardown_sync_version()
    # But since setup is async, we must yield.
    yield plugin


    # Deactivate patches
    for p in mock_otel_sdk_components["patches"].values():
        p.stop()


@pytest.mark.asyncio
async def test_setup_otel_not_available(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger=OTEL_TRACER_MODULE_LOGGER_NAME)
    with patch("genie_tooling.observability.impl.otel_tracer.OTEL_AVAILABLE", False):
        plugin_no_otel = OpenTelemetryTracerPlugin()
        await plugin_no_otel.setup()
    assert plugin_no_otel._tracer is None
    assert plugin_no_otel._provider is None
    assert "OpenTelemetry libraries not found or core components missing. This tracer will be a no-op." in caplog.text


@pytest.mark.asyncio
async def test_setup_default_console_exporter(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components):
    plugin = await anext(otel_tracer) # Use anext()
    await plugin.setup() # Default config uses console exporter

    mock_otel_sdk_components["ResourceClass"].create.assert_called_once_with({
        "service.name": "genie-tooling-application",
        "service.version": genie_version,
    })
    mock_otel_sdk_components["TracerProviderClass"].assert_called_once()
    mock_otel_sdk_components["ConsoleSpanExporterClass"].assert_called_once()
    mock_otel_sdk_components["BatchSpanProcessorClass"].assert_called_once_with(
        mock_otel_sdk_components["ConsoleSpanExporterClass"].return_value
    )
    mock_otel_sdk_components["tracer_provider_instance"].add_span_processor.assert_called_once()
    mock_otel_sdk_components["trace_module"].set_tracer_provider.assert_called_once_with(
        mock_otel_sdk_components["tracer_provider_instance"]
    )
    mock_otel_sdk_components["trace_module"].get_tracer.assert_called_once_with(OTEL_TRACER_PLUGIN_ID, genie_version)
    assert plugin._tracer is mock_otel_sdk_components["tracer_instance"]


@pytest.mark.asyncio
async def test_setup_otlp_http_exporter(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components):
    plugin = await anext(otel_tracer) # Use anext()
    config = {
        "otel_service_name": "my-http-app",
        "exporter_type": "otlp_http",
        "otlp_http_endpoint": "http://testhost:1234/traces",
        "otlp_http_headers": "key1=val1,key2=val2",
        "otlp_http_timeout": 20,
        "resource_attributes": {"env": "test"},
    }
    await plugin.setup(config)

    mock_otel_sdk_components["ResourceClass"].create.assert_called_once_with({
        "service.name": "my-http-app", "service.version": genie_version, "env": "test"
    })
    mock_otel_sdk_components["OTLPHttpSpanExporterClass"].assert_called_once_with(
        endpoint="http://testhost:1234/traces", headers={"key1": "val1", "key2": "val2"}, timeout=20
    )
    mock_otel_sdk_components["BatchSpanProcessorClass"].assert_called_once_with(
        mock_otel_sdk_components["OTLPHttpSpanExporterClass"].return_value
    )


@pytest.mark.asyncio
async def test_setup_otlp_grpc_exporter(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components):
    plugin = await anext(otel_tracer) # Use anext()
    config = {
        "exporter_type": "otlp_grpc",
        "otlp_grpc_endpoint": "grpc_host:5678",
        "otlp_grpc_insecure": True,
        "otlp_grpc_timeout": 15,
    }
    await plugin.setup(config)
    mock_otel_sdk_components["OTLPGrpcSpanExporterClass"].assert_called_once_with(
        endpoint="grpc_host:5678", insecure=True, timeout=15
    )


@pytest.mark.asyncio
async def test_setup_otlp_http_exporter_import_fails(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components, caplog):
    plugin = await anext(otel_tracer) # Use anext()
    caplog.set_level(logging.ERROR, logger=OTEL_TRACER_MODULE_LOGGER_NAME)
    mock_otel_sdk_components["patches"]["OTLPHttpSpanExporter"].stop() # Stop the successful patch
    with patch("opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter", side_effect=ImportError("cannot import OTLP HTTP")):
        await plugin.setup(config={"exporter_type": "otlp_http"})

    assert "'opentelemetry-exporter-otlp-proto-http' not installed. Cannot use OTLP HTTP exporter. Falling back to console." in caplog.text
    mock_otel_sdk_components["ConsoleSpanExporterClass"].assert_called() # Check fallback


@pytest.mark.asyncio
async def test_record_trace_no_tracer(otel_tracer: OpenTelemetryTracerPlugin, caplog):
    plugin = await anext(otel_tracer) # Use anext()
    caplog.set_level(logging.DEBUG, logger=OTEL_TRACER_MODULE_LOGGER_NAME)
    plugin._tracer = None # Simulate tracer not being initialized
    event: TraceEvent = {"event_name": "test_event", "data": {}, "timestamp": asyncio.get_event_loop().time()}
    await plugin.record_trace(event)
    assert "Tracer not available, skipping trace for event 'test_event'." in caplog.text


@pytest.mark.asyncio
async def test_record_trace_basic_event(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components):
    plugin = await anext(otel_tracer) # Use anext()
    await plugin.setup() # Initialize the tracer
    event_ts = asyncio.get_event_loop().time()
    event: TraceEvent = {
        "event_name": "user.action",
        "data": {"item_id": 123, "action_type": "click"},
        "timestamp": event_ts,
        "component": "UIComponent",
        "correlation_id": "corr-xyz",
    }
    await plugin.record_trace(event)

    mock_otel_sdk_components["tracer_instance"].start_as_current_span.assert_called_once_with(
        "user.action", start_time=int(event_ts * 1_000_000_000)
    )
    span_mock = mock_otel_sdk_components["span_instance"]
    span_mock.set_attribute.assert_any_call("component", "UIComponent")
    span_mock.set_attribute.assert_any_call("correlation_id", "corr-xyz")
    span_mock.set_attribute.assert_any_call("data.item_id", 123)
    span_mock.set_attribute.assert_any_call("data.action_type", "click")
    span_mock.set_status.assert_not_called()


@pytest.mark.asyncio
async def test_record_trace_with_llm_usage(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components):
    plugin = await anext(otel_tracer) # Use anext()
    await plugin.setup()
    event: TraceEvent = {
        "event_name": "llm.call",
        "data": {"llm.usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}},
        "timestamp": asyncio.get_event_loop().time(),
    }
    await plugin.record_trace(event)
    span_mock = mock_otel_sdk_components["span_instance"]
    span_mock.set_attribute.assert_any_call("llm.usage.prompt_tokens", 100)
    span_mock.set_attribute.assert_any_call("llm.usage.completion_tokens", 50)
    span_mock.set_attribute.assert_any_call("llm.usage.total_tokens", 150)


@pytest.mark.asyncio
async def test_record_trace_with_error_data(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components):
    plugin = await anext(otel_tracer) # Use anext()
    await plugin.setup()
    event: TraceEvent = {
        "event_name": "operation.failed",
        "data": {
            "error_message": "Something went wrong",
            "error_type": "ValueError",
            "error_stacktrace": "Traceback...",
        },
        "timestamp": asyncio.get_event_loop().time(),
    }
    await plugin.record_trace(event)
    span_mock = mock_otel_sdk_components["span_instance"]
    mock_otel_sdk_components["trace_module"].Status.assert_called_once_with(
        mock_otel_sdk_components["trace_module"].StatusCode.ERROR, description="Something went wrong"
    )
    span_mock.set_status.assert_called_once_with(mock_otel_sdk_components["trace_module"].Status.return_value)
    span_mock.set_attribute.assert_any_call("error.type", "ValueError")
    span_mock.set_attribute.assert_any_call("error.message", "Something went wrong")
    span_mock.set_attribute.assert_any_call("error.stacktrace", "Traceback...")


@pytest.mark.asyncio
async def test_record_trace_complex_data_serialization(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components):
    plugin = await anext(otel_tracer) # Use anext()
    await plugin.setup()
    complex_obj = {"nested": {"a": [1, 2]}, "flag": True}
    event: TraceEvent = {
        "event_name": "complex.data.event",
        "data": {"complex_payload": complex_obj, "simple_list": [1, "b"]},
        "timestamp": asyncio.get_event_loop().time(),
    }
    await plugin.record_trace(event)
    span_mock = mock_otel_sdk_components["span_instance"]
    span_mock.set_attribute.assert_any_call("data.complex_payload", json.dumps(complex_obj, default=str))
    span_mock.set_attribute.assert_any_call("data.simple_list", ["1", "b"]) # List of simple types


@pytest.mark.asyncio
async def test_teardown_success(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components):
    plugin = await anext(otel_tracer) # Use anext()
    await plugin.setup() # Ensure provider is set
    assert plugin._provider is mock_otel_sdk_components["tracer_provider_instance"]
    await plugin.teardown()
    mock_otel_sdk_components["tracer_provider_instance"].shutdown.assert_called_once()
    assert plugin._tracer is None
    assert plugin._provider is None


@pytest.mark.asyncio
async def test_teardown_provider_shutdown_error(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components, caplog):
    plugin = await anext(otel_tracer) # Use anext()
    caplog.set_level(logging.ERROR, logger=OTEL_TRACER_MODULE_LOGGER_NAME)
    await plugin.setup()
    mock_otel_sdk_components["tracer_provider_instance"].shutdown.side_effect = RuntimeError("Shutdown failed")
    await plugin.teardown()
    assert "Error shutting down TracerProvider: Shutdown failed" in caplog.text
    assert plugin._provider is None # Should still be nulled