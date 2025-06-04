### tests/unit/observability/impl/test_otel_tracer.py
import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest
from genie_tooling import __version__ as genie_version
from genie_tooling.observability.impl.otel_tracer import (
    OpenTelemetryTracerPlugin,
)
from genie_tooling.observability.types import TraceEvent

OTEL_TRACER_MODULE_LOGGER_NAME = "genie_tooling.observability.impl.otel_tracer"
OTEL_TRACER_PLUGIN_ID = OpenTelemetryTracerPlugin.plugin_id

try:
    from asyncio import anext
except ImportError:
    async def anext(ait): # type: ignore
        return await ait.__anext__()


@pytest.fixture
def mock_otel_sdk_components_fixture():
    mock_trace_api_module = MagicMock(name="MockOtelTraceAPIModule")
    mock_tracer_provider_instance = MagicMock(name="MockTracerProviderInstance")
    mock_tracer_instance = MagicMock(name="MockTracerInstance")
    mock_span_instance = MagicMock(name="MockSpanInstance")

    mock_trace_api_module.get_tracer_provider.return_value = mock_tracer_provider_instance
    mock_trace_api_module.set_tracer_provider = MagicMock(name="MockSetTracerProvider")
    mock_trace_api_module.get_tracer.return_value = mock_tracer_instance

    MockStatusClass = MagicMock(name="MockOtelStatusClass")
    MockStatusInstance = MagicMock(name="MockOtelStatusInstance")
    MockStatusClass.return_value = MockStatusInstance
    mock_trace_api_module.Status = MockStatusClass

    MockStatusCodeClass = MagicMock(name="MockOtelStatusCodeClass")
    MockStatusCodeInstance_ERROR = MagicMock(name="MockStatusCodeInstance_ERROR")
    MockStatusCodeClass.ERROR = MockStatusCodeInstance_ERROR
    mock_trace_api_module.StatusCode = MockStatusCodeClass

    mock_tracer_provider_instance.add_span_processor = MagicMock()
    mock_tracer_provider_instance.shutdown = MagicMock()
    mock_tracer_provider_instance.resource = MagicMock(name="MockResourceOnProvider")
    mock_tracer_provider_instance.resource.attributes = {"telemetry.sdk.version": "test.sdk.version.fixture"}

    mock_tracer_instance.start_as_current_span.return_value.__enter__.return_value = mock_span_instance
    mock_span_instance.set_attribute = MagicMock()
    mock_span_instance.set_status = MagicMock()

    MockTracerProviderClass = MagicMock(name="MockTracerProviderClass", return_value=mock_tracer_provider_instance)
    MockResourceClass = MagicMock(name="MockResourceClass")
    MockResourceClass.create = MagicMock(return_value=MagicMock(name="MockResourceInstanceFromCreate"))
    MockBatchSpanProcessorClass = MagicMock(name="MockBatchSpanProcessorClass")
    MockConsoleSpanExporterClass = MagicMock(name="MockConsoleSpanExporterClass")
    MockOTLPHttpSpanExporterSDKClass = MagicMock(name="MockOTLPHttpSpanExporterSDKClass")
    MockOTLPGrpcSpanExporterSDKClass = MagicMock(name="MockOTLPGrpcSpanExporterSDKClass")

    patches = {
        "trace_api": patch("genie_tooling.observability.impl.otel_tracer.trace_api", mock_trace_api_module),
        "TracerProvider": patch("genie_tooling.observability.impl.otel_tracer.TracerProvider", MockTracerProviderClass),
        "Resource": patch("genie_tooling.observability.impl.otel_tracer.Resource", MockResourceClass),
        "BatchSpanProcessor": patch("genie_tooling.observability.impl.otel_tracer.BatchSpanProcessor", MockBatchSpanProcessorClass),
        "ConsoleSpanExporter": patch("genie_tooling.observability.impl.otel_tracer.ConsoleSpanExporter", MockConsoleSpanExporterClass),
        "OTLPHttpSpanExporter_SDK": patch("genie_tooling.observability.impl.otel_tracer.OTLPHttpSpanExporter_SDK", MockOTLPHttpSpanExporterSDKClass, create=True),
        "OTLPGrpcSpanExporter_SDK": patch("genie_tooling.observability.impl.otel_tracer.OTLPGrpcSpanExporter_SDK", MockOTLPGrpcSpanExporterSDKClass, create=True),
        "Status": patch("genie_tooling.observability.impl.otel_tracer.Status", MockStatusClass),
        "StatusCode": patch("genie_tooling.observability.impl.otel_tracer.StatusCode", MockStatusCodeClass),
    }
    return {
        "patches": patches, "trace_api_module": mock_trace_api_module,
        "TracerProviderClass": MockTracerProviderClass, "tracer_provider_instance": mock_tracer_provider_instance,
        "tracer_instance": mock_tracer_instance, "span_instance": mock_span_instance,
        "ResourceClass": MockResourceClass, "BatchSpanProcessorClass": MockBatchSpanProcessorClass,
        "ConsoleSpanExporterClass": MockConsoleSpanExporterClass,
        "OTLPHttpSpanExporterSDKClass": MockOTLPHttpSpanExporterSDKClass,
        "OTLPGrpcSpanExporterSDKClass": MockOTLPGrpcSpanExporterSDKClass,
        "MockStatusClass": MockStatusClass,
        "MockStatusCode_ERROR_Instance": MockStatusCodeInstance_ERROR,
    }

@pytest.fixture
async def otel_tracer(mock_otel_sdk_components_fixture):
    for p_name, p_obj in mock_otel_sdk_components_fixture["patches"].items():
        try:
            p_obj.start()
        except Exception as e:
            print(f"Error starting patch {p_name}: {e}")
    plugin = OpenTelemetryTracerPlugin()
    yield plugin
    for p_name, p_obj in mock_otel_sdk_components_fixture["patches"].items():
        try:
            p_obj.stop()
        except Exception as e:
            print(f"Error stopping patch {p_name}: {e}")


@pytest.mark.asyncio
async def test_setup_default_console_exporter(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components_fixture):
    plugin = await anext(otel_tracer)
    await plugin.setup()

    mock_otel_sdk_components_fixture["trace_api_module"].set_tracer_provider.assert_called_once_with(
        mock_otel_sdk_components_fixture["tracer_provider_instance"]
    )
    mock_otel_sdk_components_fixture["ResourceClass"].create.assert_called_once_with({
        "service.name": "genie-tooling-application", "service.version": genie_version,
        "telemetry.sdk.name": "opentelemetry", "telemetry.sdk.language": "python",
        "telemetry.sdk.version": "test.sdk.version.fixture",
    })
    mock_otel_sdk_components_fixture["ConsoleSpanExporterClass"].assert_called_once()
    mock_otel_sdk_components_fixture["tracer_provider_instance"].add_span_processor.assert_called_once()
    mock_otel_sdk_components_fixture["trace_api_module"].get_tracer.assert_called_once_with(
        OTEL_TRACER_PLUGIN_ID, genie_version
    )
    assert plugin._tracer is mock_otel_sdk_components_fixture["tracer_instance"]

@pytest.mark.asyncio
async def test_setup_otlp_http_exporter(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components_fixture):
    plugin = await anext(otel_tracer)
    config = {
        "otel_service_name": "my-http-app", "exporter_type": "otlp_http",
        "otlp_http_endpoint": "http://testhost:1234/traces",
        "otlp_http_headers": "key1=val1,key2=val2", "otlp_http_timeout": 20,
        "resource_attributes": {"env": "test"},
    }
    await plugin.setup(config)
    mock_otel_sdk_components_fixture["OTLPHttpSpanExporterSDKClass"].assert_called_once_with(
        endpoint="http://testhost:1234/traces", headers={"key1": "val1", "key2": "val2"}, timeout=20
    )

@pytest.mark.asyncio
async def test_setup_otlp_grpc_exporter(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components_fixture):
    plugin = await anext(otel_tracer)
    config = {
        "exporter_type": "otlp_grpc", "otlp_grpc_endpoint": "grpc_host:5678",
        "otlp_grpc_insecure": True, "otlp_grpc_timeout": 15,
    }
    await plugin.setup(config)
    mock_otel_sdk_components_fixture["OTLPGrpcSpanExporterSDKClass"].assert_called_once_with(
        endpoint="grpc_host:5678", insecure=True, timeout=15
    )

@pytest.mark.asyncio
async def test_setup_otlp_http_exporter_import_fails(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components_fixture, caplog):
    plugin = await anext(otel_tracer)
    caplog.set_level(logging.ERROR, logger=OTEL_TRACER_MODULE_LOGGER_NAME)

    mock_otel_sdk_components_fixture["patches"]["OTLPHttpSpanExporter_SDK"].stop()
    with patch("genie_tooling.observability.impl.otel_tracer.OTLPHttpSpanExporter_SDK", None):
        await plugin.setup(config={"exporter_type": "otlp_http"})

    # CORRECTED Assertion:
    expected_log_message = f"{plugin.plugin_id}: 'opentelemetry-exporter-otlp-proto-http' not installed. Falling back to console."
    assert expected_log_message in caplog.text

    mock_otel_sdk_components_fixture["ConsoleSpanExporterClass"].assert_called()
    # mock_otel_sdk_components_fixture["patches"]["OTLPHttpSpanExporter_SDK"].start() # Restart patch if needed for other tests

@pytest.mark.asyncio
async def test_record_trace_with_error_data(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components_fixture):
    plugin = await anext(otel_tracer)
    await plugin.setup()
    event: TraceEvent = {
        "event_name": "operation.failed",
        "data": {"error_message": "Something went wrong", "error_type": "ValueError"},
        "timestamp": asyncio.get_event_loop().time(),
    }
    await plugin.record_trace(event)
    span_mock = mock_otel_sdk_components_fixture["span_instance"]
    mock_otel_sdk_components_fixture["MockStatusClass"].assert_called_once_with(
        mock_otel_sdk_components_fixture["MockStatusCode_ERROR_Instance"], description="Something went wrong"
    )
    span_mock.set_status.assert_called_once_with(mock_otel_sdk_components_fixture["MockStatusClass"].return_value)
    span_mock.set_attribute.assert_any_call("exception.type", "ValueError")
    span_mock.set_attribute.assert_any_call("exception.message", "Something went wrong")

@pytest.mark.asyncio
async def test_record_trace_complex_data_serialization(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components_fixture):
    plugin = await anext(otel_tracer)
    await plugin.setup()
    complex_obj = {"nested": {"a": [1, 2]}, "flag": True}
    event: TraceEvent = {
        "event_name": "complex.data.event",
        "data": {"complex_payload": complex_obj, "simple_list": [1, "b"]},
        "timestamp": asyncio.get_event_loop().time(),
    }
    await plugin.record_trace(event)
    span_mock = mock_otel_sdk_components_fixture["span_instance"]
    span_mock.set_attribute.assert_any_call("data.complex_payload.nested.a", ["1", "2"])
    span_mock.set_attribute.assert_any_call("data.complex_payload.flag", True)
    span_mock.set_attribute.assert_any_call("data.simple_list", ["1", "b"])

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
async def test_record_trace_no_tracer(otel_tracer: OpenTelemetryTracerPlugin, caplog):
    plugin = await anext(otel_tracer)
    caplog.set_level(logging.DEBUG, logger=OTEL_TRACER_MODULE_LOGGER_NAME)
    plugin._tracer = None
    event: TraceEvent = {"event_name": "test_event", "data": {}, "timestamp": asyncio.get_event_loop().time()}
    await plugin.record_trace(event)
    assert "Tracer not available, skipping trace for event 'test_event'." in caplog.text

@pytest.mark.asyncio
async def test_record_trace_basic_event(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components_fixture):
    plugin = await anext(otel_tracer)
    await plugin.setup()
    event_ts = asyncio.get_event_loop().time()
    event: TraceEvent = {
        "event_name": "user.action",
        "data": {"item_id": 123, "action_type": "click"},
        "timestamp": event_ts, "component": "UIComponent", "correlation_id": "corr-xyz",
    }
    await plugin.record_trace(event)
    mock_otel_sdk_components_fixture["tracer_instance"].start_as_current_span.assert_called_once_with(
        "user.action", start_time=int(event_ts * 1_000_000_000)
    )
    span_mock = mock_otel_sdk_components_fixture["span_instance"]
    span_mock.set_attribute.assert_any_call("component", "UIComponent")
    span_mock.set_attribute.assert_any_call("correlation_id", "corr-xyz")
    span_mock.set_attribute.assert_any_call("data.item_id", 123)
    span_mock.set_attribute.assert_any_call("data.action_type", "click")

@pytest.mark.asyncio
async def test_record_trace_with_llm_usage(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components_fixture):
    plugin = await anext(otel_tracer)
    await plugin.setup()
    event: TraceEvent = {
        "event_name": "llm.call",
        "data": {"llm.usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}},
        "timestamp": asyncio.get_event_loop().time(),
    }
    await plugin.record_trace(event)
    span_mock = mock_otel_sdk_components_fixture["span_instance"]
    span_mock.set_attribute.assert_any_call("data.llm.usage.prompt_tokens", 100)
    span_mock.set_attribute.assert_any_call("data.llm.usage.completion_tokens", 50)
    span_mock.set_attribute.assert_any_call("data.llm.usage.total_tokens", 150)

@pytest.mark.asyncio
async def test_teardown_success(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components_fixture):
    plugin = await anext(otel_tracer)
    await plugin.setup()
    assert plugin._provider is mock_otel_sdk_components_fixture["tracer_provider_instance"]
    await plugin.teardown()
    mock_otel_sdk_components_fixture["tracer_provider_instance"].shutdown.assert_called_once()
    assert plugin._tracer is None
    assert plugin._provider is None

@pytest.mark.asyncio
async def test_teardown_provider_shutdown_error(otel_tracer: OpenTelemetryTracerPlugin, mock_otel_sdk_components_fixture, caplog):
    plugin = await anext(otel_tracer)
    caplog.set_level(logging.ERROR, logger=OTEL_TRACER_MODULE_LOGGER_NAME)
    await plugin.setup()
    mock_otel_sdk_components_fixture["tracer_provider_instance"].shutdown.side_effect = RuntimeError("Shutdown failed")
    await plugin.teardown()
    assert "Error shutting down TracerProvider: Shutdown failed" in caplog.text
    assert plugin._provider is None
