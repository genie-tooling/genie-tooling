"""OpenTelemetryTracerPlugin: Sends trace events using OpenTelemetry."""
import asyncio
import json
import logging
from typing import Any, Dict, Optional

from genie_tooling import __version__
from genie_tooling.observability.abc import InteractionTracerPlugin
from genie_tooling.observability.types import TraceEvent

logger = logging.getLogger(__name__)

OTEL_AVAILABLE = False
trace_api = None
TracerProvider = None
Resource = None
BatchSpanProcessor = None
ConsoleSpanExporter = None
OTLPHttpSpanExporter_SDK = None
OTLPGrpcSpanExporter_SDK = None
StatusCode = None
Status = None

try:
    from opentelemetry import trace as otel_trace_api
    from opentelemetry.sdk.resources import Resource as OtelResource
    from opentelemetry.sdk.trace import TracerProvider as OtelTracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor as OtelBatchSpanProcessor,
    )
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter as OtelConsoleSpanExporter,
    )
    from opentelemetry.trace import Status as OtelStatus
    from opentelemetry.trace import StatusCode as OtelStatusCode
    OTEL_AVAILABLE = True
    trace_api = otel_trace_api
    TracerProvider = OtelTracerProvider
    Resource = OtelResource
    BatchSpanProcessor = OtelBatchSpanProcessor
    ConsoleSpanExporter = OtelConsoleSpanExporter
    Status = OtelStatus
    StatusCode = OtelStatusCode

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as OTLPHttpProtoSpanExporter,
        )
        OTLPHttpSpanExporter_SDK = OTLPHttpProtoSpanExporter
    except ImportError:
        logger.debug("OTLP HTTP Exporter for OpenTelemetry not available. Install 'opentelemetry-exporter-otlp-proto-http'.")
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as OTLPGrpcProtoSpanExporter,
        )
        OTLPGrpcSpanExporter_SDK = OTLPGrpcProtoSpanExporter
    except ImportError:
        logger.debug("OTLP gRPC Exporter for OpenTelemetry not available. Install 'opentelemetry-exporter-otlp-proto-grpc'.")

except ImportError:
    logger.warning(
        "OpenTelemetryTracerPlugin: 'opentelemetry-sdk' or 'opentelemetry-api' "
        "not installed. This tracer will be a no-op."
    )


class OpenTelemetryTracerPlugin(InteractionTracerPlugin):
    plugin_id: str = "otel_tracer_plugin_v1"
    description: str = "Sends trace events using OpenTelemetry."

    _tracer: Optional["trace_api.Tracer"] = None
    _provider: Optional["TracerProvider"] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not OTEL_AVAILABLE or not trace_api or not TracerProvider or not Resource or \
           not BatchSpanProcessor or not ConsoleSpanExporter or not Status or not StatusCode:
            logger.warning(f"{self.plugin_id}: OpenTelemetry libraries not found or core components missing. This tracer will be a no-op.")
            return

        cfg = config or {}
        service_name = cfg.get("otel_service_name", "genie-tooling-application")
        service_version = cfg.get("otel_service_version", __version__)
        exporter_type = cfg.get("exporter_type", "console").lower()

        sdk_version_val = "unknown"
        try:
            current_provider_for_sdk_ver = trace_api.get_tracer_provider()
            if hasattr(current_provider_for_sdk_ver, "resource") and \
               current_provider_for_sdk_ver.resource and \
               hasattr(current_provider_for_sdk_ver.resource, "attributes") and \
               isinstance(current_provider_for_sdk_ver.resource.attributes, dict):
                sdk_version_val = current_provider_for_sdk_ver.resource.attributes.get("telemetry.sdk.version", "unknown")
        except Exception:
            logger.debug(f"{self.plugin_id}: Could not determine OpenTelemetry SDK version dynamically, defaulting to 'unknown'.")

        base_resource_attributes = {
            "service.name": service_name, "service.version": service_version,
            "telemetry.sdk.name": "opentelemetry", "telemetry.sdk.language": "python",
            "telemetry.sdk.version": sdk_version_val
        }
        custom_resource_attrs = cfg.get("resource_attributes", {})
        final_resource_attributes = {**base_resource_attributes, **custom_resource_attrs}

        resource = Resource.create(final_resource_attributes)
        self._provider = TracerProvider(resource=resource)
        trace_api.set_tracer_provider(self._provider)
        logger.info(f"{self.plugin_id}: Set new TracerProvider globally.")

        exporter: Any = None
        if exporter_type == "otlp_http":
            if OTLPHttpSpanExporter_SDK:
                http_endpoint = cfg.get("otlp_http_endpoint", "http://localhost:4318/v1/traces")
                http_headers_str_or_dict = cfg.get("otlp_http_headers")
                http_headers_dict: Optional[Dict[str, str]] = None
                if isinstance(http_headers_str_or_dict, dict): http_headers_dict = http_headers_str_or_dict
                elif isinstance(http_headers_str_or_dict, str) and http_headers_str_or_dict.strip():
                    try: http_headers_dict = dict(item.split("=", 1) for item in http_headers_str_or_dict.split(",") if "=" in item)
                    except ValueError: logger.warning(f"{self.plugin_id}: Could not parse otlp_http_headers: '{http_headers_str_or_dict}'.")
                http_timeout = int(cfg.get("otlp_http_timeout", 10))
                http_compression = cfg.get("otlp_http_compression")
                exporter_args = {"endpoint": http_endpoint, "timeout": http_timeout}
                if http_headers_dict: exporter_args["headers"] = http_headers_dict
                if http_compression: exporter_args["compression"] = http_compression # type: ignore
                try:
                    exporter = OTLPHttpSpanExporter_SDK(**exporter_args)
                    logger.info(f"{self.plugin_id}: Using OTLP HTTP Exporter to {http_endpoint}")
                except Exception as e_http_exp_init:
                    logger.error(f"{self.plugin_id}: Failed to initialize OTLP HTTP Exporter: {e_http_exp_init}. Falling back to console.", exc_info=True)
                    exporter = ConsoleSpanExporter()
            else:
                logger.error(f"{self.plugin_id}: 'opentelemetry-exporter-otlp-proto-http' not installed. Falling back to console.")
                exporter = ConsoleSpanExporter()
        elif exporter_type == "otlp_grpc":
            if OTLPGrpcSpanExporter_SDK:
                grpc_endpoint = cfg.get("otlp_grpc_endpoint", "localhost:4317")
                grpc_insecure = bool(cfg.get("otlp_grpc_insecure", False))
                grpc_timeout = int(cfg.get("otlp_grpc_timeout", 10))
                grpc_compression = cfg.get("otlp_grpc_compression")
                exporter_args = {"endpoint": grpc_endpoint, "insecure": grpc_insecure, "timeout": grpc_timeout}
                if grpc_compression: exporter_args["compression"] = grpc_compression # type: ignore
                try:
                    exporter = OTLPGrpcSpanExporter_SDK(**exporter_args)
                    logger.info(f"{self.plugin_id}: Using OTLP gRPC Exporter to {grpc_endpoint}")
                except Exception as e_grpc_exp_init:
                    logger.error(f"{self.plugin_id}: Failed to initialize OTLP gRPC Exporter: {e_grpc_exp_init}. Falling back to console.", exc_info=True)
                    exporter = ConsoleSpanExporter()
            else:
                logger.error(f"{self.plugin_id}: 'opentelemetry-exporter-otlp-proto-grpc' not installed. Falling back to console.")
                exporter = ConsoleSpanExporter()
        else:
            exporter = ConsoleSpanExporter()
            logger.info(f"{self.plugin_id}: Using Console Exporter.")

        processor = BatchSpanProcessor(exporter)
        self._provider.add_span_processor(processor)
        self._tracer = trace_api.get_tracer(self.plugin_id, __version__)
        logger.info(f"{self.plugin_id}: OpenTelemetry Tracer initialized. Service: '{service_name}'. Exporter: {exporter_type}.")

    def _flatten_dict_for_otel(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        items: Dict[str, Any] = {}
        for k, v in data.items():
            new_key = f"{prefix}{k}"
            if isinstance(v, dict):
                items.update(self._flatten_dict_for_otel(v, f"{new_key}."))
            elif isinstance(v, list) and all(isinstance(item, (str, bool, int, float)) for item in v):
                items[new_key] = [str(item) for item in v]
            elif isinstance(v, (str, bool, int, float)):
                items[new_key] = v
            else:
                try: items[new_key] = json.dumps(v, default=str)
                except (TypeError, OverflowError): items[new_key] = f"[UnserializableValue:{type(v).__name__}]"
        return items

    async def record_trace(self, event: TraceEvent) -> None:
        if not self._tracer or not trace_api or not Status or not StatusCode:
            logger.debug(f"{self.plugin_id}: Tracer not available, skipping trace for event '{event['event_name']}'.")
            return

        start_time_ns = int(event["timestamp"] * 1_000_000_000)

        with self._tracer.start_as_current_span(event["event_name"], start_time=start_time_ns) as span:
            if event.get("component"): span.set_attribute("component", event["component"])
            if event.get("correlation_id"): span.set_attribute("correlation_id", event["correlation_id"])

            # CORRECTED: Pass "data." as the initial prefix
            flat_data = self._flatten_dict_for_otel(event.get("data", {}), prefix="data.")
            for key, value in flat_data.items():
                try: span.set_attribute(key, value)
                except Exception as e_attr:
                    logger.debug(f"{self.plugin_id}: Could not set attribute '{key}': {e_attr}")
                    span.set_attribute(key, "[AttributeError_UnserializableValue]")

            error_message = event["data"].get("error_message", event["data"].get("error"))
            error_type = event["data"].get("error_type", event["data"].get("type"))
            error_stacktrace = event["data"].get("error_stacktrace")

            if error_message or error_type:
                description = error_message if isinstance(error_message, str) else "Unknown error"
                span.set_status(Status(StatusCode.ERROR, description=description))
                if error_type and isinstance(error_type, str): span.set_attribute("exception.type", error_type)
                if error_message and isinstance(error_message, str): span.set_attribute("exception.message", error_message)
                if error_stacktrace and isinstance(error_stacktrace, str): span.set_attribute("exception.stacktrace", error_stacktrace)

    async def teardown(self) -> None:
        if self._provider:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._provider.shutdown)
                logger.info(f"{self.plugin_id}: OpenTelemetry TracerProvider shut down.")
            except Exception as e:
                logger.error(f"{self.plugin_id}: Error shutting down TracerProvider: {e}", exc_info=True)
        self._tracer = None
        self._provider = None
        logger.debug(f"{self.plugin_id}: Teardown complete.")
