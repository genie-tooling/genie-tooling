"""OpenTelemetryTracerPlugin: Sends trace events using OpenTelemetry."""
import json
import logging
from typing import Any, Dict, Optional

from genie_tooling import __version__
from genie_tooling.observability.abc import InteractionTracerPlugin
from genie_tooling.observability.types import TraceEvent

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    TracerProvider = None # type: ignore
    Resource = None # type: ignore
    BatchSpanProcessor = None # type: ignore
    ConsoleSpanExporter = None # type: ignore
    trace = None # type: ignore
    logger.warning(
        "OpenTelemetryTracerPlugin: 'opentelemetry-sdk' or 'opentelemetry-api' "
        "not installed. This tracer will be a no-op."
    )


class OpenTelemetryTracerPlugin(InteractionTracerPlugin):
    plugin_id: str = "otel_tracer_plugin_v1"
    description: str = "Sends trace events using OpenTelemetry."

    _tracer: Optional["trace.Tracer"] = None
    _provider: Optional[TracerProvider] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not OTEL_AVAILABLE or not trace or not TracerProvider or not Resource or not BatchSpanProcessor or not ConsoleSpanExporter:
            logger.warning(f"{self.plugin_id}: OpenTelemetry libraries not found or core components missing. This tracer will be a no-op.")
            return

        cfg = config or {}
        service_name = cfg.get("otel_service_name", "genie-tooling-application")
        service_version = cfg.get("otel_service_version", __version__)
        exporter_type = cfg.get("exporter_type", "console").lower()

        base_resource_attributes = {
            "service.name": service_name,
            "service.version": service_version,
        }
        custom_resource_attrs = cfg.get("resource_attributes", {})
        final_resource_attributes = {**base_resource_attributes, **custom_resource_attrs}

        resource = Resource.create(final_resource_attributes)
        self._provider = TracerProvider(resource=resource)

        exporter: Any = None
        if exporter_type == "otlp_http":
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHttpSpanExporter # type: ignore
                http_endpoint = cfg.get("otlp_http_endpoint", "http://localhost:4318/v1/traces")
                http_headers_str = cfg.get("otlp_http_headers")
                http_headers_dict: Optional[Dict[str, str]] = None
                if isinstance(http_headers_str, dict):
                    http_headers_dict = http_headers_str
                elif isinstance(http_headers_str, str) and http_headers_str.strip():
                    try:
                        http_headers_dict = dict(item.split("=", 1) for item in http_headers_str.split(",") if "=" in item)
                    except ValueError:
                        logger.warning(f"{self.plugin_id}: Could not parse otlp_http_headers string '{http_headers_str}'. Expected format 'k1=v1,k2=v2'.")
                http_timeout = int(cfg.get("otlp_http_timeout", 10))
                exporter = OTLPHttpSpanExporter(endpoint=http_endpoint, headers=http_headers_dict, timeout=http_timeout)
                logger.info(f"{self.plugin_id}: Using OTLP HTTP Exporter to {http_endpoint}")
            except ImportError:
                logger.error(f"{self.plugin_id}: 'opentelemetry-exporter-otlp-proto-http' not installed. Cannot use OTLP HTTP exporter. Falling back to console.")
                exporter = ConsoleSpanExporter()
            except Exception as e_http_exp:
                logger.error(f"{self.plugin_id}: Failed to configure OTLP HTTP Exporter: {e_http_exp}. Falling back to console.", exc_info=True)
                exporter = ConsoleSpanExporter()
        elif exporter_type == "otlp_grpc":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPGrpcSpanExporter # type: ignore
                grpc_endpoint = cfg.get("otlp_grpc_endpoint", "localhost:4317")
                grpc_insecure = bool(cfg.get("otlp_grpc_insecure", False))
                grpc_timeout = int(cfg.get("otlp_grpc_timeout", 10))
                # TODO: Handle credentials (e.g., cfg.get("otlp_grpc_credentials_ssl_file_path"))
                exporter = OTLPGrpcSpanExporter(endpoint=grpc_endpoint, insecure=grpc_insecure, timeout=grpc_timeout)
                logger.info(f"{self.plugin_id}: Using OTLP gRPC Exporter to {grpc_endpoint}")
            except ImportError:
                logger.error(f"{self.plugin_id}: 'opentelemetry-exporter-otlp-proto-grpc' not installed. Cannot use OTLP gRPC exporter. Falling back to console.")
                exporter = ConsoleSpanExporter()
            except Exception as e_grpc_exp:
                logger.error(f"{self.plugin_id}: Failed to configure OTLP gRPC Exporter: {e_grpc_exp}. Falling back to console.", exc_info=True)
                exporter = ConsoleSpanExporter()
        else:
            exporter = ConsoleSpanExporter()
            logger.info(f"{self.plugin_id}: Using Console Exporter.")

        processor = BatchSpanProcessor(exporter)
        self._provider.add_span_processor(processor)
        trace.set_tracer_provider(self._provider)
        self._tracer = trace.get_tracer(self.plugin_id, __version__)
        logger.info(f"{self.plugin_id}: OpenTelemetry Tracer initialized with service '{service_name}'. Exporter: {exporter_type}.")


    async def record_trace(self, event: TraceEvent) -> None:
        if not self._tracer or not trace: # Check trace module as well
            logger.debug(f"{self.plugin_id}: Tracer not available, skipping trace for event '{event['event_name']}'.")
            return

        start_time_ns = int(event["timestamp"] * 1_000_000_000)

        with self._tracer.start_as_current_span(event["event_name"], start_time=start_time_ns) as span:
            if event.get("component"):
                span.set_attribute("component", event["component"])
            if event.get("correlation_id"):
                span.set_attribute("correlation_id", event["correlation_id"])

            for key, value in event.get("data", {}).items():
                try:
                    if key == "llm.usage" and isinstance(value, dict):
                        for token_key, token_val in value.items():
                            if token_val is not None and isinstance(token_val, (int, float)):
                                span.set_attribute(f"llm.usage.{token_key}", int(token_val))
                        continue

                    if isinstance(value, (str, bool, int, float)):
                        span.set_attribute(f"data.{key}", value)
                    elif isinstance(value, (list, tuple)) and all(isinstance(item, (str, bool, int, float)) for item in value):
                        span.set_attribute(f"data.{key}", [str(v) for v in value])
                    else:
                        span.set_attribute(f"data.{key}", json.dumps(value, default=str))
                except Exception as e_attr:
                    logger.debug(f"{self.plugin_id}: Could not set attribute 'data.{key}' with value '{str(value)[:50]}...': {e_attr}")
                    span.set_attribute(f"data.{key}", "[UnserializableValue]")

            error_message = event["data"].get("error_message")
            error_type = event["data"].get("error_type")
            error_stacktrace = event["data"].get("error_stacktrace")

            if error_message or error_type:
                description = error_message or "Unknown error"
                span.set_status(trace.Status(trace.StatusCode.ERROR, description=description))
                if error_type:
                    span.set_attribute("error.type", error_type)
                if error_message:
                    span.set_attribute("error.message", error_message)
                if error_stacktrace:
                    span.set_attribute("error.stacktrace", error_stacktrace)

    async def teardown(self) -> None:
        if self._provider:
            try:
                self._provider.shutdown()
                logger.info(f"{self.plugin_id}: OpenTelemetry TracerProvider shut down.")
            except Exception as e:
                logger.error(f"{self.plugin_id}: Error shutting down TracerProvider: {e}", exc_info=True)
        self._tracer = None
        self._provider = None
        logger.debug(f"{self.plugin_id}: Teardown complete.")
