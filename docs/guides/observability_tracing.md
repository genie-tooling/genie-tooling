# Observability and Interaction Tracing (`genie.observability`)

Genie Tooling includes an observability interface, `genie.observability`, primarily for interaction tracing. This allows developers to record key events and data points as an agent processes requests, aiding in debugging, monitoring, and understanding agent behavior. Traces can be exported to various backends using OpenTelemetry.

## Core Concepts

*   **`ObservabilityInterface` (`genie.observability`)**: The facade interface for tracing events.
*   **`InteractionTracerPlugin`**: A plugin responsible for handling trace events.
    *   Built-in:
        *   `ConsoleTracerPlugin` (alias: `console_tracer`): Prints trace events to the console.
        *   `OpenTelemetryTracerPlugin` (alias: `otel_tracer`): Exports traces using the OpenTelemetry SDK.
*   **`TraceEvent` (TypedDict)**: The structure for trace data:
    ```python
    class TraceEvent(TypedDict):
        event_name: str  # e.g., "llm.chat.start", "tool.execute.success"
        data: Dict[str, Any] # Event-specific payload
        timestamp: float # time.time() or loop.time()
        component: Optional[str] # e.g., "LLMInterface", "ToolInvoker:calculator_tool"
        correlation_id: Optional[str] # For linking related events
    ```
    The `data` field may contain an `error_message`, `error_type`, and `error_stacktrace` if an error occurred. For LLM events, it may contain `llm.usage` with token counts (if `OpenTelemetryTracerPlugin` is used and token usage is recorded).

## Configuration

Configure the default interaction tracer via `FeatureSettings` or explicit `MiddlewareConfig` settings.

**Via `FeatureSettings`:**

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    features=FeatureSettings(
        # ... other features ...
        observability_tracer="console_tracer", # Default: prints to console
        # OR for OpenTelemetry:
        # observability_tracer="otel_tracer",
        # observability_otel_endpoint="http://localhost:4318/v1/traces" # For OTLP/HTTP
    ),
    # Example: Configure the ConsoleTracerPlugin
    observability_tracer_configurations={
        "console_tracer_plugin_v1": { # Canonical ID
            "log_level": "DEBUG" # Log traces at DEBUG level
        }
    }
    # Example: Configure OpenTelemetryTracerPlugin
    # observability_tracer_configurations={
    #     "otel_tracer_plugin_v1": {
    #         "otel_service_name": "my-genie-app",
    #         "otel_service_version": "1.2.3",
    #         "exporter_type": "otlp_http", # "console", "otlp_grpc"
    #         "otlp_http_endpoint": "http://localhost:4318/v1/traces",
    #         # "otlp_http_headers": "Authorization=Bearer mytoken,X-Custom-Header=value",
    #         # "otlp_grpc_endpoint": "localhost:4317",
    #         # "otlp_grpc_insecure": True, # Use False if your collector uses TLS
    #         "resource_attributes": {"deployment.environment": "staging"}
    #     }
    # }
)
```

### `OpenTelemetryTracerPlugin` Configuration Options

When `observability_tracer="otel_tracer"` is chosen, or `default_observability_tracer_id="otel_tracer_plugin_v1"` is set, you can configure it in `observability_tracer_configurations["otel_tracer_plugin_v1"]`:

*   `otel_service_name` (str, default: "genie-tooling-application"): The name of your service as it will appear in traces.
*   `otel_service_version` (str, default: Genie Tooling library version): Version of your service.
*   `exporter_type` (str, default: "console"):
    *   `"console"`: Prints spans to the console (useful for local debugging).
    *   `"otlp_http"`: Exports spans via OTLP/HTTP protocol. Requires `opentelemetry-exporter-otlp-proto-http`.
    *   `"otlp_grpc"`: Exports spans via OTLP/gRPC protocol. Requires `opentelemetry-exporter-otlp-proto-grpc`.
*   `otlp_http_endpoint` (str, default: "http://localhost:4318/v1/traces"): Endpoint for OTLP/HTTP exporter.
*   `otlp_http_headers` (Optional[Dict[str,str]] or str like "k1=v1,k2=v2"): Custom headers for OTLP/HTTP.
*   `otlp_http_timeout` (int, default: 10): Timeout in seconds for OTLP/HTTP.
*   `otlp_grpc_endpoint` (str, default: "localhost:4317"): Endpoint for OTLP/gRPC exporter.
*   `otlp_grpc_insecure` (bool, default: `False`): Whether to use an insecure gRPC connection.
*   `otlp_grpc_timeout` (int, default: 10): Timeout in seconds for OTLP/gRPC.
*   `resource_attributes` (Optional[Dict[str, Any]]): Additional attributes to add to the OTel Resource (e.g., `{"deployment.environment": "production"}`).

## Automatic Tracing

Core `Genie` methods are automatically instrumented. Key attributes often included:
*   `component`: The Genie component emitting the trace (e.g., `LLMInterface`, `ToolInvoker:my_tool`).
*   `correlation_id`: Links events within a single logical operation (e.g., one `genie.run_command()` call).
*   `llm.usage.prompt_tokens`, `llm.usage.completion_tokens`, `llm.usage.total_tokens`: For LLM call success events (when using `OpenTelemetryTracerPlugin` and token usage is available from the LLM provider).
*   `error.type`, `error.message`, `error.stacktrace`: If an error occurs during the traced operation.

## Manual Tracing

Use `await genie.observability.trace_event(...)` for custom application events.

```python
import uuid
# await genie.observability.trace_event(
#     event_name="my_app.custom_process.start",
#     data={"input_id": "123", "user_category": "premium"},
#     component="MyCustomModule",
#     correlation_id=str(uuid.uuid4())
# )
```

## Viewing Traces

When using OTLP exporters, configure them to point to an OpenTelemetry collector (e.g., OTel Collector, Jaeger All-in-One, Grafana Agent, SigNoz, or a cloud vendor's OTel endpoint). The collector will then process and forward traces to your chosen backend (Jaeger, Prometheus, Zipkin, etc.) for visualization and analysis.
