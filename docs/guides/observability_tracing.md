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
        # error: Optional[Dict] # Optional structured error info
    ```
    The `data` field may contain an `error_message`, `error_type`, and `error_stacktrace` if an error occurred. For LLM events, it may contain `llm.usage` with token counts.

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
    #         # "otlp_grpc_endpoint": "localhost:4317",
    #         # "otlp_grpc_insecure": True,
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
    *   `"console"`: Prints spans to the console.
    *   `"otlp_http"`: Exports spans via OTLP/HTTP protocol.
    *   `"otlp_grpc"`: Exports spans via OTLP/gRPC protocol.
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
*   `llm.usage.prompt_tokens`, `llm.usage.completion_tokens`, `llm.usage.total_tokens`: For LLM call success events.
*   `error.type`, `error.message`, `error.stacktrace`: If an error occurs during the traced operation.

## Manual Tracing

Use `await genie.observability.trace_event(...)` for custom application events.

## Viewing Traces

When using OTLP exporters, configure them to point to an OpenTelemetry collector (e.g., OTel Collector, Jaeger All-in-One, Grafana Agent, SigNoz, or a cloud vendor's OTel endpoint). The collector will then process and forward traces to your chosen backend (Jaeger, Prometheus, Zipkin, etc.) for visualization and analysis.
###<END-docs/guides/observability_tracing.md>###

# --- examples/E17_observability_tracing_example.py ---
info "Updating examples/E17_observability_tracing_example.py..."
cat << 'EOF' > examples/E17_observability_tracing_example.py
# examples/E17_observability_tracing_example.py
"""
Example: Using Observability and Interaction Tracing (`genie.observability`)
---------------------------------------------------------------------------
This example demonstrates how to enable and use the interaction tracing
feature of Genie Tooling. It shows configuration for both ConsoleTracerPlugin
and the new OpenTelemetryTracerPlugin.

To Run with Console Tracer:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Run from the root of the project:
   `poetry run python examples/E17_observability_tracing_example.py`
   You should see trace logs in your console output.

To Run with OpenTelemetry Tracer (e.g., to Jaeger):
1. Ensure OTel dependencies are installed: `poetry install --extras observability`
2. Start an OTel collector (e.g., Jaeger all-in-one):
   `docker run -d --name jaeger -e COLLECTOR_OTLP_ENABLED=true -p 16686:16686 -p 4317:4317 -p 4318:4318 jaegertracing/all-in-one:latest`
3. Modify the `app_config` below to use `otel_tracer` and configure the OTLP endpoint.
4. Run the script. Traces should appear in Jaeger UI (http://localhost:16686).
"""
import asyncio
import logging
import uuid
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


async def run_observability_demo():
    print("--- Observability and Tracing Example ---")

    # Configure basic logging to see standard logs AND trace logs from ConsoleTracerPlugin
    logging.basicConfig(level=logging.INFO) # Set root logger to INFO
    # For more fine-grained control:
    # logging.getLogger("genie_tooling.observability.impl.console_tracer").setLevel(logging.DEBUG)
    # logging.getLogger("genie_tooling.observability.impl.otel_tracer").setLevel(logging.DEBUG)
    # logging.getLogger("genie_tooling").setLevel(logging.INFO)

    # --- Configuration for Console Tracer (Default) ---
    app_config_console = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama", 
            llm_ollama_model_name="mistral:latest", # Ensure ollama & mistral are available
            observability_tracer="console_tracer"
        ),
        observability_tracer_configurations={
            "console_tracer_plugin_v1": {
                "log_level": "INFO" 
            }
        }
    )

    # --- Configuration for OpenTelemetry Tracer (Example for OTLP/HTTP to Jaeger) ---
    # Uncomment and use this config to send traces to an OTel collector.
    # app_config_otel = MiddlewareConfig(
    #     features=FeatureSettings(
    #         llm="ollama",
    #         llm_ollama_model_name="mistral:latest",
    #         observability_tracer="otel_tracer"
    #     ),
    #     observability_tracer_configurations={
    #         "otel_tracer_plugin_v1": {
    #             "otel_service_name": "genie-e17-demo-app",
    #             "otel_service_version": "0.1.0",
    #             "exporter_type": "otlp_http", # or "otlp_grpc" or "console"
    #             "otlp_http_endpoint": "http://localhost:4318/v1/traces",
    #             # For OTLP gRPC:
    #             # "otlp_grpc_endpoint": "localhost:4317",
    #             # "otlp_grpc_insecure": True, # Use False if your collector uses TLS
    #             "resource_attributes": {"deployment.environment": "development_e17"}
    #         }
    #     }
    # )

    # Select the configuration to use:
    app_config = app_config_console
    # app_config = app_config_otel # Uncomment to use OTel

    genie: Optional[Genie] = None
    try:
        print(f"\nInitializing Genie with tracer: {app_config.features.observability_tracer}...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized! Automatic traces will now be generated.")

        # --- Operations that generate automatic traces ---
        print("\n--- Performing operations that trigger automatic traces ---")

        # 1. LLM Chat (will generate llm.chat.start, llm.chat.success/error)
        # This will also include llm.usage.* attributes if using otel_tracer
        try:
            chat_response = await genie.llm.chat([{"role": "user", "content": "Hello Tracer! Tell me a short story."}])
            print(f"LLM Response: {chat_response['message']['content'][:60]}...")
        except Exception as e_llm:
            print(f"LLM call failed (expected if Ollama not running): {e_llm}")


        # 2. Custom trace event
        print("\n--- Emitting a custom trace event ---")
        custom_correlation_id = str(uuid.uuid4())
        await genie.observability.trace_event(
            event_name="my_app.custom_operation.start",
            data={"user_id": "test_user", "input_param": "example_value"},
            component="MyApplicationLogic",
            correlation_id=custom_correlation_id
        )
        # Simulate some work
        await asyncio.sleep(0.1)
        # Simulate an error in the custom operation
        try:
            raise ValueError("Something went wrong in custom op!")
        except ValueError as e_custom:
            await genie.observability.trace_event(
                event_name="my_app.custom_operation.error",
                data={
                    "status": "failed",
                    "error_message": str(e_custom),
                    "error_type": type(e_custom).__name__,
                    # "error_stacktrace": traceback.format_exc() # If you want to include stack
                    },
                component="MyApplicationLogic",
                correlation_id=custom_correlation_id
            )
        print("Custom trace events (including an error) emitted.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("Observability demo error details:")
    finally:
        if genie:
            # Teardown also generates a trace event
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    asyncio.run(run_observability_demo())
