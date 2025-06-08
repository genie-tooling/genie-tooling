# Observability and Interaction Tracing (`genie.observability`)

Genie Tooling is designed for production use, and a key requirement for any production system is deep visibility into its internal workings. The framework is heavily instrumented to provide "zero-effort" observability. By simply enabling a tracer plugin, you can get detailed, correlated traces for the entire lifecycle of an agentic operation, from command processing and tool lookup to LLM calls and tool execution.

## Core Concepts

*   **`ObservabilityInterface` (`genie.observability`)**: The facade interface for tracing events.
*   **`InteractionTracerPlugin`**: A plugin responsible for handling trace events.
    *   Built-in:
        *   `ConsoleTracerPlugin` (alias: `console_tracer`): Prints trace events to the console.
        *   `OpenTelemetryTracerPlugin` (alias: `otel_tracer`): Exports traces using the OpenTelemetry SDK to backends like Jaeger, SigNoz, etc.
*   **`LogAdapterPlugin`**: A plugin that formats trace events for output. The `ConsoleTracerPlugin` delegates its formatting to the configured `LogAdapterPlugin` (e.g., `DefaultLogAdapter` or `PyviderTelemetryLogAdapter`). This decouples tracing from formatting.
*   **`TraceEvent` (TypedDict)**: The raw, structured data for a trace event.
*   **Correlation ID**: A unique ID that links all events within a single top-level operation (e.g., one `genie.run_command()` call), allowing observability backends to construct a complete trace hierarchy.

## Automatic Tracing

When a tracer is enabled, the framework automatically emits detailed events for:
*   **Facade Calls**: Start and end of `genie.run_command`, `genie.execute_tool`, `genie.rag.index_*`, `genie.rag.search`, etc.
*   **Command Processing**: The full lifecycle, including which tools were considered (`tool_lookup.end`), the prompt sent to the LLM (`command_processor.llm_assisted.prompt_context_ready`), and the LLM's parsed decision (`command_processor.llm_assisted.result`).
*   **Tool Invocation**: Caching checks (`invocation.cache.hit/miss`), input validation, the actual tool execution (`tool.execute.start/end`), and output transformation.
*   **LLM Calls**: All `genie.llm.chat/generate` calls, including the provider used, all parameters (`temperature`, `model`, etc.), and token usage from the response.

## Configuration

Configure the default interaction tracer via `FeatureSettings`.

### Example 1: Simple Console Tracing
This is the easiest way to see what's happening during development.

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    features=FeatureSettings(
        # ... other features ...
        observability_tracer="console_tracer",
    ),
    # Optional: Configure the log adapter used by the console tracer
    log_adapter_configurations={
        "default_log_adapter_v1": {
            "log_level": "DEBUG" # Set the level for the 'genie_tooling' logger
        }
    }
)
```

### Example 2: OpenTelemetry Tracing (for Jaeger, SigNoz, etc.)

This configuration exports traces to an OpenTelemetry collector.

```python
# Prerequisite: Start an OTel collector (e.g., Jaeger all-in-one)
# docker run -d --name jaeger -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one:latest
# With this setup, running your Genie application will send detailed traces to Jaeger, which you can view at `http://localhost:16686`.

## Application-Level Tracing

Beyond the automatic framework traces, Genie provides tools to seamlessly integrate your application's logic into the same trace.
=======

app_config = MiddlewareConfig(
    features=FeatureSettings(
        observability_tracer="otel_tracer",
        # The endpoint for the OTLP/HTTP exporter
        observability_otel_endpoint="http://localhost:4318/v1/traces"
    ),
    observability_tracer_configurations={
        "otel_tracer_plugin_v1": { # Canonical ID
            "otel_service_name": "my-genie-agent-app",
            "otel_service_version": "1.2.3",
            "resource_attributes": {"deployment.environment": "staging"}
        }
    }
)
```

### Simplified Tracing with the `@traceable` Decorator

For the common case of tracing an entire function call, Genie provides the `@traceable` decorator. This is the recommended approach for instrumenting functions within your tools.

**How it works:**
*   It automatically creates a new OpenTelemetry span when the decorated function is called.
*   It links this new span to the parent span found in the `context` argument.
*   It records function arguments as span attributes.
*   It automatically records exceptions and sets the span status to `ERROR`.

**Example:**

```python
from genie_tooling import tool
from genie_tooling.observability import traceable
from typing import Dict, Any

@traceable
async def _perform_database_query(query: str, context: Dict[str, Any]):
    # A span for '_perform_database_query' is automatically created
    # and linked to the parent 'get_user_data.execute' span.
    # The 'query' argument will be added as an attribute to the span.
    # ... database logic ...
    return {"id": 123, "name": "John Doe"}

@tool
class UserDataTool:
    # ... (plugin_id, get_metadata, etc.) ...
    async def execute(self, params: Dict[str, Any], key_provider: Any, context: Dict[str, Any]) -> Any:
        user_id = params.get("user_id")
        
        # The context dictionary received here contains the OTel context,
        # which is automatically passed to the @traceable function.
        db_result = await _perform_database_query(
            query=f"SELECT * FROM users WHERE id={user_id}",
            context=context
        )
        return db_result
```

### Context Propagation and Auto-Instrumentation

The `@traceable` decorator works because Genie automatically propagates the OpenTelemetry `Context` object. When Genie's `ToolInvoker` calls your tool's `execute` method, the `context` dictionary it passes now contains a special key, `otel_context`.

This seamless context propagation means that standard OpenTelemetry auto-instrumentation libraries (e.g., `opentelemetry-instrumentation-httpx`, `opentelemetry-instrumentation-psycopg2`) will work out-of-the-box. If your traceable function makes a call using an instrumented library, that library will automatically create a child span, giving you an incredibly detailed, end-to-end trace with zero extra effort.
While most tracing is automatic, you can add custom events from your own application logic using `await genie.observability.trace_event(...)`.

```python
import uuid
# correlation_id = str(uuid.uuid4()) # Start a new logical operation
# await genie.observability.trace_event(
#     event_name="my_app.custom_process.start",
#     data={"input_id": "123", "user_category": "premium"},
#     component="MyCustomModule",
#     correlation_id=correlation_id
# )
```
This allows your application-specific events to appear within the same trace as the framework's internal events, providing a complete picture.
