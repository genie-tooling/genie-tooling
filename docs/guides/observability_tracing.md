# Observability and Interaction Tracing (`genie.observability`)

Genie Tooling includes an observability interface, `genie.observability`, primarily for interaction tracing. This allows developers to record key events and data points as an agent processes requests, aiding in debugging, monitoring, and understanding agent behavior.

## Core Concepts

*   **`ObservabilityInterface` (`genie.observability`)**: The facade interface for tracing events.
*   **`InteractionTracerPlugin`**: A plugin responsible for handling trace events (e.g., printing to console, sending to an OpenTelemetry collector, logging to a file/database).
    *   Built-in: `ConsoleTracerPlugin` (alias: `console_tracer`), `OpenTelemetryTracerPlugin` (alias: `otel_tracer`, currently a stub).
*   **`TraceEvent` (TypedDict)**: The structure for trace data:
    ```python
    class TraceEvent(TypedDict):
        event_name: str  # e.g., "llm.chat.start", "tool.execute.success"
        data: Dict[str, Any] # Event-specific payload
        timestamp: float # time.time() or loop.time()
        component: Optional[str] # e.g., "LLMInterface", "ToolInvoker:calculator_tool"
        correlation_id: Optional[str] # For linking related events
    ```

## Configuration

You configure the default interaction tracer via `FeatureSettings` or explicit `MiddlewareConfig` settings.

**Via `FeatureSettings`:**

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    features=FeatureSettings(
        # ... other features ...
        observability_tracer="console_tracer", # Default: prints to console
        # observability_tracer="otel_tracer",
        # observability_otel_endpoint="http://localhost:4318/v1/traces" # If using OTel
    ),
    # Example: Configure the ConsoleTracerPlugin
    observability_tracer_configurations={
        "console_tracer_plugin_v1": { # Canonical ID
            "log_level": "DEBUG" # Log traces at DEBUG level
        }
    }
)
```

**Explicit Configuration:**

```python
app_config = MiddlewareConfig(
    default_observability_tracer_id="console_tracer_plugin_v1",
    observability_tracer_configurations={
        "console_tracer_plugin_v1": {"log_level": "INFO"}
    }
)
```

## Automatic Tracing

Several core `Genie` methods are automatically instrumented to emit trace events:
*   `genie.llm.chat()` and `genie.llm.generate()`
*   `genie.rag.index_directory()`, `genie.rag.index_web_page()`, `genie.rag.search()`
*   `genie.execute_tool()`
*   `genie.run_command()` (including sub-steps like HITL requests/responses)
*   `genie.register_tool_functions()`
*   `genie.close()`

These automatic traces provide a good baseline for understanding the flow of operations.

## Manual Tracing with `genie.observability.trace_event()`

You can emit custom trace events from your application logic or custom plugins:

```python
import uuid

async def my_custom_agent_step(genie, user_input: str):
    correlation_id = str(uuid.uuid4()) # Generate a unique ID for this interaction flow
    
    await genie.observability.trace_event(
        event_name="my_agent.step.start",
        data={"input_length": len(user_input)},
        component="MyCustomAgent",
        correlation_id=correlation_id
    )

    # ... agent logic ...
    processed_data = user_input.upper()

    await genie.observability.trace_event(
        event_name="my_agent.step.end",
        data={"output_length": len(processed_data), "status": "success"},
        component="MyCustomAgent",
        correlation_id=correlation_id
    )
    return processed_data
```

**Parameters for `trace_event`:**
*   `event_name` (str): A dot-separated name for the event (e.g., `myapp.module.action`).
*   `data` (Dict[str, Any]): A dictionary of serializable data related to the event.
*   `component` (Optional[str]): Name of the component emitting the event.
*   `correlation_id` (Optional[str]): An ID to link multiple events part of the same logical operation or request flow.

## Creating Custom Tracer Plugins

Implement the `InteractionTracerPlugin` protocol, defining the `record_trace(event: TraceEvent)` method. This method will receive `TraceEvent` dictionaries and should handle their persistence or transmission. Register your plugin via entry points or `plugin_dev_dirs`.
