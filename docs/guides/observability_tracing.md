# Observability and Interaction Tracing (`genie.observability`)

Genie Tooling is designed for production use, and a key requirement for any production system is deep visibility into its internal workings. The framework is heavily instrumented to provide "zero-effort" observability. By simply enabling a tracer plugin, you can get detailed, correlated traces for the entire lifecycle of an agentic operation.

## Core Concepts

*   **`ObservabilityInterface` (`genie.observability`)**: The facade interface for tracing events.
*   **`InteractionTracerPlugin`**: A plugin responsible for handling trace events.
    *   `ConsoleTracerPlugin` (alias: `console_tracer`): Prints trace events to the console via the configured Log Adapter.
    *   `OpenTelemetryTracerPlugin` (alias: `otel_tracer`): Exports traces using the OpenTelemetry SDK to backends like Jaeger, SigNoz, DataDog, etc.
*   **`@traceable` Decorator**: A powerful decorator to automatically trace custom functions and link them to the ongoing operation trace.

## Automatic Tracing

When a tracer is enabled, the framework automatically emits detailed events for every major operation, including facade calls, command processing, tool lookup, tool execution, LLM API calls, guardrail decisions, and λ-CQS decision records.

## Audit Events for Corporate Harnesses

Three structured trace events form the audit-trail backbone:

### `audit.decision_record`

Emitted **exactly once per `genie.context.resolve_and_formulate(...)` call**. The
single most important event for audit-bound deployments — it joins
every signal from the λ-CQS pipeline into one structured record. The
same record is also available in-process via
`genie.context.last_decision`.

Schema (see `src/genie_tooling/context/audit.py` for the canonical
`DecisionRecord` dataclass):

| Field | Meaning |
|---|---|
| `decision_id`, `session_id`, `user_identity`, `query` | Inputs |
| `inferred_context` | The inference plugin's view of who is asking |
| `predicate`, `predicate_extractor_id` | Predicate extraction step |
| `rule_engine_id`, `ranked_rules`, `winning_rule_id` | Rule evaluation step — **the full ranked list**, not just the winner |
| `c_d`, `c_f` | Aggregated derivation + formulation constraint dicts |
| `derivation_strategy_id`, `derivation_status`, `derivation_result_preview` | Derivation step |
| `formulation_strategy_id`, `formulation_template_id`, `formulation_constraints_text` | Formulation step — **the exact instruction text the LLM saw** |
| `final_response` | What was sent back to the caller |
| `stage_timings_ms` | Per-stage latency breakdown |
| `started_at`, `completed_at` | Wall-clock bracket |

See the [Context Scoping guide](context_scoping.md) for a walkthrough.

### `guardrail.decision`

Emitted whenever a guardrail blocks or warns about input/output. The
event carries `guardrail_id`, `reason`, `actor` (which session/user
triggered it), `decision` (`"block"` / `"warn"`), and a truncated
`trigger_preview`. Use `decision_id` to join to the parent
`audit.decision_record`.

### Tool execution provenance (`caller_chain`)

Every `genie.execute_tool` trace event carries:

- `tool_id`, formatted `params`, the `result`, duration
- `key_provider_identity` — which secret was used
- **`caller_chain`** — list of `(component_id, correlation_id)` tuples
  showing the nesting: e.g. `[("react_agent", "...") , ("rule_engine.filesystem_rule_engine_v1", "..."), ("genie", "...")]`
- `parent_correlation_id` — for joining to the immediate parent span

`caller_chain` is the answer to "which agent, in which rule, in which
Genie call invoked this tool?".

## Configuration

Configure the default interaction tracer via `FeatureSettings`.

### Example 1: Console Tracing with Pyvider
This provides rich, structured, and emoji-enhanced logs in your console for easy debugging.

```python
app_config = MiddlewareConfig(
    features=FeatureSettings(
        observability_tracer="console_tracer",
        logging_adapter="pyvider_log_adapter"
    ),
    log_adapter_configurations={
        "pyvider_telemetry_log_adapter_v1": {
            "service_name": "MyGenieApp",
            "default_level": "DEBUG",
        }
    }
)
```

### Example 2: OpenTelemetry Tracing (for Production)

This configuration exports traces to an OpenTelemetry collector.

```python
# Prerequisite: Start an OTel collector (e.g., Jaeger all-in-one)
# docker run -d --name jaeger -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one:latest

app_config = MiddlewareConfig(
    features=FeatureSettings(
        observability_tracer="otel_tracer",
        observability_otel_endpoint="http://localhost:4318/v1/traces" # OTLP/HTTP
    ),
    observability_tracer_configurations={
        "otel_tracer_plugin_v1": {
            "otel_service_name": "my-genie-agent-app",
            "otel_service_version": "1.2.3",
        }
    }
)
```
After running your application, you can view the detailed trace hierarchy in Jaeger at `http://localhost:16686`.

## Application-Level Tracing with `@traceable`

Beyond the automatic framework traces, the `@traceable` decorator is the recommended way to integrate your application's logic into the same trace.

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
async def get_user_data(user_id: int, context: Dict[str, Any]) -> Dict[str, Any]:
    # The context dictionary received here contains the OTel context,
    # which is automatically passed to the @traceable function.
    db_result = await _perform_database_query(
        query=f"SELECT * FROM users WHERE id={user_id}",
        context=context
    )
    return db_result
```
This pattern creates a nested trace where the `_perform_database_query` span appears as a child of the `get_user_data` tool execution span, providing a complete, detailed view of the operation.

### Context Propagation and Auto-Instrumentation

The `@traceable` decorator works because Genie automatically propagates the OpenTelemetry `Context` object. When Genie's `ToolInvoker` calls your tool's `execute` method, the `context` dictionary it passes now contains a special key, `otel_context`.

This seamless context propagation means that standard OpenTelemetry auto-instrumentation libraries (e.g., `opentelemetry-instrumentation-httpx`) will work out-of-the-box. If your traceable function makes a call using an instrumented library, that library will automatically create a child span, giving you an incredibly detailed, end-to-end trace with zero extra effort.
