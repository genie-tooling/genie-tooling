# Token Usage Tracking (`genie.usage`)

Genie Tooling provides an interface for tracking token usage by LLM providers, accessible via `genie.usage`. This helps in monitoring costs and understanding LLM consumption patterns.

## Core Concepts

*   **`UsageTrackingInterface` (`genie.usage`)**: The facade interface for recording and summarizing token usage.
*   **`TokenUsageRecorderPlugin`**: A plugin responsible for storing or exporting token usage records.
    *   Built-in:
        *   `InMemoryTokenUsageRecorderPlugin` (alias: `in_memory_token_recorder`): Stores records in memory.
        *   `OpenTelemetryMetricsTokenRecorderPlugin` (alias: `otel_metrics_recorder`): Emits token counts as OpenTelemetry metrics.
*   **`TokenUsageRecord` (TypedDict)**: The structure for a single token usage event:
    ```python
    class TokenUsageRecord(TypedDict, total=False):
        provider_id: str
        model_name: str
        prompt_tokens: Optional[int]
        completion_tokens: Optional[int]
        total_tokens: Optional[int]
        timestamp: float # time.time()
        call_type: Optional[str] # "chat", "generate", "generate_stream_end", "chat_stream_end"
        user_id: Optional[str]
        session_id: Optional[str]
        custom_tags: Optional[dict]
    ```

## Configuration

Configure the default token usage recorder via `FeatureSettings` or explicit `MiddlewareConfig`.

**Via `FeatureSettings`:**

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    features=FeatureSettings(
        # ... other features ...
        token_usage_recorder="in_memory_token_recorder" # Default in-memory
        # OR to use OTel Metrics:
        # token_usage_recorder="otel_metrics_recorder",
        # observability_tracer="otel_tracer" # OTel SDK needs to be initialized
    )
    # Example: Configure the OpenTelemetryMetricsTokenRecorderPlugin if chosen
    # (This plugin itself has no specific config options, but relies on OTel SDK setup)
    # token_usage_recorder_configurations={
    #     "otel_metrics_token_recorder_v1": {}
    # },
    # observability_tracer_configurations={ # Ensure OTel SDK is initialized
    #     "otel_tracer_plugin_v1": {
    #         "otel_service_name": "my-app-with-token-metrics",
    #         "exporter_type": "console" # Or your preferred OTel exporter
    #     }
    # }
)
```

## Automatic Recording

When a `TokenUsageManager` is active (i.e., `features.token_usage_recorder` is not `"none"`), token usage is automatically recorded by:
*   `genie.llm.chat()`
*   `genie.llm.generate()`

The LLM provider plugins are responsible for returning `LLMUsageInfo` in their responses (or in the final chunk of a stream). The `LLMInterface` then uses this information to create and record `TokenUsageRecord`s. The `call_type` will distinguish between regular calls and the end of streaming calls (e.g., `generate_stream_end`, `chat_stream_end`).

## Manual Usage

### 1. Getting a Usage Summary (for `in_memory_token_recorder`)

You can retrieve a summary of recorded token usage if using a recorder that supports it (like `InMemoryTokenUsageRecorderPlugin`).

```python
# Assuming genie is initialized with token_usage_recorder="in_memory_token_recorder"
summary = await genie.usage.get_summary()
# summary = await genie.usage.get_summary(recorder_id="my_custom_recorder") # If using a specific recorder
# summary = await genie.usage.get_summary(filter_criteria={"user_id": "user123"}) # Filter

print(f"Total records: {summary.get('total_records')}")
print(f"Total tokens overall: {summary.get('total_tokens_overall')}")
if summary.get("by_model"):
    for model, data in summary["by_model"].items():
        print(f"  Model: {model}, Total Tokens: {data['total']}, Count: {data['count']}")
```
The `OpenTelemetryMetricsTokenRecorderPlugin` will log a warning if `get_summary()` is called, as metrics are viewed in an OTel backend.

### 2. Manually Recording Usage (Advanced)

While most recording is automatic, you can manually record usage if needed:
```python
from genie_tooling.token_usage.types import TokenUsageRecord
import time

manual_record = TokenUsageRecord(
    provider_id="my_external_llm_service",
    model_name="external_model_vX",
    prompt_tokens=1000,
    completion_tokens=500,
    total_tokens=1500,
    timestamp=time.time(),
    call_type="custom_batch_job",
    user_id="batch_processor"
)
await genie.usage.record_usage(manual_record)
```

## Using `OpenTelemetryMetricsTokenRecorderPlugin`

When `token_usage_recorder="otel_metrics_recorder"` is configured, this plugin will emit the following OTel metrics:
*   `llm.request.tokens.prompt` (Counter, unit: `1` {token})
*   `llm.request.tokens.completion` (Counter, unit: `1` {token})
*   `llm.request.tokens.total` (Counter, unit: `1` {token})

These metrics will have attributes like `llm.provider.id`, `llm.model.name`, `llm.call_type`, `genie.client.user_id`, `genie.client.session_id`, and any `genie.tag.*` from custom tags. Configure an OTel collector (e.g., with Prometheus exporter) to scrape and visualize these metrics.

**Example PromQL Queries:**
*   Total prompt tokens per model (rate over 5m): `sum(rate(llm_request_tokens_prompt_total[5m])) by (llm_model_name)`
*   Total completion tokens per provider (rate over 5m): `sum(rate(llm_request_tokens_completion_total[5m])) by (llm_provider_id)`

## Creating Custom Token Usage Recorders

Implement the `TokenUsageRecorderPlugin` protocol, defining `record_usage`, `get_summary`, and `clear_records` methods to interact with your chosen storage backend (e.g., database, logging service). Register your plugin via entry points or `plugin_dev_dirs`.
