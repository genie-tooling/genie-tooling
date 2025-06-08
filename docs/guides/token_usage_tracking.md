# Token Usage Tracking (`genie.usage`)

Genie Tooling provides an interface for tracking token usage by LLM providers, accessible via `genie.usage`. This helps in monitoring costs and understanding LLM consumption patterns.

## Core Concepts

*   **`UsageTrackingInterface` (`genie.usage`)**: The facade interface for recording and summarizing token usage.
*   **`TokenUsageRecorderPlugin`**: A plugin responsible for storing or exporting token usage records.
    *   Built-in:
        *   `InMemoryTokenUsageRecorderPlugin` (alias: `in_memory_token_recorder`): Stores records in memory. Useful for simple summaries and testing.
        *   `OpenTelemetryMetricsTokenRecorderPlugin` (alias: `otel_metrics_recorder`): **Recommended for production.** Emits token counts as standard OpenTelemetry metrics, which can be scraped by systems like Prometheus.
*   **`TokenUsageRecord` (TypedDict)**: The structure for a single token usage event.

## Configuration

Configure the default token usage recorder via `FeatureSettings`.

### Example 1: In-Memory Recorder (for Development)

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    features=FeatureSettings(
        # ... other features ...
        token_usage_recorder="in_memory_token_recorder"
    )
)
```

### Example 2: OpenTelemetry Metrics Recorder (for Production)

This is the recommended approach for production monitoring. It requires an OpenTelemetry collector setup.

```python
# Prerequisite: An OTel collector that can scrape Prometheus metrics.
app_config = MiddlewareConfig(
    features=FeatureSettings(
        # ... other features ...
        token_usage_recorder="otel_metrics_recorder",
        # The OTel SDK must be initialized, which is done by the OTel tracer.
        # So, enable the tracer, even if you only care about metrics.
        observability_tracer="otel_tracer",
    ),
    # Configure the OTel tracer to initialize the SDK
    observability_tracer_configurations={
        "otel_tracer_plugin_v1": {
            "otel_service_name": "my-app-with-token-metrics",
            "exporter_type": "console" # Or your preferred OTel trace exporter
        }
    }
)
```

## How It Works

Token usage is **automatically recorded** by `genie.llm.chat()` and `genie.llm.generate()` whenever the underlying LLM provider returns usage information.

### Using the In-Memory Recorder

If you use `"in_memory_token_recorder"`, you can get a simple summary:

```python
# Assuming genie is initialized with the in-memory recorder
summary = await genie.usage.get_summary()

# Example output:
# {
#   "in_memory_token_usage_recorder_v1": {
#     "total_records": 5,
#     "total_prompt_tokens": 1234,
#     "total_completion_tokens": 567,
#     "total_tokens_overall": 1801,
#     "by_model": {
#       "mistral:latest": { "prompt": 1234, "completion": 567, "total": 1801, "count": 5 }
#     }
#   }
# }
```

### Using the OpenTelemetry Metrics Recorder

When `token_usage_recorder="otel_metrics_recorder"` is configured, this plugin emits the following OTel metrics:
*   `llm.request.tokens.prompt` (Counter, unit: `{token}`)
*   `llm.request.tokens.completion` (Counter, unit: `{token}`)
*   `llm.request.tokens.total` (Counter, unit: `{token}`)

These metrics will have attributes (labels) like `llm.provider.id`, `llm.model.name`, and `llm.call_type`. Configure an OTel collector (e.g., with a Prometheus exporter) to scrape and visualize these metrics in a dashboarding tool like Grafana.

**Example PromQL Queries:**
*   Total prompt tokens per model (rate over 5m): `sum(rate(llm_request_tokens_prompt_total[5m])) by (llm_model_name)`
*   Total completion tokens per provider (rate over 5m): `sum(rate(llm_request_tokens_completion_total[5m])) by (llm_provider_id)`
