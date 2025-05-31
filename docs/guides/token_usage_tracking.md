# Token Usage Tracking (`genie.usage`)

Genie Tooling provides an interface for tracking token usage by LLM providers, accessible via `genie.usage`. This helps in monitoring costs and understanding LLM consumption patterns.

## Core Concepts

*   **`UsageTrackingInterface` (`genie.usage`)**: The facade interface for recording and summarizing token usage.
*   **`TokenUsageRecorderPlugin`**: A plugin responsible for storing token usage records.
    *   Built-in: `InMemoryTokenUsageRecorderPlugin` (alias: `in_memory_token_recorder`).
*   **`TokenUsageRecord` (TypedDict)**: The structure for a single token usage event:
    ```python
    class TokenUsageRecord(TypedDict, total=False):
        provider_id: str
        model_name: str
        prompt_tokens: Optional[int]
        completion_tokens: Optional[int]
        total_tokens: Optional[int]
        timestamp: float # time.time()
        call_type: Optional[str] # "chat", "generate"
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
        token_usage_recorder="in_memory_token_recorder" # Default
    )
    # No specific config needed for InMemoryTokenUsageRecorderPlugin by default
)
```

## Automatic Recording

When a `TokenUsageManager` is active (i.e., `features.token_usage_recorder` is not `"none"`), token usage is automatically recorded by:
*   `genie.llm.chat()`
*   `genie.llm.generate()`

The LLM provider plugins are responsible for returning `LLMUsageInfo` in their responses, which the `LLMInterface` then uses to create and record `TokenUsageRecord`s.

## Manual Usage

### 1. Getting a Usage Summary

You can retrieve a summary of recorded token usage:

```python
# Assuming genie is initialized with a token usage recorder
summary = await genie.usage.get_summary()
# summary = await genie.usage.get_summary(recorder_id="my_custom_recorder") # If using a specific recorder
# summary = await genie.usage.get_summary(filter_criteria={"user_id": "user123"}) # Filter

print(f"Total records: {summary.get('total_records')}")
print(f"Total tokens overall: {summary.get('total_tokens_overall')}")
if summary.get("by_model"):
    for model, data in summary["by_model"].items():
        print(f"  Model: {model}, Total Tokens: {data['total']}, Count: {data['count']}")

# Example output for InMemoryTokenUsageRecorderPlugin:
# {
#     'total_records': 2,
#     'total_prompt_tokens': 150,
#     'total_completion_tokens': 350,
#     'total_tokens_overall': 500,
#     'by_model': {
#         'gpt-3.5-turbo': {'prompt': 100, 'completion': 200, 'total': 300, 'count': 1},
#         'mistral:latest': {'prompt': 50, 'completion': 150, 'total': 200, 'count': 1}
#     }
# }
```
The structure of the summary depends on the `TokenUsageRecorderPlugin` implementation. The `InMemoryTokenUsageRecorderPlugin` provides totals and a breakdown by model.

### 2. Manually Recording Usage (Advanced)

While most recording is automatic, you can manually record usage if needed, for example, if you are interacting with an LLM outside of `genie.llm` but still want to track its usage within Genie's system.

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

## Creating Custom Token Usage Recorders

Implement the `TokenUsageRecorderPlugin` protocol, defining `record_usage`, `get_summary`, and `clear_records` methods to interact with your chosen storage backend (e.g., database, logging service). Register your plugin via entry points or `plugin_dev_dirs`.
