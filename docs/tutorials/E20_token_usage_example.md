# Tutorial: Token Usage Tracking Example

This tutorial corresponds to the example file `examples/E20_token_usage_example.py`.

It demonstrates how to:
- Enable and configure a `TokenUsageRecorderPlugin` (e.g., `InMemoryTokenUsageRecorderPlugin`).
- Observe automatic token usage recording from `genie.llm` calls.
- Retrieve and interpret token usage summaries using `genie.usage.get_summary()`.

```python
# Full code from examples/E20_token_usage_example.py
```

**Key Takeaways:**
- Token usage is automatically tracked for supported LLM provider operations.
- `genie.usage.get_summary()` provides insights into token consumption.
