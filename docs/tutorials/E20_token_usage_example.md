# Tutorial: Token Usage Tracking (E20)

This tutorial corresponds to the example file `examples/E20_token_usage_example.py`.

It demonstrates how to monitor LLM token consumption for cost and performance analysis. It shows how to:
- Configure a `TokenUsageRecorderPlugin` (e.g., `in_memory_token_recorder`).
- Observe how token usage is automatically recorded after `genie.llm` calls.
- Retrieve and interpret a usage summary using `genie.usage.get_summary()`.

## Example Code

--8<-- "examples/E20_token_usage_example.py"
