# E29 — Anthropic Native Tool Use

Source: [`examples/E29_anthropic_native_tool_use.py`](https://github.com/genie-tooling/genie-tooling/blob/main/examples/E29_anthropic_native_tool_use.py)

Demonstrates `ReActAgent` driving Claude via provider-native `tool_use`
round-trips, with `@tool` decorator side-effect metadata feeding the
Claude-Code permission model.

## Highlights

* `@tool(side_effects="read", idempotent=True, cacheable=True)` on the
  weather lookup — the permission plugin auto-allows it.
* `@tool(side_effects="none")` on the C→F converter — pure compute.
* `agent_config={"use_native_tool_use": True}` switches the agent to
  the native `tool_calls` path instead of regex parsing.
* `input_context={"attribution_tags": {"demo": "E29"}}` flows through
  every LLM and tool call for cost attribution.

## Required setup

```bash
poetry install --extras anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
poetry run python examples/E29_anthropic_native_tool_use.py
```

The agent will call `get_current_weather("London")` and
`celsius_to_fahrenheit(12.0)` and report the final answer. Each LLM
round-trip uses Anthropic's native tool_use blocks.
