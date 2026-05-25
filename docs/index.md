# Genie Tooling

**A sovereign, plugin-based Python middleware for building auditable AI
agents and LLM applications.**

Genie Tooling is async-first, dependency-light, and designed for use
inside *corporate harnesses* — environments where every decision the
system makes must be reconstructable, every tool invocation traceable,
and every policy change reviewable in version control. It is not a
LangChain / LangGraph wrapper; it does not import third-party agent
frameworks.

## Why this exists

Most LLM-powered systems answer "why did the system do that?" with
"the model decided." Genie Tooling reverses the default: deterministic
policy, structured audit records, hot-swappable plugins so that the
parts that *aren't* deterministic (the model) are tightly bounded by
the parts that are.

What that buys you:

*   **λ-CQS context scoping** — YAML-driven rule engine that decides
    both how a query is routed and how the response is shaped (tone,
    verbosity, redactions, persona). Same query, two profiles, two
    different responses, two different audit records. See
    [Context Scoping](guides/context_scoping.md).
*   **`DecisionRecord` audit schema** — one structured record per
    `resolve_and_formulate` call joining query, user identity, inferred
    context, ranked rules, aggregated constraints, derivation result,
    formulation prompt and final output, with per-stage timings.
    Emitted as an `audit.decision_record` trace event and available
    in-process via `genie.context.last_decision`.
*   **Production-grade HITL ladder** — CLI / dev-auto (with loud
    production warnings) / webhook (Slack/Teams/JIRA) / YAML-policy
    auto-approve. Per-action approval gate on `ReActAgent` too.
*   **Tool execution provenance** — every `genie.execute_tool` emits a
    trace event with the full `caller_chain` so audit can reconstruct
    which rule, in which agent, in which Genie call, invoked a given
    tool.

## What's new in 0.2.0

*   **Anthropic LLM provider plugin** with native `tool_use`
    round-trips, streaming, vision, and forced-tool-use structured
    outputs.
*   **Native structured outputs** via `response_schema=YourPydanticModel`
    on `genie.llm.chat(...)` / `genie.llm.generate(...)`. OpenAI uses
    `json_schema` strict mode, Anthropic uses forced tool-use, Gemini
    uses native `response_schema`. Ollama/llama.cpp fall through to
    client-side parsing.
*   **Vision in `ChatMessage`** — `content` can now be a list of
    `TextContentBlock` / `ImageContentBlock`. Image blocks route through
    each provider's native vision API.
*   **MCP client** (`mcp_client_tool_v1`) — connect to any MCP server
    over stdio, expose its tools as Genie tools transparently.
*   **MCP server bootstrap** (`mcp_server_bootstrap_v1`) — expose Genie
    tools over MCP for Claude Desktop / IDE plugins / other agents.
*   **Provider-native tool-use loops** in `ReActAgent` via
    `agent_config["use_native_tool_use"] = True`, with parallel tool
    calls in a single turn.

See [Changelog](changelog.md) for the full Phase 3 / 4 / 5 history.

## Core ideas

*   **`Genie` facade** — `genie.llm`, `genie.rag`, `genie.tools`,
    `genie.context`, `genie.observability`, `genie.human_in_loop`,
    `genie.run_command(...)`, `genie.execute_tool(...)`.
*   **Plugins everywhere** — LLM providers, command processors, tools,
    RAG components, guardrails, HITL approvers, distributed task
    queues, λ-CQS context-engine components. Registered via Poetry
    entry points; loaded by ID at startup.
*   **Explicit tool enablement** — tools are only active if listed in
    `tool_configurations`. `auto_enable_registered_tools=True` is
    convenient for development; `False` is the recommended production
    default.
*   **`MiddlewareConfig.environment`** — set to `"production"` to make
    the framework refuse to silently run with the dev auto-approve
    HITL plugin.
*   **`@tool` decorator** — turn an async Python function into a Genie
    tool with auto-generated metadata and JSON schema.
*   **Zero-effort observability** — enable a tracer
    (`observability_tracer="console_tracer"` or `"otel_tracer"`) and
    every facade call, tool execution, LLM round-trip, guardrail
    decision and audit `DecisionRecord` flows through it.

## Quick start

```python
import asyncio
import logging
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings
from genie_tooling.genie import Genie

async def main():
    logging.basicConfig(level=logging.INFO)

    cfg = MiddlewareConfig(
        environment="development",
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name="qwen3.6:35b",
            command_processor="llm_assisted",
            tool_lookup="hybrid",
            observability_tracer="console_tracer",
        ),
        tool_configurations={"calculator_tool": {}},
    )
    genie = await Genie.create(config=cfg)
    try:
        chat = await genie.llm.chat([{"role": "user", "content": "Tell me about Genie Tooling in one sentence."}])
        print(chat["message"]["content"])

        cmd = await genie.run_command("What is 5 times 12?")
        print(cmd.get("tool_result"))
    finally:
        await genie.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Dive deeper

*   **User Guide**
    *   [Installation](guides/installation.md)
    *   [Configuration](guides/configuration.md) /
        [Simplified Configuration](guides/simplified_configuration.md)
    *   [Using LLM Providers](guides/using_llm_providers.md) — Anthropic,
        structured outputs, vision
    *   [Context Scoping (λ-CQS)](guides/context_scoping.md) — the
        audit substrate
    *   [Using Tools](guides/using_tools.md)
    *   [Using RAG](guides/using_rag.md)
    *   [Using Command Processors](guides/using_command_processors.md)
    *   [Tool Lookup](guides/tool_lookup.md)
    *   [Prompt Management](guides/using_prompts.md)
    *   [Conversation State](guides/using_conversation_state.md)
    *   [Observability & Tracing](guides/observability_tracing.md) —
        `@traceable`, OTel, `DecisionRecord`
    *   [Human-in-the-Loop (HITL)](guides/using_human_in_loop.md) —
        CLI / dev / webhook / policy
    *   [Token Usage Tracking](guides/token_usage_tracking.md)
    *   [Guardrails](guides/using_guardrails.md)
    *   [Distributed Tasks](guides/distributed_tasks.md)
    *   [Logging](guides/logging.md)
*   **Developer Guide**
    *   [Plugin Architecture](guides/plugin_architecture.md)
    *   [Extending Genie Tooling](guides/extending_genie_tooling.md)
    *   [Creating Plugins (Overview)](guides/creating_plugins.md)
    *   [Creating Tool Plugins](guides/creating_tool_plugins.md)
    *   [Creating RAG Plugins](guides/creating_rag_plugins.md)
    *   [Creating Other Plugins](guides/creating_other_plugins.md)
*   **API Reference** — see the [API section](api/index.md) for the
    `Genie` facade, decorators, core types, `MiddlewareConfig`, and
    `FeatureSettings`.
*   **Tutorials & Examples** — start with E01–E04 for chat, E05 for
    RAG, E06–E07 for command processing, E08 for the `@tool`
    decorator, E11 for an advanced showcase.
