# Genie Tooling

[![Pytest Status](https://github.com/genie-tooling/genie-tooling/actions/workflows/python_ci.yml/badge.svg)](https://github.com/genie-tooling/genie-tooling/actions/workflows/python_ci.yml)

**A sovereign, plugin-based Python middleware for building auditable AI
agents and LLM applications.**

`genie-tooling` is async-first, dependency-light, and designed for use
inside *corporate harnesses* — environments where every decision the
system makes must be reconstructable, every tool invocation traceable,
and every policy change reviewable in version control. It is not a
LangChain / LangGraph wrapper. It does not import third-party agent
frameworks. The plugin protocols, the agent loops, the audit record,
the policy engine — all first-party.

## Why this exists

Generic LLM-powered tools answer most "why did the system do that?"
questions with "the model decided." That isn't an answer audit teams or
regulators can act on. `genie-tooling` is built around the opposite
default: deterministic policy, structured audit records, and
hot-swappable plugins so that the parts that *aren't* deterministic
(the model itself) are tightly bounded by the parts that are.

What that buys you:

- **λ-CQS context scoping** — YAML-driven rule engine that decides how
  a query is routed AND how the response is shaped (tone, verbosity,
  redactions, persona). Same query, two profiles, two different
  responses, two different `DecisionRecord` audit entries.
  See [docs/guides/context_scoping.md](docs/guides/context_scoping.md).
- **`DecisionRecord` audit schema** — one structured record per query
  joining the user identity, inferred context, ranked rules, aggregated
  constraints, derivation result, formulation prompt and final output,
  with per-stage timings. Emitted as an `audit.decision_record` trace
  event AND available in-process at `genie.context.last_decision`.
- **Production-grade HITL ladder** — CLI, dev-auto (with loud production
  warnings), webhook (Slack/Teams/JIRA), YAML-policy auto-approve.
  Per-action approval gate on `ReActAgent` for "every database write
  needs review" workflows.
- **Tool execution provenance** — every `genie.execute_tool` emits a
  trace event with the full `caller_chain` so you can reconstruct which
  rule, in which agent, in which Genie call, invoked a given tool.

## What's new in 0.2.0 (2026-05-25)

- **Anthropic LLM provider plugin** with native `tool_use` round-trips,
  streaming, vision, and forced-tool-use structured outputs.
- **Native structured outputs**: `response_schema=YourPydanticModel`
  works against OpenAI (`json_schema` strict mode), Anthropic, and
  Gemini. Ollama/llama.cpp fall through to client-side parsing.
- **Vision in `ChatMessage`**: `content` can now be a list of
  `TextContentBlock` / `ImageContentBlock`. Image blocks route through
  each provider's native vision API.
- **MCP client** (`mcp_client_tool_v1`): connect to any MCP server over
  stdio, expose its tools as Genie tools transparently.
- **MCP server bootstrap** (`mcp_server_bootstrap_v1`): expose Genie
  tools over MCP for Claude Desktop / IDE plugins / other agents.
- **Provider-native tool-use loops** in `ReActAgent` via
  `agent_config["use_native_tool_use"] = True`, with parallel tool
  calls in a single turn.

See [CHANGELOG.md](CHANGELOG.md) for the full Phase 3/4/5 history.

## Core concepts

- **`Genie` facade**: the primary entry point — `genie.llm`,
  `genie.rag`, `genie.tools`, `genie.context`, `genie.observability`,
  `genie.human_in_loop`, `genie.run_command(...)`,
  `genie.execute_tool(...)`.
- **Plugins everywhere**: LLM providers, command processors, tools,
  RAG components (loaders/splitters/embedders/stores), caching,
  guardrails, output parsers, HITL approvers, distributed task queues,
  context engine components. Registered via Poetry entry points; loaded
  by ID at startup.
- **Explicit tool enablement**: tools are only active if listed in
  `tool_configurations`. `auto_enable_registered_tools=True` is
  convenient for development; `False` is the recommended production
  default — it gives you a single explicit manifest of what the agent
  can do.
- **`MiddlewareConfig.environment`**: set to `"production"` to make
  the framework refuse to silently run with the dev auto-approve HITL
  plugin.
- **`@tool` decorator**: turn an async Python function into a Genie
  tool. Metadata, JSON schema, and entry point are derived from the
  signature and docstring.
- **Zero-effort observability**: enable a tracer (e.g.
  `observability_tracer="console_tracer"` or `"otel_tracer"`) and
  every facade call, tool execution, LLM round-trip, guardrail decision
  and audit `DecisionRecord` flows through it.

## Key plugin categories

- **LLM providers**: Anthropic (Claude), OpenAI, Ollama, Gemini,
  Llama.cpp (server + internal).
- **Command processors**: `llm_assisted`, `rewoo`, `simple_keyword`.
- **Agents** (bundled): `ReActAgent`, `PlanAndExecuteAgent`,
  `DeepResearchAgent`.
- **Tools**: calculator, sandboxed FS, code execution (sandboxed +
  Docker), web scraper, web search, ArXiv, PDF extractor, content
  retriever, OpenWeatherMap, MCP remote tools, and more.
- **HITL approvers**: CLI, dev-auto, webhook, policy-driven YAML.
- **Context engine** (λ-CQS): context sources, predicate extractors,
  rule engines (deterministic filesystem + opt-in vector-DB), derivation
  strategies, formulation strategies, constraint translator.
- **RAG**: file/web loaders, character-recursive splitter,
  sentence-transformer / OpenAI embedders, FAISS / Chroma / Qdrant
  stores, similarity / hybrid retrievers.
- **Guardrails**: keyword blocklist, schema-aware redactor, custom.
- **Observability**: `ConsoleTracerPlugin`, `OpenTelemetryTracerPlugin`,
  `DefaultLogAdapter`, `PyviderTelemetryLogAdapter`, in-memory + OTel
  token usage recorders.
- **Task queues**: Celery, Redis Queue (RQ).

Run `python -c "import importlib.metadata as md; [print(ep.name) for ep in md.entry_points().select(group='genie_tooling.plugins')]"` for the complete current list, or see `pyproject.toml`.

## Installation

```bash
git clone https://github.com/genie-tooling/genie-tooling.git
cd genie-tooling
poetry install --all-extras
```

Optional extras (install only what you need):

| Extra | Adds |
|---|---|
| `anthropic` | Anthropic Claude provider |
| `mcp` | MCP client + server |
| `openai_services` | OpenAI provider + embeddings |
| `llama_cpp_internal` | In-process llama.cpp |
| `local_rag` | sentence-transformers + FAISS |
| `chromadb` | Chroma vector store |
| `qdrant` | Qdrant vector store |
| `research_tools` | arxiv, pypdf, sympy |
| `distributed_tasks` | Celery + RQ |
| `observability` | OpenTelemetry exporters |
| `full` | All of the above |

## Quick start — local llama.cpp + RAG + code execution

```python
import asyncio
import logging
from pathlib import Path

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


async def main():
    logging.basicConfig(level=logging.INFO)

    gguf_path = Path("/path/to/your/model.gguf")  # <-- set me
    if not gguf_path.exists():
        raise SystemExit(f"GGUF model not found at {gguf_path}")

    cfg = MiddlewareConfig(
        environment="development",  # set to "production" in prod
        auto_enable_registered_tools=True,
        features=FeatureSettings(
            llm="llama_cpp_internal",
            llm_llama_cpp_internal_model_path=str(gguf_path.resolve()),
            llm_llama_cpp_internal_n_gpu_layers=-1,
            llm_llama_cpp_internal_n_ctx=4096,
            command_processor="llm_assisted",
            tool_lookup="embedding",
            rag_embedder="sentence_transformer",
            rag_vector_store="faiss",
            observability_tracer="console_tracer",
            logging_adapter="pyvider_log_adapter",
        ),
        tool_configurations={
            "calculator_tool": {},
            "sandboxed_fs_tool_v1": {"sandbox_base_path": "./sandbox"},
            "generic_code_execution_tool": {},
        },
    )

    genie = await Genie.create(config=cfg)
    try:
        # Chat
        r = await genie.llm.chat(
            [{"role": "user", "content": "One sentence: what is genie-tooling?"}]
        )
        print(r["message"]["content"])

        # Command → tool selection + execution
        cmd = await genie.run_command(
            "Run Python: print(f'7 * 8 = {7*8}')"
        )
        print(cmd.get("tool_result", {}).get("stdout", "").strip())
    finally:
        await genie.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## Quick start — Anthropic with native tool-use and structured output

```python
import asyncio
from pydantic import BaseModel

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


class WeatherReport(BaseModel):
    city: str
    temperature_c: float
    condition: str


async def main():
    cfg = MiddlewareConfig(
        environment="production",
        features=FeatureSettings(
            llm="anthropic",
            llm_anthropic_model_name="claude-sonnet-4-6",
        ),
    )
    genie = await Genie.create(config=cfg)
    try:
        r = await genie.llm.chat(
            messages=[
                {"role": "user", "content": "London: 12C, light rain. Report it."}
            ],
            response_schema=WeatherReport,   # native structured output
        )
        report = WeatherReport.model_validate_json(r["message"]["content"])
        print(report)
    finally:
        await genie.close()


asyncio.run(main())
```

Requires `ANTHROPIC_API_KEY` in the environment (or supply a custom
`KeyProvider` to `Genie.create`).

## Quick start — λ-CQS with audit record

Same query, two profiles, two routings, two audit records. Full walk-through in
[docs/guides/context_scoping.md](docs/guides/context_scoping.md).

```python
cfg = MiddlewareConfig(
    features=FeatureSettings(
        llm="ollama",
        llm_ollama_model_name="qwen3.6:35b",
    ),
    extension_configurations={
        "context_engine": {
            "context_source_config": {
                "default_profile": {"expertise": "expert", "state": "Stressed"}
            },
        },
    },
)
genie = await Genie.create(config=cfg)
response = await genie.context.resolve_and_formulate(
    query="explain why this calculation is failing",
    session_id="s1",
    user_identity={"sub": "user-1", "role": "engineer"},
)
record = genie.context.last_decision
# record.winning_rule_id, record.c_f, record.formulation_constraints_text,
# record.stage_timings_ms, etc.
```

## Documentation

- [Installation](docs/guides/installation.md)
- [Configuration](docs/guides/configuration.md) /
  [Simplified configuration](docs/guides/simplified_configuration.md)
- [LLM providers](docs/guides/using_llm_providers.md) — including
  Anthropic, structured outputs, vision
- [Context scoping (λ-CQS)](docs/guides/context_scoping.md) — the audit
  substrate
- [Human-in-the-loop](docs/guides/using_human_in_loop.md) — CLI / dev /
  webhook / policy approvers
- [Observability & tracing](docs/guides/observability_tracing.md) —
  `@traceable`, OTel, `DecisionRecord`
- [Tools](docs/guides/using_tools.md), [RAG](docs/guides/using_rag.md),
  [Command processors](docs/guides/using_command_processors.md)
- [Guardrails](docs/guides/using_guardrails.md),
  [Prompts](docs/guides/using_prompts.md),
  [Conversation state](docs/guides/using_conversation_state.md)
- [Plugin architecture](docs/guides/plugin_architecture.md) /
  [Creating plugins](docs/guides/creating_plugins.md)
- [CHANGELOG](CHANGELOG.md)

## Test counts

- 1531 unit tests passing.
- 25/25 live integration tests passing against `qwen3.6:35b` on Ollama
  (λ-CQS rules end-to-end, agent loops, audit-record emission, rule
  reload, HITL flows).
- Anthropic / OpenAI / Gemini covered via mocked unit tests.

## License

Apache 2.0 — see `LICENSE`.
