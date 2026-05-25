# Changelog

All notable changes to genie-tooling are documented in this file.

## [0.2.0] — 2026-05-25

The "corporate-harness + 2026 modernization" release. Phase 3 hardened the
audit story; Phase 4 wired the deterministic policy layer through to actual
LLM behavior; Phase 5 closed the 2026 provider/protocol gaps. Net: 1531+
unit tests passing, full live coverage against Ollama (gemma3:4b, qwen3.6:35b)
and unit-test coverage against mocked OpenAI / Anthropic / Gemini.

### Added — Phase 5 (modernization)

- **M1 — Anthropic LLM provider plugin** (`anthropic_llm_provider_v1`). Full
  `LLMProviderPlugin` implementation against the Anthropic Messages API
  (Claude). System-message extraction, tool_use round-trips, streaming,
  structured outputs via forced tool-use rounds, ChatMessage ↔ Anthropic
  content-block translation. Optional `anthropic` extra.
- **M4 — Native structured outputs on `LLMProviderPlugin`**. New canonical
  ``response_schema: type[BaseModel]`` kwarg on `chat()` / `completion()`.
  OpenAI uses `response_format={"type": "json_schema", strict: true, ...}`
  with automatic Pydantic-schema strict-mode patching; Anthropic uses the
  forced tool-use round-trip; Gemini uses native `response_schema`. Ollama
  / llama.cpp providers ignore the kwarg — callers fall back to
  `PydanticOutputParserPlugin` client-side validation.
- **M5 — Vision in `ChatMessage`**. `ChatMessage.content` is now
  `Union[str, List[ContentBlock]]`. New `TextContentBlock`,
  `ImageContentBlock` types. OpenAI, Anthropic providers route image
  blocks through their native vision APIs; Ollama provider collapses
  to text with a placeholder so text-only models still receive a
  well-formed prompt.
- **M2 — MCP client integration** (`mcp_client_tool_v1`). Connects to an
  MCP server (stdio transport), discovers its tools, and produces one
  `MCPRemoteTool` per remote tool. Agents see them as normal Genie tools.
  Optional `mcp` extra.
- **M3 — MCP server bootstrap** (`mcp_server_bootstrap_v1`). Exposes every
  registered Genie tool over MCP so external clients (Claude Desktop,
  IDE plugins, other agents) can call them. Configurable via
  `extension_configurations["mcp_server"]`.
- **M7 — Provider-native tool-use loops in `ReActAgent`**. New
  `agent_config["use_native_tool_use"] = True` opt-in that drives the
  loop via provider-native `tool_calls` instead of regex-parsing free
  text. Honors the `hitl_per_action` gate from B3 on the native path too.
- **M8 — Parallel tool calls in the native loop**. A single assistant
  turn may emit multiple `tool_calls`; the loop executes each (in order)
  and sends each result back keyed to its `tool_call_id`. Partial failure
  in a batch doesn't short-circuit the rest.

### Added — Phase 4 (corporate-harness audit hardening)

- **A1 — `context/constraints.py`**: deterministic constraint translator.
  Converts a C_F dict (`{tone: "formal", verbosity: "low", ...}`) into a
  natural-language `Response guidelines:` block prepended to every
  formulation prompt. C_F constraints now demonstrably reach the LLM.
- **A2 — Rule schema validation at load time**. `FileSystemRuleEnginePlugin`
  cross-checks every action against the registered plugin set. Broken
  rules fail loud with `RuleValidationError` at startup, not silently
  at first match.
- **A3 — `DecisionRecord` audit schema** + ContextManager assembly.
  Every `resolve_and_formulate` call emits one structured
  `audit.decision_record` trace event joining query, user identity,
  inferred context, ranked rules, aggregated C_D/C_F, derivation result,
  formulation, per-stage timings. Available in-process via
  `genie.context.last_decision`.
- **A4 — Bundled rules fixed end-to-end** and a missing
  `default_formulation_prompt.prompt` template added.
- **B1 — `WebhookApprovalPlugin`** (`webhook_approval_v1`): POSTs each
  approval request to a user-supplied URL. Safe-by-default deny on
  timeout/error.
- **B2 — `PolicyAutoApproveHITLPlugin`** (`policy_auto_approve_hitl_v1`):
  YAML-policy-driven auto-approver with glob and role-gated rules. Every
  decision logged with the matching policy ID.
- **B3 — Per-action HITL gate on `ReActAgent`** via
  `agent_config["hitl_per_action"]`. Honored on both the regex and
  native loops.
- **B4 — Renamed `AutoApproveHITLPlugin` → `dev_auto_approve_hitl_v1`**.
  Backward-compat alias kept one cycle with deprecation warning. New
  `MiddlewareConfig.environment` field triggers a loud error log when
  the dev approver is used in a production-tagged environment.
- **C1 — Tool execution provenance tracing**. `genie.execute_tool` trace
  events now carry `caller_chain` + `parent_correlation_id` so audit can
  reconstruct who invoked a tool through which path.
- **C2 — Rule reload at runtime** via `genie.context.reload_rules()`.
  Atomic rollback on validation failure.
- **C3 — Structured `guardrail.decision` trace events**. Every block/warn
  decision includes guardrail_id, reason, actor, and a truncated trigger
  preview joinable via `decision_id`.
- **D1–D4 — Coverage gaps closed**: all rule condition operators
  (`==`, `!=`, `>`, `<`, `>=`, `<=`, `in`, `contains`), all
  `_aggregate_constraints` ops (`set`, `default`, `add`), heuristic
  predicate extractor keyword table, `VectorDBRuleEnginePlugin`
  deterministic mode + audit caveat.

### Added — Phase 3 (post-consolidation bug fixes)

- `ToolsInterface` + `PluginsInterface` on the `Genie` facade — agents and
  command processors no longer reach into `genie._tool_manager` /
  `genie._plugin_manager`.
- `RuleEnginePlugin` protocol unified to take `genie: Genie` on both
  `load_rules` and `evaluate` (eliminates runtime `inspect.signature`
  branching that paper'd over a protocol drift).
- `genie_tooling.context` (formerly the `cqs-engine` satellite repo)
  folded into the core package.
- `FeatureSettings.llm_ollama_base_url`: configure remote Ollama via the
  high-level shortcut without dropping to `llm_provider_configurations`.

### Fixed

- `_evaluate_condition` `in` operator semantics were reversed
  (was `expected in actual`, now `actual in expected`). The
  `["AudienceProfile.state", "in", ["Stressed", "Agitated"]]` rule
  condition shape now behaves as written.
- `PluginManager.get_all_plugin_instances_by_type` no longer
  pre-instantiates every discovered plugin with an empty config to
  filter by type — that was poisoning the instance cache and silently
  ignoring real per-call configs (e.g. `rules_path` on
  `FileSystemRuleEnginePlugin`).
- Bundled agents (`ReActAgent`, `PlanAndExecuteAgent`) had no shipped
  system-prompt templates. Added `DEFAULT_REACT_SYSTEM_PROMPT` and
  `DEFAULT_PLANNER_SYSTEM_PROMPT` module constants with proper Jinja2
  rendering and `{% autoescape false %}` so quotes in tool definitions
  reach the LLM unmangled.
- `PlanAndExecuteAgent` had a mandatory HITL gate with no auto-approve
  default. Out of the box every step failed. Fixed by shipping
  `DevAutoApproveHITLPlugin` and wiring it as a feature alias.

### Deprecated

- `auto_approve_hitl_v1` plugin id — use `dev_auto_approve_hitl_v1`.
  The old id continues to work with a warning for one cycle.

### Test counts

- 1531+ unit tests passing.
- 25/25 live integration tests passing against qwen3.6:35b on Ollama
  (cqs bundled rules end-to-end, agent loops, audit-record emission,
  rule reload, HITL flows).

### Known gaps

- Streaming-of-tool-calls (chunked deltas of `tool_calls`) is not yet
  implemented end-to-end across providers. Non-stream parallel tool
  calls work; streaming chunks currently surface text deltas only.
- Computer-use / browser-use providers are not yet integrated.

## [0.1.0] — 2026-05-24

Initial consolidated release after merging the three sibling repos
(`genie-tooling`, `genie-tooling-agents`, `cqs-engine`) into one package.
See AUDIT.md for the post-merge audit and follow-up fixes that became
Phase 3.
