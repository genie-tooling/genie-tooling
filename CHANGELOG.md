# Changelog

All notable changes to genie-tooling are documented in this file.

## [0.3.4] — 2026-05-25 — Extras cleanup + packaging fix

Discovered during a Docker build that **half the required dependencies
weren't getting installed** for users who selected extras at install
time. Root cause: required deps were also listed in `[tool.poetry.extras]`
groups, which causes Poetry to gate the package on the matching `extra ==`
marker in `poetry.lock`. The result: `pip install genie-tooling[anthropic]`
would silently skip `aiofiles`, `pypdf`, `redis`, `Jinja2`, `numpy`,
`sympy`, `rank-bm25`, `pyvider-telemetry`, `httpx`, `jsonschema`,
`googlesearch-python`, `beautifulsoup4` — all of which the framework
imports at module load time.

### Fixed

- **Packaging**: removed every required dep from `[tool.poetry.extras]`
  lists. Extras now only contain truly optional packages. The lockfile
  regenerates without the bogus `extra == "..."` markers on required
  deps, so a base install (`pip install genie-tooling`) now correctly
  pulls everything the framework actually imports.
- **`hnswlib`**: demoted from required → optional + added new `hnsw` extra.
  It's not imported anywhere in `src/` but the C++ compile requirement
  was breaking slim-image Docker builds. If you actually want HNSW-based
  RAG, install via `pip install genie-tooling[hnsw]`.
- **Several extras emptied** because every dep they listed was already
  required: `ollama`, `slack`, `prompts`, `llama_cpp_server`,
  `pyvider_adapter`, `text_ranking`, `symbolic_math`. Listed for
  back-compat but contain no packages.
- **`task_queues`** trimmed to `["celery", "rq"]` (redis was required).
- **`arxiv_tool`** trimmed to `["arxiv"]` (pypdf was required).

### Dockerfile

- `weekly_async/Dockerfile` rewritten to use `poetry install` with the
  bundled `poetry.lock` instead of `pip install dir[extras]` — the
  latter combination silently drops deps from the wheel metadata when
  the package uses poetry-core as its build backend. Now reproducible
  and pulls every required dep. `git` added to the build apt-install
  step because `pyvider-telemetry` is a git URL dep.
- weekly_async itself installed via `pip install --no-deps` since the
  genie-tooling version isn't on PyPI yet (the version pin can't
  resolve against a local source dir).

### Net delivery

- Image build succeeds end-to-end. `weekly-async --dry-run` against the
  bundled `teams.example.yaml` processes all 3 teams cleanly.
- Image size 806 MB (Python 3.12 + Node 20 + cached MCP packages).

## [0.3.3] — 2026-05-25 — Deep security sweep (31 → 1 advisory)

Second pass after a fresh `pip-audit` surfaced 25 additional advisories
the first sweep missed (mostly out-of-date pypdf, plus transformers and
pytest). Net: from 31 known vulnerabilities → 1, and the remaining one
has no upstream fix.

### Security — bumped

- **pypdf** 4.3.1 → **6.12.1** — closed 23 CVEs (CVE-2025-55197 through
  CVE-2026-41314 covering ReDoS, infinite loops, memory exhaustion,
  malformed-stream parsing bugs). `PdfReader` / `.pages` API used by
  our PDF tool is stable across the 4 → 6 major bump.
- **transformers** 4.57.6 → **5.3.0** — CVE-2026-1839 (RCE in
  `transformers.utils.load_torch_state_dict`, fixed in 5.0.0rc3) +
  PYSEC-2025-217. Now an explicit direct optional dep with `>=5.0.0`
  floor, listed in the `local_rag` + `full` extras.
- **sentence-transformers** ^2.7.0 → **5.5.1** — pulled in transformers
  ≥5; constraint widened to `>=5.0.0,<6.0`. Embedder API
  (`SentenceTransformer(name).encode(...)`) is unchanged.
- **pytest** ^8.2.2 → **^9.0.3** — CVE-2025-71176 (local
  /tmp/pytest-of-{user} privilege escalation / DoS). Required two test
  fixtures to migrate from the removed pytest-asyncio `event_loop` parameter
  to `@pytest_asyncio.fixture` async-generator form
  (`tests/unit/test_genie_facade.py::fully_mocked_genie` and
  `tests/unit/caching/impl/test_in_memory_cache.py::mem_cache`).
- **pytest-asyncio** ^0.23.8 → **^1.3.0**.

### Security — accepted residual risks (documented)

- **requests** stays at `>=2.32.4` (currently 2.32.4 in lock).
  CVE-2026-25645 (`requests.utils.extract_zipped_paths()` insecure
  temp-file reuse) is fixed in 2.33.0, but `arxiv` (optional, via
  `research_tools` extra) caps `requests <2.33`. Genie never calls
  the vulnerable function — accept until arxiv updates upstream.
- **diskcache** 5.6.3 — CVE-2025-69872 (pickle-based cache allows
  arbitrary code execution if an attacker has write access to the
  cache directory). **No upstream fix exists** — this is a fundamental
  design property of `diskcache`. Pulled in only by `llama-cpp-python`
  (optional `llama_cpp_internal` extra). Mitigation documented in
  `pyproject.toml`: run with a private cache dir (700 perms) and only
  enable this extra when needed.

### Build / configuration

- `transformers` added as a direct optional dep + included in `local_rag`
  and `full` extras alongside `sentence-transformers` + `torch`.
- 1654 unit tests still passing on pytest 9.0.3.

## [0.3.2] — 2026-05-25 — Security floors + lockfile refresh

Closes 6 Dependabot advisories on the lockfile. Two transitive deps
needed actual bumps; the rest already resolved by current pins but
now have explicit floors in `pyproject.toml` to prevent regression.

### Security

- **urllib3** 2.4.0 → **2.7.0** — CVE-2025-50181 + CVE-2025-50182
  (redirect-control issues in browsers / Node.js). Floor pinned at
  `>=2.5.0`.
- **torch** 2.7.1 → **2.12.0** — GHSA-887c-mr87-cxwp (Improper
  Resource Shutdown or Release). Floor pinned at `>=2.8.0`; listed in
  the `local_rag` and `full` extras alongside `sentence-transformers`.
- **Explicit floors** for already-resolved CVEs (lockfile audit hygiene):
    - `pillow >= 10.3.0` (CVE-2024-28219, BCn buffer overflow)
    - `requests >= 2.32.4` (CVE-2024-47081, .netrc credential leak)
    - `black >= 24.4.2` already pinned via dev deps (CVE-2024-21503 ReDoS)

### Changed

- **Python upper bound** narrowed from `<4.0` → `<3.14` so torch 2.8+
  resolves cleanly (its transitive `triton` requires `<3.15`). CI
  matrix is unchanged (`3.11` + `3.12`).
- Tests: 1654 unit + 32 weekly_async still passing after the
  dependency refresh.

## [0.3.1] — 2026-05-25 — Phase 6 wiring-fix release

Post-Phase-6 audit found that several plugins shipped without their
end-to-end wiring. This release closes the 13 gaps and lifts the test
count from 1634 → 1654 (+20 new tests).

### Fixed (wiring)

- **F1 — FeatureSettings shortcuts** for the Phase 6 plugins:
  `hitl_ledger`, `agent_checkpointer`, `progress_sinks`. New aliases in
  the resolver for ledger / checkpointer / sinks / MCP composition. New
  `MiddlewareConfig` fields: `default_agent_checkpointer_id` (+ configs),
  `default_progress_sink_ids` (+ configs).
- **F2 — MCP composition wired into `Genie.create()`.** Reads
  `extension_configurations["mcp_composition"]`, instantiates the
  `MCPCompositionPlugin`, and registers its discovered tools via the new
  `ToolManager.register_tool_instance()` method. Survives shutdown
  cleanly. Previously the plugin existed but was never instantiated.
- **F3 — ReActAgent threads attribution + budget kwargs** to LLM calls
  and `genie.execute_tool` calls. Both regex and native loops. Helper
  methods `_llm_attribution_kwargs` and `_tool_context` keep it DRY.
  Previously agent-emitted LLM calls were unattributed and unbudgeted.
- **F4 — Agent checkpointer wired into ReActAgent.** Saves state every
  iteration boundary and on terminal exit. New `resume_from_run_id`
  kwarg on `agent.run(...)` loads prior scratchpad and continues from
  the saved iteration. Native loop supports resume too.
- **F5 — Replay harness wired into `LLMInterface`.** `replay_recorder`
  and `replay_player` kwargs + `set_replay_recorder()` /
  `set_replay_player()` for runtime swap. When a player is set, every
  `chat`/`generate` serves from the recording without touching the
  provider.
- **F6 — Progress sinks load from `MiddlewareConfig`.** `Genie.create()`
  auto-instantiates plugins listed in `default_progress_sink_ids` and
  hands them to every agent run. Caller-supplied sinks still stack on top.
- **F7 — Progress streaming in `PlanAndExecuteAgent`.** Mirrors the
  ReActAgent pattern.
- **F8 — README + docs/index updated for 0.3.0**.
- **F9 — `PHASE_6_PLAN.md` §9 acceptance table** rewritten to reflect
  what's actually wired.
- **F10 — Tutorial walkthroughs** for E29–E33 filled out (were stubs).
- **F11 — Five new feature guides**: `permissions.md`, `budget.md`,
  `checkpointing.md`, `mcp_composition.md`, `progress.md`.
- **F12 — `slack` extra** added to `pyproject.toml`.
- **F13 — 10 CLI smoke tests** for `genie-lint` + `genie-mcp-serve`.

### Net delivery

- 1654 unit tests passing (+20)
- All Phase 6 features now usable through `FeatureSettings` — no more
  raw plugin-ID plumbing required for the Weekly Async use case.
- New `WEEKLY_ASYNC_AGENT.md` deployment runbook at the repo root.

## [0.3.0] — 2026-05-25

The **corporate agentic harness** release. Phase 6 turns the audit-aware
framework from 0.2.0 into a deployable substrate for SRE on-call and
dev-team automation. Net: 1634+ unit tests passing, 98 registered
plugins, MCP composition layer, durable checkpointer + approval ledger,
hard budget enforcement, Claude-Code-style permission model.

### Added — Phase 6A (safety primitives)

- **6A.1 — Tool side-effect metadata.** `Tool.get_metadata` now documents
  `side_effects` (`none`/`read`/`write`/`destructive`/`unknown`),
  `requires_approval`, and `idempotent`. The `@tool` decorator supports
  both bare and parameterized forms — `@tool(side_effects="destructive",
  requires_approval=True)`.
- **6A.2 — Durable agent checkpointer** (`AgentCheckpointerPlugin`).
  Two implementations bundled: `in_memory_agent_checkpointer_v1` for
  tests, `sqlite_agent_checkpointer_v1` using stdlib `sqlite3` over
  `asyncio.to_thread` for single-host production. Postgres impl is the
  intended production target (designed; not bundled).
- **6A.3 — Budget enforcement.** New `BudgetEnforcerPlugin` protocol +
  `in_memory_budget_enforcer_v1`. Hard caps per scope on tokens, USD
  cost, tool-calls, LLM-calls, wall-clock. `genie.llm.chat` and
  `genie.execute_tool` accept a `budget_scope=` kwarg and raise
  `BudgetExceeded` past the cap. `genie.budget` exposes the public API.
- **6A.4 — Pre-flight policy on tool params.** `genie.run_command`,
  `ReActAgent` (both regex and native loops), and `PlanAndExecuteAgent`
  now populate `data_to_approve["tool_metadata"]` + `context.user_identity`
  / `session_id` so the policy approver can match on side-effects /
  param patterns / identity, not just `tool_id`.
- **6A.5 — Approval routing.** `WebhookApprovalPlugin` supports
  per-request routing via a `routes:` list: destructive → PagerDuty,
  code changes → GitHub reviewers, default → general endpoint. Match
  keys mirror the permission plugin (`tool_id`, `tool_id_in`,
  `side_effects_in`, `params_match`, `user_identity`).
- **6A.5b — Claude-Code-style permission model** (`claude_code_permissions_v1`).
  Three-tier `allow` / `ask` / `deny` with glob match on tool ID AND
  parameters (jmespath-style), side-effects-driven defaults, per-session
  always-allow overrides, and `ask_human` status that delegates to the
  next approver in the HITLManager chain. `HITLManager` gains chain
  support via `default_approver_chain`.
- **6A.6 — Cost attribution tags.** `genie.llm.chat`/`generate` accept
  `attribution_tags`, `session_id`, `user_id` framework kwargs that are
  *popped* before forwarding to the provider, land in the token-usage
  record, and flow into trace events for SIEM-side correlation.
- **6A.7 — Durable approval ledger** (`HITLLedgerPlugin`). Two impls:
  `in_memory_hitl_ledger_v1`, `sqlite_hitl_ledger_v1`. Every HITL
  decision auto-persists with `decision_id` / `correlation_id` joins to
  the parent `DecisionRecord`. Queryable by tool_id, status, attribution
  tag, time window.

### Added — Phase 6B (corporate tool integrations)

- **6B.1 — MCP composition layer** (`mcp_composition_v1`). Multi-server
  ingest with side-effect overlay registry, response-redaction hook,
  re-export gateway pattern (Genie as a *policy-controlled corporate
  MCP gateway*). `MCPRemoteTool` honors overlay metadata so policy
  decisions apply to MCP-ingested tools just like native ones.
- **6B.2 — Curated overlay catalog.** YAML overlays for 11 popular MCP
  servers: Slack, GitHub, Notion, Linear, JIRA, AWS-API, Filesystem,
  Postgres, Sentry, Datadog, Grafana, Prometheus, Google Drive, Gmail.
  Each pins side-effects per tool and flags destructive operations as
  requires_approval.
- **6B.3.1 — Native Slack tool** (`slack_post_message_v1`,
  `slack_add_reaction_v1`, `slack_get_user_profile_v1`,
  `slack_list_channels_v1`, `slack_get_channel_history_v1`,
  `slack_thread_progress_sink_v1`). Native rather than MCP so threaded
  progress streaming works cleanly. Side-effects metadata declared;
  writes return an `audit_artifact` capturing the exact API request.

### Added — Phase 6C (operational maturity)

- **6C.2 — Streaming progress updates.** `ProgressSinkPlugin` protocol;
  `console_progress_sink_v1` + `webhook_progress_sink_v1` + the Slack
  thread sink above. `ReActAgent` emits progress at iteration boundaries
  and tool calls; sinks fan out fire-and-forget so a slow sink doesn't
  block the agent.
- **6C.6 — Audit attestation field.** Tools may return an
  `audit_artifact` field (raw command line / HTTP body / payload) that's
  captured into `genie.execute_tool.success` trace events for forensic
  reconstruction.
- **6C.7 — KeyProvider scoping.** `KeyProvider.get_key(name, scope=...)`
  with `tenant` / `team` / `env` scoping. `EnvironmentKeyProvider`
  resolves scoped keys via uppercase-prefixed env vars.
- **6C.8 — KeyProvider hot reload.** `KeyProvider.refresh()` for
  long-running services rotating credentials without restart.
- **6C.10 — MCP server binary.** `python -m genie_tooling.mcp_server
  --config foo.yml` and `genie-mcp-serve` console-script. Bootstraps a
  Genie instance and exposes registered tools over MCP stdio for Claude
  Desktop / IDE plugins.
- **6C.11 — Policy-as-code linting CLI.** `genie-lint` console-script
  runs the runtime `RuleValidationError` checks in CI before merging
  YAML changes. Lints both λ-CQS rules and MCP overlays.

### Added — Phase 6D (orchestration & eval)

- **6D.2 — Strict replay harness.** `ReplayRecorder` + `ReplayPlayer` +
  `ReplayMiss`. Byte-for-byte recording of LLM + tool boundary calls
  into a JSON fixture; replay substitutes recorded responses for live
  calls. Framework-level kwargs (`attribution_tags`, etc.) are stripped
  from the hash so the same LLM request collides on the same hash
  regardless of caller metadata. `assert_exhausted()` catches regressions
  where the replay run does fewer calls than the recording.
- **6D.4 — Conversation forking.** `genie.conversation.fork(session_id)`
  for parallel investigation branches. Deep-copies state and stamps a
  new session_id.

### Added — Phase 6E (residuals)

- **6E.1 — Example scripts E29-E33** covering Anthropic native tool-use,
  MCP composition + overlays, budget + audit ledger, replay harness, and
  the Weekly Async planner skeleton.
- Bundled overlay catalog and MCP server README added to package data.

### Test counts

- 1634+ unit tests passing (+103 in Phase 6).
- 98 registered `genie_tooling.plugins` entry points (was 83 → +15).

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
