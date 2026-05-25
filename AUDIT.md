# Post-Consolidation Audit

Date: 2026-05-24
Scope: the consolidated `genie-tooling` repo after folding in `cqs-engine` as `genie_tooling.context` and confirming `genie-tooling-agents` was a near-duplicate snapshot of code already in core. No code edits in this report — findings only, with recommended fixes.

## TL;DR

Phase 1 consolidation is structurally clean. The cqs source moved into `genie_tooling.context.*` with imports rewritten, plugin entry points registered, the bootstrap entry attached, and the `extension_configurations` key renamed from `cqs_engine` → `context_engine`. **77 plugin entry points + 1 bootstrap plugin discoverable**. Smoke imports for all cqs modules pass.

`pytest tests/ --ignore=tests/integration`: **1320 passed, 3 failed, 16 errored**.

- The **3 failures** are all in `tests/unit/context/` and reflect scaffold-era assumptions that broke when the tests started seeing the real `Genie.create()` API instead of a stub. These are pre-existing test bugs that the satellite layout hid; the merge surfaced them.
- The **16 errors** are pytest-asyncio v1 incompatibility (`fixture 'event_loop' not found`). uv installed pytest-asyncio 1.3.0 against a pyproject pin of `^0.23.8`. Environmental, not consolidation-related.

`ruff check src/genie_tooling/`: 120 errors (81 auto-fixable). 59 errors in `context/` alone. Mostly cosmetic (trailing whitespace, missing newlines, unsorted imports, multi-statement lines).

Nothing in this audit is a blocker for `from genie_tooling.context import ...` working today. The structural merge is sound. The follow-up work is hygiene and bringing the cqs tests into the real `Genie` API.

---

## Critical

### C0. Bundled cqs rules reference plugins and tools that don't exist [NEW]

Discovered during source-read after the user pointed out that passing tests don't prove cqs works. The bundled YAML rules at `src/genie_tooling/context/rules/*.yml` were sample data that was never validated against the actual registered plugin set.

**Concrete defects:**
- `01_fact_finding.yml` and `fact_finding_rule.yml` both define `rule_id: RULE_FACT_FINDING` with **different action lists**. Load order is OS-dependent (`Path.glob("*.yml")`), so the active rule is non-deterministic.
- `01_fact_finding.yml:11` references `derivation_strategy_id: karta_lookup_derivation_v1` — the Karta package was an external optional dep of cqs-engine and is not in this repo. Rule has no path to execution.
- `02_calculate.yml:11`, `10_expert_user.yml:13`, `99_default_fallback.yml:11` all reference `derivation_strategy_id: agentic_derivation_v1` — **no such plugin is registered**. The real IDs are `generic_agent_derivation_v1` and `generic_tool_derivation_v1`. These rules will fail at `_derivation_step` and return an error every time.
- `fact_finding_rule.yml` references `tool_id: fact_lookup_tool` — not in the registered tool list. Even if the rule loaded, the tool execution would fail.
- `casual_chat_rule.yml:8` uses key `derivation_strategy` (no `_id` suffix). `_derivation_step` reads `derivation_strategy_id` and falls through to the default — the rule's action is silently ignored.
- `casual_chat_rule.yml` (priority 99, unconditional match on `predicate: "*"`) shadows `99_default_fallback.yml` (priority 999, also unconditional). The default fallback is dead code in the current bundle.

**Tests don't catch any of this** because every cqs test uses handcrafted in-memory rules with mock plugins. None of the tests load real bundled rules and validate references against the registered plugin set.

**Fix:**
- Delete the duplicates: pick one of `01_fact_finding.yml` / `fact_finding_rule.yml`, delete the other.
- Replace every `agentic_derivation_v1` with `generic_agent_derivation_v1` and add `command_processor_id` action (the strategy needs a processor to dispatch to — see `research_rule.yml` for the correct shape).
- Replace `karta_lookup_derivation_v1` with `generic_tool_derivation_v1` + a real bundled tool, or delete the rule.
- Replace `fact_lookup_tool` with a real bundled tool, or delete the rule.
- Fix `derivation_strategy` → `derivation_strategy_id` in `casual_chat_rule.yml`.
- Pick one between `casual_chat_rule.yml` and `99_default_fallback.yml`; delete the other.
- Add a startup validator on `FileSystemRuleEnginePlugin.load_rules()` that cross-checks every `derivation_strategy_id`, `command_processor_id`, and `tool_id` against the registered plugin manager and fails loud (or warns loud) at boot time.

### C1. cqs unit tests rely on a stub `Genie.create()` that no longer exists

**Files:**
- `tests/unit/context/test_bootstrap.py:17`
- `tests/unit/context/test_manager_pluggable.py:90`
- `tests/unit/context/test_rule_engines.py:48`
- `tests/unit/context/conftest.py` (mock infrastructure)

When the cqs satellite ran in isolation, `tests/scaffolds/genie_tooling/src/genie_tooling/genie.py` provided a permissive `Genie` stub that accepted `config=<dict>` and exposed plugin_manager injection differently. The conftest path hack pointed `sys.path` at that scaffold.

After consolidation the scaffold path is gone (it would shadow the real package) and the tests now hit `genie_tooling.genie.Genie.create()` for real. Real `Genie.create()` requires a `MiddlewareConfig`, not a dict — line 229 of `src/genie_tooling/genie.py` calls `config.key_provider_id` directly. The mock plugin manager fixture is also no longer plumbed through the right path: in `test_manager_pluggable`, the `ContextManager` falls through to "Failed to load ContextSourcePlugin 'in_memory_user_profile_source_v1'" because the mock PM never reaches the manager.

**Fix:** update the three failing tests to construct a `MiddlewareConfig` (with `extension_configurations={"context_engine": {...}}`) and to plumb the mock plugin manager through `Genie.create(..., plugin_manager=mock_pm)`. The `test_vectordb_engine_evaluation_flow` failure is a separate signature drift — `VectorDBRuleEnginePlugin.evaluate(self, inferred_context, predicate, *, genie)` requires `genie=` but the test calls it without. Pass a sentinel/mock.

### C2. Default `rules_path = "./context_rules"` is CWD-relative and won't resolve for installed users

**Files:**
- `src/genie_tooling/context/plugins/rule_engines/filesystem_engine.py:24`
- `src/genie_tooling/context/plugins/rule_engines/vectordb_engine.py:40`

Both rule engines default to `Path("./context_rules")`. The bundled sample rules now live at `src/genie_tooling/context/rules/` inside the installed package. A user who pip-installs the package and configures `filesystem_rule_engine_v1` with no `rules_path` will get a directory-not-found warning even though sample rules are available.

**Fix:** resolve the default via `importlib.resources.files("genie_tooling.context") / "rules"`, falling back to `./context_rules` (CWD) for backward compatibility. Same change in both engines.

### C3. Bundled rule and prompt files not declared as package data

**File:** `pyproject.toml:9`

`packages = [{include = "genie_tooling", from = "src"}]` — Poetry's default `include` will pick up `.py` and `.typed` files but not necessarily `.yml` or `.prompt` files. If wheels exclude the bundled `context/rules/*.yml` and `context/prompt_templates/*.prompt`, installed users get an empty package even after C2 is fixed.

**Fix:** add explicit pattern-include for the data files. With Poetry:
```toml
include = [
    {path = "src/genie_tooling/context/rules/*.yml", format = ["sdist", "wheel"]},
    {path = "src/genie_tooling/context/prompt_templates/*.prompt", format = ["sdist", "wheel"]},
]
```
Or move the assets out of the package to `examples/context_rules/` and document that users copy them.

---

## High

### H1. Runtime signature introspection in `context/manager.py` papering over a protocol drift

**File:** `src/genie_tooling/context/manager.py:61-77` and `124-130`

```python
load_rules_sig = inspect.signature(self._rule_engine.load_rules)
if 'genie' in load_rules_sig.parameters:
    await self._rule_engine.load_rules(genie=self._genie)
else:
    await self._rule_engine.load_rules()
```

The manager calls `load_rules` (and `evaluate`) with conditional arg lists depending on whether the implementation accepts `genie`. This was added because `VectorDBRuleEnginePlugin` needs the facade and `FileSystemRuleEnginePlugin` doesn't. The original "FIX" comments in the source make this explicit.

Two interface variants for the same protocol is fragile: future plugin authors won't know which signature is expected, and the protocol declaration in `protocols.py` doesn't reflect either variant accurately.

**Fix:** make `genie` a non-optional keyword arg on `RuleEnginePlugin.load_rules` and `RuleEnginePlugin.evaluate`. Update `FileSystemRuleEnginePlugin` to accept (and ignore) the arg. Delete the runtime introspection.

### H2. Agents and command processors reach into private managers

**Files (13+ call sites):**
- `src/genie_tooling/agents/react_agent.py:79, 83`
- `src/genie_tooling/agents/plan_and_execute_agent.py:53, 56`
- `src/genie_tooling/agents/deep_research_agent.py:197, 203`
- `src/genie_tooling/agents/math_proof_assistant_agent.py:160, 161`
- `src/genie_tooling/command_processors/impl/rewoo_processor.py:708, 716`
- `src/genie_tooling/command_processors/impl/llm_assisted_processor.py:83, 107`
- `src/genie_tooling/context/manager.py:104`

All four agents, two command processors, and the new context manager bypass the `Genie` facade and reach into `genie._tool_manager.X(...)` or `genie._plugin_manager.X(...)`. Most call sites carry a `# type: ignore` because the underscore prefix marks them as private.

**Fix:** expose the two methods used everywhere — `list_tools(enabled_only=bool)` and `get_formatted_tool_definition(tool_id, formatter_id)` — as `genie.tools.list(...)` and `genie.tools.get_definition(...)` on the public facade. Then sweep the call sites. Same treatment for the one `_plugin_manager` reference in `context/manager.py`.

### ~~H3. ReActAgent reimplements JSON parsing~~ [REVISED — not a bug]

Original claim: `react_agent.py:139` uses raw `json.loads(tool_params_json_str or "{}")` instead of the framework's parser plugin.

On second look, the call site is parsing free-form tool params with no known schema (params vary per tool). The `pydantic_output_parser_v1` plugin needs a Pydantic schema and doesn't apply here. The other agents (`PlanAndExecuteAgent`, `MathProofAssistantAgent`) use `parse_output` because they have known schemas (`PlanModelPydantic`, `IntentResponse`). ReActAgent's situation is genuinely different.

The existing `json.loads` is defensive (try/except, falls through to LLM feedback on error) and appropriate. Closing this finding.

### H4. `pytest-asyncio` version mismatch in dev deps

**File:** `pyproject.toml:48`

Pinned at `^0.23.8`; uv resolved `1.3.0`. pytest-asyncio 1.x changed the fixture API — the legacy `event_loop` fixture was removed, causing 16 `event_loop` fixture errors in `tests/unit/test_genie_facade.py` and `tests/unit/caching/impl/test_in_memory_cache.py`.

**Fix:** tighten the pin to `pytest-asyncio = "~0.23.8"` (caret accepts ^0.23 → ^1.x, tilde stays in 0.23.x) OR migrate the failing tests to the new fixture model (`pytest_asyncio.fixture` + scope). The pin fix is one line and unblocks 16 tests; the migration is the long-term correct call.

---

## Medium

### M1. Ruff config uses deprecated top-level keys

**File:** `pyproject.toml:92-96`

```toml
[tool.ruff]
select = ["E", "W", ...]
ignore = ["E501", ...]
```

Ruff 0.5+ moved these under `[tool.ruff.lint]`. Current config emits a deprecation warning on every run.

**Fix:**
```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "C", "B", "Q", "ASYNC", "NPY", "PYI", "S", "A", "PT", "RUF"]
ignore = ["E501", "B008", "S101", "C901"]
```

### M2. 120 ruff errors across `src/`, 59 in `context/`

Top categories (whole src):
- 35 × `RUF022` unsorted `__all__` (cosmetic)
- 25 × `W293` blank-line-with-whitespace (auto-fixable)
- 11 × `W292` missing-newline-at-end-of-file (auto-fixable)
- 8 × `A002` builtin-argument-shadowing (e.g. `type=...`, `id=...`)
- 8 × `E701` multiple-statements-on-one-line (style)
- 6 × `E721` type-comparison (`type(x) == Y` instead of `isinstance`)
- 4 × `ASYNC240` blocking-path-method-in-async-function — actual correctness smell, not cosmetic
- 2 × `ASYNC230` blocking-open-call-in-async-function — same

**Fix:** run `ruff check --fix` to auto-resolve 81 of them. Audit the remaining 39 by hand (`A002`, `E701`, `E721`, `ASYNC*`, and a few security warnings — see M3).

### M3. Localized security-warning hits

**File-level findings:**
- `S301` suspicious-pickle-usage — 1 site. Pickle deserialization can execute arbitrary code; replace with JSON if possible.
- `S324` hashlib-insecure-hash-function — 1 site. `hashlib.md5()`/`hashlib.sha1()` used somewhere; if for caching keys it's fine but should be marked `usedforsecurity=False`.
- `S608` hardcoded-sql-expression — 1 site. Inspect for injection risk.
- `S110` try-except-pass — 1 site. Silently swallowing exceptions is a debugging hazard.

**Fix:** run `ruff check --select S301,S324,S608,S110 --no-fix src/` to get the exact file:line, then triage individually. None are obvious crit-vulns but each warrants a comment explaining why it's safe.

### M4. cqs `context/__init__.py` is empty — no re-exports

**File:** `src/genie_tooling/context/__init__.py`

Just `# Mark the directory as a Python package.` — no public surface. Users must import deep paths like `from genie_tooling.context.manager import ContextManager`. The rest of the package (`agents/`, `rag/`, etc.) follows the same pattern, so this isn't inconsistent — but if the goal is "λ-CQS is the headline differentiator," exposing `ContextManager`, `ContextInterface`, and the protocols at `genie_tooling.context` would help.

**Fix:** add re-exports in `context/__init__.py`:
```python
from .interface import ContextInterface
from .manager import ContextManager
from .protocols import (
    ContextSourcePlugin, ContextInferencePlugin, PredicateExtractorPlugin,
    RuleEnginePlugin, DerivationStrategyPlugin, FormulationStrategyPlugin,
)
from .types import InferredContext, RuleObject
__all__ = [...]
```

### M5. Plugin ID `cqs_engine_bootstrap_v1` is now namespace-inconsistent

**File:** `src/genie_tooling/context/bootstrap.py:19`

The module moved to `genie_tooling.context.bootstrap` but the plugin ID still announces "cqs_engine". Users who configure plugins by ID will see a mismatch between the import path and the identifier.

**Fix:** add a `context_engine_bootstrap_v1` ID and keep `cqs_engine_bootstrap_v1` as a deprecated alias for one minor cycle. Register both in pyproject. Document in CHANGELOG.

### M6. Two `tests/integration/__init__.py` paths but no top-level integration runner

Pytest's `testpaths = ["tests"]` picks up `tests/integration/` as well, but during this audit I ran `--ignore=tests/integration` because integration tests need API keys / live services. There's no obvious CI matrix that separates unit vs integration.

**Fix:** introduce explicit pytest markers (the `needs_api_key` marker is already declared in `[tool.pytest.ini_options]` but appears underused) and a CI job that runs only marker-tagged subsets when secrets are available.

---

## Low

### L1. OpenAI SDK pinned to `openai = {version = "^1.10.0", optional = true}` (Jan 2024)

Modern features (parallel tool calls, `response_format: json_schema`, vision in chat completions, streaming tool calls) require newer SDK versions. Not blocking, but anchors the framework to 2-year-old provider patterns.

**Fix:** bump to `^1.50` or current, run the openai test suite, update any usage that relies on deprecated request shapes.

### L2. Stale TODO

**File:** `src/genie_tooling/task_queues/impl/celery_queue.py:43`

```python
# TODO: Consider how tasks are discovered/registered if this plugin is generic.
```

Open question that's been deferred. Either resolve or convert to a tracked issue.

### L3. `arxiv`, `sympy`, `pypdf`, `rank-bm25` listed as required in pyproject

**File:** `pyproject.toml:35, 43, 44`

These are dependencies of agent-specific tools (research agents, math proof assistant). Most users of the core library won't need them. They were marked `Now required, not optional` in the agents satellite pyproject — suggests this is intentional, but it makes the install footprint larger than necessary.

**Fix (if footprint matters):** move to optional, gate behind `research_tools` and `symbolic_math` extras (the extras already exist; they just have no required deps). The relevant tool plugins would need defensive `try: import sympy` patterns at module load to avoid hard failures when extras aren't installed. Not urgent.

### L4. cqs `tests/scaffolds/` removed at copy time — verify nothing else referenced it

I dropped `tests/scaffolds/genie_tooling/` when moving cqs tests (rationale: the consolidated tree has the real package, scaffolds are dead weight). The conftest's `sys.path.insert(...)` was cleaned up. Spot-check: nothing in `tests/unit/context/` still references `scaffolds/` after grep. Confirmed clean.

### L5. Two README files now redundant

`README.md` at repo root is the core's README. The cqs `README.md` was not moved — its content is similar enough that it would duplicate, but the λ-CQS philosophy section (in `cqs-engine/docs/overview.md` from the original satellite — also not moved) has more substance than anything in the core README.

**Fix:** rewrite the core README's "Core Concepts" section to include a paragraph on the context engine, with a pointer to a new `docs/guides/context_scoping.md` (port the original cqs overview).

---

## Verification artifacts

- `find src/genie_tooling/context -type f | wc -l` → **54 files** (Python, YAML rules, .prompt templates).
- `importlib.metadata.entry_points()` → **77 plugins**, **1 bootstrap**. All 10 context-related plugin entry points present and resolvable.
- `python -c "from genie_tooling.context.<X> import <Y>"` succeeds for: `bootstrap.CqsEngineBootstrapPlugin`, `manager.ContextManager`, `interface.ContextInterface`, all 6 protocols, all 9 plugin implementations.
- `pytest tests/ --ignore=tests/integration` → **1320 passed, 3 failed, 16 errored** (failures and errors as detailed above).
- Original `/home/kal/code/genie-tooling/cqs-engine/` and `/home/kal/code/genie-tooling/genie-tooling-agents/` are untouched — still on disk, safe to delete after the user verifies this audit.

## Recommended priority order

1. C1 (fix the 3 cqs test failures) and H4 (pin pytest-asyncio) — these together restore the test suite to fully green.
2. C2 + C3 (bundled assets resolution + packaging) — without these, the cqs feature is broken for installed users.
3. H1 (unify rule-engine protocol) — kills the runtime introspection.
4. H2 (public tool API on facade) — removes 13+ `_underscore_manager` accesses.
5. H3 (ReActAgent uses parser plugin) — small change, big consistency win.
6. M1 + M2 (ruff config migration + autofix) — runs in minutes.
7. Everything else as time permits.

---

## Phase 3 status — applied 2026-05-24

All Phase 3 fixes have been applied. Remaining audit findings have been re-classified or downgraded as follows:

### Resolved during Phase 3

- **C0** (broken bundled cqs rules) — fixed: deleted 2 duplicate rule files, replaced `agentic_derivation_v1`/`karta_lookup_derivation_v1`/`fact_lookup_tool` with real registered IDs, fixed `derivation_strategy` → `derivation_strategy_id` typo. New `tests/unit/context/test_bundled_rules.py` guards against regression by cross-checking every action against the registered plugin set.
- **C1** (cqs unit tests against stub Genie) — rewrote `test_bootstrap.py` to exercise the bootstrap plugin in isolation; added `spec=` to MagicMocks in `test_manager_pluggable.py`; passed `genie=` to `VectorDBRuleEnginePlugin.evaluate` in `test_rule_engines.py`. All 22 cqs tests now green.
- **C2** (CWD-relative rules path) — both rule engines now resolve `rules_path` default via `importlib.resources.files("genie_tooling.context") / "rules"`. New test `test_filesystem_engine_default_path_resolves_to_bundled_rules` asserts the default points at on-disk bundled rules.
- **C3** (bundled assets not in wheel) — added explicit `include = [...]` patterns to `pyproject.toml` for `*.yml` and `*.prompt`. Verified by building a wheel and confirming the files appear under `genie_tooling/context/rules/` and `prompt_templates/`.
- **H1** (runtime signature introspection) — `RuleEnginePlugin` protocol now declares `load_rules(genie)` and `evaluate(..., genie)` as the canonical contract. `FileSystemRuleEnginePlugin` accepts-and-ignores. `ContextManager` calls both methods unconditionally with `genie=self._genie`. The `inspect.signature` branches in `manager.py` are gone.
- **H2** (private manager access) — `ToolsInterface` and `PluginsInterface` added to `interfaces.py` and attached to `Genie` as `genie.tools` and `genie.plugins`. All 13+ `genie._tool_manager.X` / `genie._plugin_manager.X` call sites in agents, command processors, and `context/manager.py` were swept. The pyproject's optional dep block remains the only "register stuff at startup" place.
- **H3** — re-examined; the original recommendation was misaligned (ReActAgent parses free-form tool params with no Pydantic schema; the framework's parser plugin needs one). Closed without change.
- **H4** — `pytest-asyncio` pin clarified with a comment in pyproject; venv reinstalled with the right version. 16 erroring tests now pass.
- **M1** (deprecated ruff config) — migrated `[tool.ruff]` → `[tool.ruff.lint]`. No deprecation warning.
- **M2** (120 ruff errors) — autofix resolved 76. Manual triage handled:
  - 6× E721 (`type(x) == Y`) → `is` comparison.
  - 1× E722 bare `except:` → `except Exception:`.
  - 1× S324 md5 for cache key → `usedforsecurity=False`.
  - 1× S301 pickle.loads → justification comment + `# noqa: S301`. The doc store file is written and read by the same plugin to an app-controlled location; not untrusted input.
  - 1× S110 try-except-pass → narrowed to `except json.JSONDecodeError` with debug log.
  - 1× S608 → false positive on a log message; `# noqa: S608` with comment.
  - 2× RUF059 unused unpacks → prefixed with `_`.
  - 4× ASYNC230/ASYNC240 in the cqs rule engines → wrapped in `asyncio.to_thread`.
- **F2b/F2c/F2d** (new tasks generated from user feedback) — see audit C0 above for what was discovered and fixed.

### Knowingly deferred (low-leverage at this stage)

- **24 remaining ruff findings**, all in non-cqs core files: 4× ASYNC240 (setup-time `Path.is_dir`/`Path.resolve` in `document_loaders/impl/file_system.py`, `prompts/impl/file_system_prompt_registry.py`, `tools/impl/sandboxed_fs_tool.py` — stat syscalls at init; impact is minor), 8× A002 builtin-arg-shadowing (kwargs named `type=`/`id=` — calling-convention preference), 8× E701 multi-statements-on-one-line (style), 4× RUF022 unsorted `__all__` (cosmetic).
- **L1, L2, M5, M6** — left for Phase 4 (modernization) and a future minor release.

### Test counts after Phase 3

`pytest tests/ --ignore=tests/integration`: **1346 passed, 0 failed, 0 errored** (up from 1320 passed / 3 failed / 16 errored at audit start). 22 of those are new or rewritten cqs tests covering the bundled rule integrity and the public facade APIs.

`pytest tests/integration/context/`: **3 passed** — new live tests against a real Ollama (gemma3:4b) instance exercising the full cqs pipeline including LLM-driven context inference, deterministic calculator derivation, and LLM-driven formulation.

---

## Bugs discovered via live testing (added 2026-05-24)

Running the cqs pipeline against real Ollama surfaced two real bugs that the mocked unit tests could never have caught:

### F14. PluginManager.get_all_plugin_instances_by_type poisons the instance cache [FIXED]

**File:** `src/genie_tooling/core/plugin_manager.py:219`

`get_all_plugin_instances_by_type` was iterating *every* discovered plugin class and calling `get_plugin_instance(plugin_id, config={})` to check the type — effectively pre-instantiating *all* plugins with an empty config during bootstrap discovery. The instance cache then returned those empty-config instances on subsequent calls, silently dropping any per-call config like `rules_path` for `FileSystemRuleEnginePlugin`.

This bug made it impossible to override any plugin's setup config after the first call to `get_all_plugin_instances_by_type`. Bootstrap discovery runs that call for `BootstrapPlugin` at the end of `Genie.create()`, so it bit every plugin downstream.

**Symptom (caught by live test):** configuring `extension_configurations["context_engine"]["rule_engine_config"]["rules_path"]` to point at a custom rules dir was silently ignored — the rule engine kept loading the bundled rules.

**Fix:** filter classes by `protocol in plugin_class.__mro__` *before* instantiating. Avoids the cache poisoning. Required because `issubclass(SomeClass, SomeProtocol)` raises `TypeError: Protocols with non-method members don't support issubclass()` for the typical Plugin Protocol shape, so the MRO check is the practical alternative for first-party plugins (which all explicitly inherit from their Protocol).

A unit test `test_get_all_plugin_instances_by_type` previously asserted the broken behavior (expected a Beta plugin to be set up with a config from an Alpha-typed query) — updated to assert the correct behavior.

### F13. FeatureSettings lacks `llm_ollama_base_url` [open]

**File:** `src/genie_tooling/config/features.py`

The `FeatureSettings` shortcut for Ollama exposes only `llm_ollama_model_name`. There's no `llm_ollama_base_url` field, so users with Ollama running on a remote host can't configure it via the high-level shortcut — they have to bypass `FeatureSettings` and populate `llm_provider_configurations["ollama_llm_provider_v1"]["base_url"]` directly. Trivial gap to close.

**Fix:** add an optional `llm_ollama_base_url: Optional[str]` field to `FeatureSettings`, wire it into `resolver.py` around line 86 alongside the existing `model_name` mapping. Defer to a follow-up since it's UX polish, not a correctness bug.

---

## Bundled agents fail with "Failed to render prompt" out of the box [open — surfaced by F15/live-agentic tests]

**Files:**
- `src/genie_tooling/agents/react_agent.py:92` — `render_prompt(name="react_agent_system_prompt_v1")`
- `src/genie_tooling/agents/plan_and_execute_agent.py:68` — `render_prompt(name="plan_and_execute_planner_prompt_v1")`

`ReActAgent.run()` and `PlanAndExecuteAgent.run()` both call `genie.prompts.render_prompt(name=<system_prompt_id>, data=...)` without a `template_content` fallback. The default IDs point at templates that **are not shipped anywhere in the package**:

```bash
$ grep -r "react_agent_system_prompt_v1\." src/genie_tooling/  # no template files
```

So out-of-the-box usage fails:

```python
agent = ReActAgent(genie=genie)
result = await agent.run(goal="...")
# -> {"status": "error", "output": "Failed to render ReAct prompt.", "history": [...]}
```

This is masked by the existing unit tests because they mock `genie.prompts.render_prompt` to return a canned string. Live agentic integration tests caught it.

**Recommended fix:** mirror the ReWOO processor's pattern — keep a `_DEFAULT_REACT_SYSTEM_PROMPT` class constant in `react_agent.py` (likewise for plan-and-execute) and call `render_prompt(template_content=cls._DEFAULT_REACT_SYSTEM_PROMPT, data=..., template_engine_id="jinja2_chat_template_v1")`. Users overriding via `agent_config["system_prompt_id"]` continue to work; users with no override get a working default.

**Workaround for users today:** ship the templates in a directory and point the FileSystemPromptRegistry at it via `prompt_registry_configurations` — the test workaround in `test_live_agentic_loop.py` demonstrates this.

---

## Bundled cqs prompt templates use Jinja syntax against str.format engine [open]

**Files:** `src/genie_tooling/context/prompt_templates/direct_fact_formulation.prompt`, `summarize_agent_output.prompt`

Both bundled templates use Jinja-style `{{ var.dotted.access }}`. The default prompt template engine is `basic_string_format_template_v1`, which uses Python's `str.format()` and renders `{{...}}` as a literal `{...}`. So the bundled templates produce literal placeholder strings instead of substituted values when rendered through the default engine.

Either:
- Convert templates to str.format-compatible syntax (single-brace, no dotted access — the templates would need restructuring since `raw_data.result.fact.value` style accesses aren't expressible with str.format).
- Make the cqs bootstrap default the prompt template engine to a Jinja-capable plugin. The bundled `jinja2_chat_template_v1` is chat-message-oriented, not plain-text; a `jinja2_template_v1` plain-text plugin would need to be added.

Either approach is out of Phase 3 scope; live tests work around it with a custom template.
