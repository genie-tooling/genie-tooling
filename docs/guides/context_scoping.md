# Context Scoping — The λ-CQS Guide

This guide walks through `genie_tooling.context`, the bundled λ-CQS
(Lambda Contextual Query Scoping) implementation. It exists because for
corporate / regulated / audit-bound deployments you need to be able to
answer two questions with confidence:

1.  *Why did the system route this query the way it did?*
2.  *Why did the system respond in this particular tone / format /
    redaction posture?*

A generic LLM-powered system answers both with "the model decided." That
isn't an answer audit teams can act on. λ-CQS makes both decisions
mechanical: deterministic rules in version control, applied by a
deterministic pipeline, captured in a single structured audit record per
query.

## The pipeline at a glance

```
                 ┌─────────────────────────────────────────┐
   user query    │                                         │
   + session     │                cqs pipeline             │
   + identity ──►│                                         │ ──► response
                 │  ContextSource → Inference → Predicate  │
                 │  → RuleEngine → Aggregate (C_D, C_F)    │
                 │  → Derivation → Formulation             │
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      │ emits
                                      ▼
                              DecisionRecord
                            (audit.decision_record)
```

Every stage is a swappable plugin:

| Stage | Plugin Protocol | Bundled implementations |
|---|---|---|
| Profile / history | `ContextSourcePlugin` | `configurable_context_source_v1` |
| Inference | `ContextInferencePlugin` | `llm_context_inference_v1` |
| Predicate | `PredicateExtractorPlugin` | `heuristic_predicate_extractor_v1`, `llm_predicate_extractor_v1` |
| Rule eval | `RuleEnginePlugin` | `filesystem_rule_engine_v1` (deterministic), `vectordb_rule_engine_v1` (semantic, deterministic mode default) |
| Derivation | `DerivationStrategyPlugin` | `generic_tool_derivation_v1`, `generic_agent_derivation_v1` |
| Formulation | `FormulationStrategyPlugin` | `llm_prompt_formulation_v1` |

## Why deterministic? Why YAML?

A YAML rule under version control is the audit substrate. When someone
asks "why did this query route to the calculator on Tuesday but to the
research agent on Wednesday?" the answer is:

```bash
$ git diff Tuesday..Wednesday -- src/genie_tooling/context/rules/
$ git blame src/genie_tooling/context/rules/02_calculate.yml
```

The deterministic `FileSystemRuleEnginePlugin` is the recommended choice
for audit-bound deployments. The `VectorDBRuleEnginePlugin` ships in
deterministic mode by default but supports an opt-in LLM-enriched mode
that's explicitly flagged as non-deterministic — use the semantic mode
only when fuzzy matching matters more than reproducibility.

## A rule, end to end

`src/genie_tooling/context/rules/02_calculate.yml`:

```yaml
rule_id: "RULE_CALCULATION"
predicate: "predicate_calculate"
priority: 10
description: "Handles mathematical calculation requests."
conditions:
  - ["AudienceProfile.intent", "==", "computation"]
actions:
  - ["C_D", "set", "derivation_strategy_id", "generic_agent_derivation_v1"]
  - ["C_D", "set", "command_processor_id", "llm_assisted_tool_selection_processor_v1"]
  - ["C_F", "set", "format", "direct_answer"]
```

Walked through:

1.  **Predicate match**. The `HeuristicPredicateExtractor` returns
    `"predicate_calculate"` if the user's query contains any of `is`,
    `are`, `what`, `calculate`, … (table-driven; see
    `tests/unit/context/test_phase4_coverage.py`). The rule only
    considers itself a candidate when the predicate matches its own.
2.  **Conditions**. The rule fires only when
    `AudienceProfile.intent == "computation"`. The `==` here is one of
    the supported operators (`==`, `!=`, `>`, `<`, `>=`, `<=`, `in`,
    `contains`). All operators are unit-test pinned for behavior
    determinism.
3.  **C_D ("derivation constraints")** — what to do. Set the derivation
    strategy and the command processor it should use.
4.  **C_F ("formulation constraints")** — how to say it. `format: direct_answer`
    is fed through the constraint translator to the formulation LLM as
    *"Provide the direct answer only, with no preamble or commentary."*

## The headline demo: same query, two routings, two audit trails

This is the proof that C_F constraints aren't theater. Two profiles, one
query, two visibly different responses and two DecisionRecords.

```python
import asyncio
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie

QUERY = "explain why this calculation is failing"

async def two_routings():
    common = dict(
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name="qwen3.6:35b",
            llm_ollama_base_url="http://192.168.68.58:11434",
            command_processor="llm_assisted",
            conversation_state_provider="in_memory_convo_provider",
            default_llm_output_parser="pydantic_output_parser",
            tool_lookup="none",
            hitl_approver="dev_auto_approve_hitl",  # dev only
        ),
        tool_configurations={"calculator_tool": {}},
        auto_enable_registered_tools=True,
    )

    # Profile 1: stressed expert
    stressed_cfg = MiddlewareConfig(
        **common,
        extension_configurations={
            "context_engine": {
                "context_source_config": {
                    "default_profile": {"expertise": "expert", "state": "Stressed"}
                },
            }
        },
    )
    stressed = await Genie.create(config=stressed_cfg)
    try:
        stressed_resp = await stressed.context.resolve_and_formulate(
            query=QUERY, session_id="user-1",
            user_identity={"sub": "u1", "role": "engineer"},
        )
        stressed_rec = stressed.context.last_decision
    finally:
        await stressed.close()

    # Profile 2: calm expert
    calm_cfg = MiddlewareConfig(
        **common,
        extension_configurations={
            "context_engine": {
                "context_source_config": {
                    "default_profile": {"expertise": "expert", "state": "Neutral"}
                },
            }
        },
    )
    calm = await Genie.create(config=calm_cfg)
    try:
        calm_resp = await calm.context.resolve_and_formulate(
            query=QUERY, session_id="user-2",
            user_identity={"sub": "u2", "role": "engineer"},
        )
        calm_rec = calm.context.last_decision
    finally:
        await calm.close()

    print("=== STRESSED (RULE_STRESSED_AUDIENCE) ===")
    print(f"  rule: {stressed_rec.winning_rule_id}")
    print(f"  C_F:  {stressed_rec.c_f}")
    print(f"  instructions: {stressed_rec.formulation_constraints_text}")
    print(f"  response: {stressed_resp}")
    print()
    print("=== CALM (RULE_EXPERT_AUDIENCE) ===")
    print(f"  rule: {calm_rec.winning_rule_id}")
    print(f"  C_F:  {calm_rec.c_f}")
    print(f"  instructions: {calm_rec.formulation_constraints_text}")
    print(f"  response: {calm_resp}")

asyncio.run(two_routings())
```

You get two distinct decision records:

```
=== STRESSED (RULE_STRESSED_AUDIENCE) ===
  rule: RULE_STRESSED_AUDIENCE
  C_F:  {'tone': 'calm and reassuring', 'verbosity': 'concise', 'empathy_level': 'high'}
  instructions: Response guidelines:
                  - Use a calm and reassuring tone in your response.
                  - Keep your response concise — one or two sentences.
                  - Show genuine empathy in your phrasing. Acknowledge the user's emotional state.
  response: I can see this is frustrating. The calculation is failing because…

=== CALM (RULE_EXPERT_AUDIENCE) ===
  rule: RULE_EXPERT_AUDIENCE
  C_F:  {'tone': 'formal and technical', 'verbosity': 'high'}
  instructions: Response guidelines:
                  - Use a formal and technical tone in your response.
                  - Provide a detailed, thorough response.
  response: The calculation fails at evaluation because the input domain…
```

Same query, same model, same code path. The only difference is the
profile that the rule engine matched on. **Both responses are reproducible
from their respective DecisionRecord** — feed the same query + profile
in tomorrow and you get the same rule match, the same C_F dict, the same
constraint-instruction text, and (modulo LLM stochasticity inside the
formulation step) the same response shape.

## The DecisionRecord

Every `resolve_and_formulate` call emits exactly one `audit.decision_record`
trace event AND populates `genie.context.last_decision`. The record carries:

- `decision_id`, `session_id`, `user_identity`, `query`
- `inferred_context` — the LLM's view of who's asking
- `predicate`, `predicate_extractor_id`
- `rule_engine_id`, `ranked_rules` (full ranked list, not just the winner),
  `winning_rule_id`
- `c_d`, `c_f` — the aggregated constraint dicts
- `derivation_strategy_id`, `derivation_status`, `derivation_result_preview`
- `formulation_strategy_id`, `formulation_template_id`,
  **`formulation_constraints_text`** (the exact instruction text the LLM saw)
- `final_response`
- `stage_timings_ms` per pipeline stage
- `started_at`, `completed_at`

Subscribe an OpenTelemetry exporter or a custom log adapter to the
`audit.decision_record` event channel to ship records into a SIEM / log
warehouse. Or grab them in-process via `genie.context.last_decision`
during tests.

## Validation: broken rules fail at startup, not at first match

```python
# This file references a plugin that doesn't exist:
$ cat src/genie_tooling/context/rules/typo.yml
rule_id: "TYPO"
predicate: "predicate_calculate"
priority: 1
conditions: []
actions:
  - ["C_D", "set", "derivation_strategy_id", "agentic_derivation_v1"]  # typo!
```

```python
$ python -c "
import asyncio
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings
from genie_tooling.genie import Genie

async def main():
    cfg = MiddlewareConfig(features=FeatureSettings(llm='ollama'),
                            extension_configurations={'context_engine': {}})
    await Genie.create(config=cfg)

asyncio.run(main())
"
RuleValidationError: Rule validation failed with 1 error(s):
  - rule_id 'TYPO': derivation_strategy_id='agentic_derivation_v1' is not a registered plugin.
```

You find out at startup, not the first time a stressed user asks a question.
Audit teams ship the YAML; the framework refuses to load anything broken.

## Operational: rule reload at runtime

When governance updates a rule, `genie.context.reload_rules()` picks up
the new content atomically. If the new content fails validation, the
previous rule set stays active.

```python
ok = await genie.context.reload_rules()
if not ok:
    # Reload failed; previous rules still active. Check logs for the
    # specific validation error.
    ...
```

## Constraint vocabulary

The translator (`src/genie_tooling/context/constraints.py`) knows about
these keys:

| Key | Behavior |
|---|---|
| `tone` | "Use a `<value>` tone in your response." |
| `verbosity` | Maps `low/concise` → "Keep your response concise — one or two sentences." Similarly `moderate` and `high`. |
| `format` | `direct_answer` → "Provide the direct answer only, with no preamble or commentary." `bullet_list`, `numbered_list`, `json` also recognized. |
| `empathy_level` | `high` → "Show genuine empathy. Acknowledge the user's emotional state." `low` → "Keep the response neutral and factual; do not editorialise." |
| `redact` | `["pii", "internal_terms"]` → "Do NOT mention any of the following in your response: pii, internal_terms." Also accepts a single string. |
| `audience_register` | "Match your language register to a `<value>` audience." |
| `persona` | "Adopt this persona for your response: `<value>`." |

Unknown keys fall through to a generic template
(`"Apply this guideline: <key> = <value>."`) so a rule author can
introduce a new key without a code change. Add the key to the known-key
registry when you want sharper steering.

## HITL: from CLI to webhook to policy

The bundled HITL approvers cover four corporate postures:

1.  **`cli_approval_plugin_v1`** — interactive CLI prompt. Useful only
    in development.
2.  **`dev_auto_approve_hitl_v1`** — always approves. Loud warning if
    used while `MiddlewareConfig.environment == "production"`. Use in
    CI and prototyping.
3.  **`webhook_approval_v1`** — POSTs each approval request to a
    configured URL; expects a JSON response with `{status: "approved" | "denied"}`.
    Plug a Slack interactive-message endpoint, a Teams adaptive card,
    or a JIRA/ServiceNow ticket workflow behind this.
4.  **`policy_auto_approve_hitl_v1`** — reads a YAML policy file; each
    decision logged with the matching policy ID. Suitable for high-volume
    deployments where most decisions are deterministic ("admins can write,
    nobody else can") with human-in-the-loop only for the edge cases.

`ReActAgent` ALSO has a per-action HITL gate
(`agent_config["hitl_per_action"] = True`) that fires before every tool
call inside a single agent invocation — useful when the agent loop
itself might invoke destructive tools and you want a per-action review.

## Where to look

- Source: `src/genie_tooling/context/`
- Tests: `tests/unit/context/`, `tests/integration/context/`
- Bundled rules: `src/genie_tooling/context/rules/`
- Bundled prompt templates: `src/genie_tooling/context/prompt_templates/`
- Audit schema: `src/genie_tooling/context/audit.py`
- Constraint translator: `src/genie_tooling/context/constraints.py`
