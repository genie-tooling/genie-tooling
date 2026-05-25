# Changelog

The full changelog lives at the repository root:
[CHANGELOG.md](https://github.com/genie-tooling/genie-tooling/blob/main/CHANGELOG.md).

## Latest: 0.2.0 (2026-05-25) — "corporate-harness + 2026 modernization"

Highlights:

- **Phase 3** — post-consolidation hardening. `ToolsInterface` /
  `PluginsInterface` on the facade; unified `RuleEnginePlugin` protocol;
  `cqs-engine` folded into core as `genie_tooling.context`; bundled
  agent system prompts; several reversed-semantics bugs fixed.
- **Phase 4** — corporate-harness audit hardening. **C_F constraints
  now actually reach the LLM** (deterministic constraint translator);
  rule schema validation at load time; **`DecisionRecord` audit schema**;
  `WebhookApprovalPlugin` and `PolicyAutoApproveHITLPlugin`; per-action
  HITL gate on `ReActAgent`; tool-execution `caller_chain` provenance;
  runtime rule reload; structured `guardrail.decision` events.
- **Phase 5** — 2026 modernization. **Anthropic provider** with native
  `tool_use`; **native structured outputs** (`response_schema=YourModel`)
  on OpenAI / Anthropic / Gemini; **vision in `ChatMessage`** via
  `ContentBlock`; **MCP client** + **MCP server bootstrap**; provider-native
  tool-use loops in `ReActAgent` with parallel tool calls.

See [CHANGELOG.md](https://github.com/genie-tooling/genie-tooling/blob/main/CHANGELOG.md)
for the full Phase 3 / 4 / 5 history.

## 0.1.0

Initial consolidated release after merging the three sibling repos
(`genie-tooling`, `genie-tooling-agents`, `cqs-engine`) into one package.
