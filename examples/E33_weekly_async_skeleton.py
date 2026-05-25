"""E33 — Weekly Async planning agent skeleton (Phase 6 §9 first production use case).

Wires together the Phase 6 pieces that the weekly-async planning agent
needs: MCP composition for Linear/Slack/Notion, Claude-Code permissions
controlling who can write where, attribution tags + budget scoping
per team-run, durable checkpointing in case a worker dies mid-run,
audit ledger so every published draft is queryable.

This is a SKELETON — the actual ReActAgent flow that calls Linear /
Slack / Notion lives in your operator code (or an extension of this
script). What's demonstrated here is **the configuration**: how to
assemble a corporate-harness Genie that has all the safety + audit
+ MCP ingest properties wired in.

To actually run:
- Install the `mcp` extra: `poetry install --extras mcp`.
- Provide SLACK_BOT_TOKEN, LINEAR_API_KEY, NOTION_API_KEY in env.
- Point overlays_dir at the bundled mcp_overlays directory.
"""
import asyncio
import logging
import os
import tempfile
from pathlib import Path

from genie_tooling.budget import BudgetSpec
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


TEAMS = [
    {"name": "platform", "linear_team_id": "PLAT", "slack_channel": "#platform-async"},
    {"name": "search", "linear_team_id": "SRCH", "slack_channel": "#search-async"},
]


async def run_team(genie: Genie, team: dict, week_label: str) -> None:
    """The per-team agent run — placeholder. Replace with a ReActAgent
    that uses the discovered MCP tools."""
    scope = f"weekly_async:{team['name']}:{week_label}"
    await genie.budget.set_budget(scope, BudgetSpec(max_tokens=20_000, max_cost_usd=0.50, max_wall_clock_seconds=600))

    print(f"\n=== Drafting week {week_label} plan for team {team['name']} ===")
    # Here you'd assemble a ReActAgent with input_context including the
    # budget_scope, attribution_tags, and a Slack-thread progress sink.
    # See E29 for the ReActAgent wiring pattern.

    snap = await genie.budget.get_usage(scope)
    if snap:
        print(f"  budget after run: tokens={snap.tokens} llm_calls={snap.llm_calls}")


async def main():
    logging.basicConfig(level=logging.INFO)

    overlays_dir = (
        Path(__file__).parent.parent
        / "src"
        / "genie_tooling"
        / "tools"
        / "impl"
        / "mcp_overlays"
    )

    cfg = MiddlewareConfig(
        environment="production",
        features=FeatureSettings(
            llm="anthropic",
            llm_anthropic_model_name="claude-sonnet-4-6",
            hitl_approver_chain=["claude_code_permissions", "webhook_approval_v1"],
            budget_enforcer="in_memory_budget_enforcer",
            token_usage_recorder="in_memory_token_recorder",
            observability_tracer="otel_tracer",  # ship traces to your central collector
        ),
        default_hitl_ledger_id="sqlite_hitl_ledger_v1",
        hitl_ledger_configurations={
            "sqlite_hitl_ledger_v1": {"db_path": os.path.join(tempfile.gettempdir(), "weekly_async_hitl.sqlite")},
        },
        hitl_approver_configurations={
            "webhook_approval_v1": {
                # Route destructive ops to your on-call channel; everything else to the team channel.
                "routes": [
                    {
                        "match": {"side_effects_in": ["destructive"]},
                        "url": "https://hooks.example.com/oncall-approve",
                    },
                    {
                        "match": {"tool_id_in": ["mcp_notion_*"]},
                        "url": "https://hooks.example.com/team-approve",
                    },
                ],
                "default_url": "https://hooks.example.com/general-approve",
            },
            "claude_code_permissions_v1": {
                "policy_inline": {
                    "rules": [
                        # Auto-allow Notion page creation under the /team-plans/ parent.
                        {
                            "id": "ALLOW_TEAM_PLAN_CREATE",
                            "match": {
                                "tool_id_in": ["mcp_notion_create_page"],
                                "params_match": {"parent_id": "team_plans_*"},
                            },
                            "decision": "allow",
                            "reason": "weekly async drafts under /team-plans/ are pre-authorised",
                        },
                        # Slack progress notifications to the team channel — also auto.
                        {
                            "id": "ALLOW_TEAM_CHANNEL_POSTS",
                            "match": {
                                "tool_id_in": ["slack_post_message", "mcp_slack_postMessage"],
                                "params_match": {"channel": "#*-async"},
                            },
                            "decision": "allow",
                        },
                        # Everything else write/destructive → ask
                        {
                            "id": "ASK_REST",
                            "match": {"side_effects_in": ["write", "destructive"]},
                            "decision": "ask",
                        },
                    ],
                },
            },
        },
        extension_configurations={
            "mcp_composition": {
                "overlays_dir": str(overlays_dir),
                "servers": [
                    {"name": "linear", "command": "npx", "args": ["-y", "@linear/mcp-server"], "env": {"LINEAR_API_KEY": os.environ.get("LINEAR_API_KEY", "")}},
                    {"name": "slack", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-slack"], "env": {"SLACK_BOT_TOKEN": os.environ.get("SLACK_BOT_TOKEN", "")}},
                    {"name": "notion", "command": "npx", "args": ["-y", "@notionhq/notion-mcp-server"], "env": {"NOTION_API_KEY": os.environ.get("NOTION_API_KEY", "")}},
                ],
            },
        },
    )
    # In practice you'd run this from a cron job / k8s CronJob on Friday evening.
    # For demo purposes we just attempt setup and report what got wired up.
    genie = await Genie.create(config=cfg)
    try:
        tools = await genie.tools.list()
        print(f"\n{len(tools)} tools available across MCP servers and native plugins:")
        for t in sorted(tools, key=lambda t: t.identifier)[:20]:
            print(f"  - {t.identifier}")

        week_label = "W47"
        for team in TEAMS:
            try:
                await run_team(genie, team, week_label)
            except Exception as e:
                print(f"  team {team['name']!r} failed: {e}")
    finally:
        await genie.close()


if __name__ == "__main__":
    asyncio.run(main())
