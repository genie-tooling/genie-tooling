"""E31 — Budget enforcement + audit ledger end-to-end (Phase 6A.3 + 6A.7).

Sets a session budget cap, runs LLM calls, watches the budget enforce; then
queries the durable approval ledger for everything that happened in the
session.

This is the corporate-harness flow in miniature: every call attributed to
a session, capped on tokens and cost, with every approval (if any) durably
recorded.
"""
import asyncio
import logging
import os
import tempfile

from genie_tooling.budget import BudgetSpec, BudgetExceeded
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


async def main():
    logging.basicConfig(level=logging.INFO)

    ledger_db = os.path.join(tempfile.gettempdir(), "genie_demo_ledger.sqlite")

    cfg = MiddlewareConfig(
        environment="development",
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name="qwen3.6:35b",
            token_usage_recorder="in_memory_token_recorder",
            budget_enforcer="in_memory_budget_enforcer",
        ),
        default_hitl_ledger_id="sqlite_hitl_ledger_v1",
        hitl_ledger_configurations={
            "sqlite_hitl_ledger_v1": {"db_path": ledger_db},
        },
    )
    genie = await Genie.create(config=cfg)
    try:
        # Set a tight budget for this session.
        await genie.budget.set_budget(
            "incident:SEV2-1234",
            BudgetSpec(max_tokens=500, max_llm_calls=3, max_cost_usd=0.10),
        )

        # Run an LLM call attributed to that session.
        try:
            resp = await genie.llm.chat(
                messages=[{"role": "user", "content": "One sentence on Genie Tooling."}],
                budget_scope="incident:SEV2-1234",
                attribution_tags={"incident": "SEV2-1234", "team": "platform"},
            )
            print(f"Response: {resp['message']['content'][:120]}...")
        except BudgetExceeded as e:
            print(f"Budget refused the call: {e}")

        snap = await genie.budget.get_usage("incident:SEV2-1234")
        if snap:
            print(f"Session usage so far: tokens={snap.tokens} llm_calls={snap.llm_calls}")
    finally:
        await genie.close()


if __name__ == "__main__":
    asyncio.run(main())
