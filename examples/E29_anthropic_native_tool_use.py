"""E29 — ReActAgent with native tool-use against Anthropic (Phase 5 M7 + Phase 6).

Demonstrates the modern path: ReActAgent uses provider-native tool_use
round-trips against Claude, with side-effect metadata on tools driving
HITL gating via the Claude-Code permission model.

Requires: ANTHROPIC_API_KEY in env, and the `anthropic` extra:
    poetry install --extras anthropic
"""
import asyncio
import logging
import os

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.decorators import tool
from genie_tooling.agents.react_agent import ReActAgent
from genie_tooling.genie import Genie


@tool(side_effects="read", idempotent=True, cacheable=True, cache_ttl_seconds=60)
async def get_current_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: Name of the city.

    Returns:
        Weather report dict.
    """
    # Mock implementation
    fake = {"London": 12.0, "San Francisco": 18.0, "Tokyo": 25.0}
    return {"city": city, "temperature_c": fake.get(city, 20.0), "condition": "cloudy"}


@tool(side_effects="none")
async def celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return c * 9 / 5 + 32


async def main():
    logging.basicConfig(level=logging.INFO)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY in the environment to run this example.")
        return

    cfg = MiddlewareConfig(
        environment="development",
        features=FeatureSettings(
            llm="anthropic",
            llm_anthropic_model_name="claude-sonnet-4-6",
            hitl_approver="claude_code_permissions",
        ),
        tool_configurations={},  # we register via @tool below
    )
    genie = await Genie.create(config=cfg)
    try:
        await genie.register_tool_functions([get_current_weather, celsius_to_fahrenheit])

        agent = ReActAgent(
            genie=genie,
            agent_config={
                "use_native_tool_use": True,
                "max_iterations": 5,
            },
        )
        result = await agent.run(
            goal="What's the weather in London in Fahrenheit?",
            input_context={"attribution_tags": {"demo": "E29"}},
        )
        print(f"\n=== status: {result['status']} ===")
        print(f"output: {result['output']}")
    finally:
        await genie.close()


if __name__ == "__main__":
    asyncio.run(main())
