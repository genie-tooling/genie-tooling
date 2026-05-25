"""E32 — Strict replay harness end-to-end (Phase 6D.2).

Captures a live "agent loop" worth of LLM + tool calls into a fixture file,
then replays it deterministically. Use this pattern for CI regression tests
of policy / prompt / agent-loop changes.

The actual wiring of recorder/player to your live LLM provider is application
code; here we use the primitives directly to make the round-trip concrete.
"""
import asyncio
import os
import tempfile

from genie_tooling.replay import ReplayPlayer, ReplayRecorder


async def main():
    fixture = os.path.join(tempfile.gettempdir(), "demo_replay.json")

    # --- record ---
    rec = ReplayRecorder(fixture_path=fixture, run_label="sev2_runbook")
    rec.record_llm_chat(
        provider_id="anthropic",
        messages=[{"role": "user", "content": "What's the temperature in London?"}],
        sampling_kwargs={"temperature": 0.0},
        response={
            "message": {"role": "assistant", "content": None,
                        "tool_calls": [{"id": "c1", "type": "function",
                                        "function": {"name": "get_weather", "arguments": '{"city": "London"}'}}]},
        },
    )
    rec.record_tool_call("get_weather", {"city": "London"}, {"temperature_c": 12.0})
    rec.record_llm_chat(
        provider_id="anthropic",
        messages=[
            {"role": "user", "content": "What's the temperature in London?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "get_weather"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"temperature_c": 12.0}'},
        ],
        sampling_kwargs={"temperature": 0.0},
        response={"message": {"role": "assistant", "content": "It's 12°C in London."}},
    )
    rec.save()
    print(f"Recorded {len(rec.recording.entries)} entries to {fixture}")

    # --- replay ---
    player = ReplayPlayer.from_fixture(fixture)
    # First LLM round-trip
    first_chat = player.play_llm_chat(
        "anthropic",
        [{"role": "user", "content": "What's the temperature in London?"}],
        {"temperature": 0.0},
    )
    assert "tool_calls" in first_chat["message"]

    # Tool dispatch
    tool_result = player.play_tool_call("get_weather", {"city": "London"})
    assert tool_result == {"temperature_c": 12.0}

    # Second (final) round-trip
    final = player.play_llm_chat(
        "anthropic",
        [
            {"role": "user", "content": "What's the temperature in London?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "function": {"name": "get_weather"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"temperature_c": 12.0}'},
        ],
        {"temperature": 0.0},
    )
    print(f"Replay final answer: {final['message']['content']}")

    # Make sure the recorded fixture was fully consumed — if the agent loop
    # changed and made fewer calls, this would raise.
    player.assert_exhausted()
    print(f"Player stats: {player.stats}")


if __name__ == "__main__":
    asyncio.run(main())
