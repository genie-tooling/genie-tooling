# E32 — Strict Replay Harness

Source: [`examples/E32_replay_harness.py`](https://github.com/genie-tooling/genie-tooling/blob/main/examples/E32_replay_harness.py)

Captures a live agent run's LLM round-trips + tool I/O into a JSON
fixture, then replays it deterministically. The replay run uses zero
LLM tokens and produces the same outputs as the recording.

## What it shows

* `ReplayRecorder.record_llm_chat(...)` and `record_tool_call(...)`
  capture each boundary call.
* `ReplayPlayer.from_fixture(path)` reads the recording back.
* `player.play_llm_chat(...)` serves recorded responses; raises
  `ReplayMiss` for unrecorded inputs (the replay is strict).
* `player.assert_exhausted()` catches *regressions* where the new run
  makes FEWER calls than the recording.

## Framework integration

In the example, recorder/player calls are made directly to keep the
mechanics visible. In real use, hand them to `LLMInterface`:

```python
genie.llm.set_replay_recorder(recorder)   # record live runs
# … later, in CI …
genie.llm.set_replay_player(player)       # replay deterministically
```

Once attached, `genie.llm.chat(...)` and `genie.llm.generate(...)` are
intercepted transparently — your agent code doesn't change.

## Run

```bash
poetry run python examples/E32_replay_harness.py
```

The script records 3 entries, replays them, and verifies the player was
fully exhausted (i.e. the replay run made exactly the same number of
calls as the recording).
