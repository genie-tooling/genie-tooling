"""Phase 6D.2 — strict replay harness tests."""
from __future__ import annotations

import pytest
from genie_tooling.replay import (
    ReplayMiss,
    ReplayPlayer,
    ReplayRecorder,
)


def test_recorder_captures_three_call_kinds_in_order():
    r = ReplayRecorder(run_label="sev2_runbook")
    r.record_llm_chat("anthropic", [{"role": "user", "content": "hi"}], {"temperature": 0}, {"role": "assistant", "content": "hello"})
    r.record_tool_call("calculator", {"a": 1, "b": 2}, {"result": 3})
    r.record_llm_chat("anthropic", [{"role": "user", "content": "next"}], {"temperature": 0}, {"role": "assistant", "content": "ok"})
    rec = r.recording
    assert len(rec.entries) == 3
    assert [e.kind for e in rec.entries] == ["llm_chat", "tool_call", "llm_chat"]
    assert [e.sequence for e in rec.entries] == [0, 1, 2]


def test_player_replays_recorded_llm_chat_byte_for_byte():
    r = ReplayRecorder()
    msgs = [{"role": "user", "content": "what is 2+2"}]
    expected = {"role": "assistant", "content": "4"}
    r.record_llm_chat("openai", msgs, {"temperature": 0.0}, expected)

    p = ReplayPlayer(r.recording)
    actual = p.play_llm_chat("openai", msgs, {"temperature": 0.0})
    assert actual == expected
    assert p.stats["hits"] == 1


def test_player_serves_duplicate_hash_calls_in_recorded_order():
    """Two identical LLM calls in a row → two recorded responses → replay
    serves them in the same order."""
    r = ReplayRecorder()
    msgs = [{"role": "user", "content": "ping"}]
    r.record_llm_chat("ollama", msgs, {}, {"content": "first"})
    r.record_llm_chat("ollama", msgs, {}, {"content": "second"})

    p = ReplayPlayer(r.recording)
    first = p.play_llm_chat("ollama", msgs, {})
    second = p.play_llm_chat("ollama", msgs, {})
    assert first == {"content": "first"}
    assert second == {"content": "second"}


def test_player_raises_replay_miss_for_unrecorded_call():
    r = ReplayRecorder()
    r.record_llm_chat("openai", [{"role": "user", "content": "known"}], {}, {"content": "ok"})

    p = ReplayPlayer(r.recording)
    with pytest.raises(ReplayMiss) as ei:
        p.play_llm_chat("openai", [{"role": "user", "content": "unknown"}], {})
    assert ei.value.kind == "llm_chat"
    assert "unknown" in ei.value.request_summary


def test_player_tool_call_match():
    r = ReplayRecorder()
    r.record_tool_call("calc", {"a": 5, "b": 7}, {"result": 12})
    p = ReplayPlayer(r.recording)
    assert p.play_tool_call("calc", {"a": 5, "b": 7}) == {"result": 12}


def test_framework_level_kwargs_are_stripped_from_hash():
    """attribution_tags / session_id / budget_scope must not affect the
    replay hash — they're framework metadata, not LLM request content."""
    r = ReplayRecorder()
    msgs = [{"role": "user", "content": "x"}]
    r.record_llm_chat("openai", msgs, {"temperature": 0, "attribution_tags": {"team": "platform"}}, {"ok": True})

    p = ReplayPlayer(r.recording)
    # Same request but different attribution tags should still match.
    actual = p.play_llm_chat("openai", msgs, {"temperature": 0, "attribution_tags": {"team": "search"}})
    assert actual == {"ok": True}


def test_recording_serializes_to_json_roundtrip(tmp_path):
    r = ReplayRecorder(fixture_path=str(tmp_path / "rec.json"))
    r.record_llm_chat("openai", [{"role": "user", "content": "hi"}], {}, {"ok": True})
    r.record_tool_call("t", {"x": 1}, {"y": 2})
    path = r.save()
    assert path.exists()

    # Round-trip
    loaded = ReplayPlayer.from_fixture(path)
    assert loaded.play_llm_chat("openai", [{"role": "user", "content": "hi"}], {}) == {"ok": True}
    assert loaded.play_tool_call("t", {"x": 1}) == {"y": 2}


def test_assert_exhausted_catches_regression_with_fewer_calls():
    r = ReplayRecorder()
    r.record_llm_chat("o", [{"role": "user", "content": "1"}], {}, "first")
    r.record_llm_chat("o", [{"role": "user", "content": "2"}], {}, "second")

    p = ReplayPlayer(r.recording)
    p.play_llm_chat("o", [{"role": "user", "content": "1"}], {})
    # Skipped the second one — regression: agent loop now does less work.
    with pytest.raises(AssertionError, match="unconsumed"):
        p.assert_exhausted()


def test_assert_exhausted_clean_when_all_consumed():
    r = ReplayRecorder()
    r.record_llm_chat("o", [{"role": "user", "content": "1"}], {}, "x")
    p = ReplayPlayer(r.recording)
    p.play_llm_chat("o", [{"role": "user", "content": "1"}], {})
    p.assert_exhausted()  # should not raise


def test_player_stats_reflect_hits_and_remaining():
    r = ReplayRecorder()
    r.record_llm_chat("o", [{"role": "user", "content": "a"}], {}, "1")
    r.record_llm_chat("o", [{"role": "user", "content": "b"}], {}, "2")
    p = ReplayPlayer(r.recording)
    assert p.stats == {"hits": 0, "misses": 0, "remaining_entries": 2}
    p.play_llm_chat("o", [{"role": "user", "content": "a"}], {})
    assert p.stats["hits"] == 1
    assert p.stats["remaining_entries"] == 1
