"""F5 — verify ReplayRecorder + ReplayPlayer wire transparently through LLMInterface."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.interfaces import LLMInterface
from genie_tooling.replay import ReplayPlayer, ReplayRecorder


def _make_provider(response):
    p = MagicMock()
    p._model_name = "test-model"
    p.chat = AsyncMock(return_value=response)
    p.generate = AsyncMock(return_value=response)
    return p


def _make_iface(provider, recorder=None, player=None):
    pm = MagicMock()
    pm.get_llm_provider = AsyncMock(return_value=provider)
    return LLMInterface(
        llm_provider_manager=pm,
        default_provider_id="x",
        output_parser_manager=MagicMock(),
        replay_recorder=recorder,
        replay_player=player,
    )


@pytest.mark.asyncio()
async def test_recorder_captures_chat_call_through_interface():
    response = {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
    provider = _make_provider(response)
    recorder = ReplayRecorder(run_label="test")
    iface = _make_iface(provider, recorder=recorder)

    result = await iface.chat([{"role": "user", "content": "hello"}])
    assert result == response
    assert len(recorder.recording.entries) == 1
    assert recorder.recording.entries[0].kind == "llm_chat"


@pytest.mark.asyncio()
async def test_player_intercepts_chat_call_without_calling_provider():
    """When a player is set, the provider should NOT be called."""
    response = {"message": {"role": "assistant", "content": "live!"}, "finish_reason": "stop"}
    recorded = {"message": {"role": "assistant", "content": "recorded!"}, "finish_reason": "stop"}

    provider = _make_provider(response)
    # Pre-fill a recording
    rec = ReplayRecorder()
    rec.record_llm_chat("x", [{"role": "user", "content": "hello"}], {}, recorded)
    player = ReplayPlayer(rec.recording)

    iface = _make_iface(provider, player=player)
    result = await iface.chat([{"role": "user", "content": "hello"}])

    assert result == recorded
    provider.chat.assert_not_called()


@pytest.mark.asyncio()
async def test_player_raises_on_replay_miss():
    from genie_tooling.replay import ReplayMiss

    provider = _make_provider({})
    rec = ReplayRecorder()
    rec.record_llm_chat("x", [{"role": "user", "content": "known"}], {}, {"ok": True})
    player = ReplayPlayer(rec.recording)

    iface = _make_iface(provider, player=player)
    with pytest.raises(ReplayMiss):
        await iface.chat([{"role": "user", "content": "unknown"}])


@pytest.mark.asyncio()
async def test_recorder_captures_generate_call():
    response = {"text": "hello back", "finish_reason": "stop"}
    provider = _make_provider(response)
    recorder = ReplayRecorder()

    iface = _make_iface(provider, recorder=recorder)
    await iface.generate("hello there")

    assert len(recorder.recording.entries) == 1
    assert recorder.recording.entries[0].kind == "llm_generate"


@pytest.mark.asyncio()
async def test_record_then_replay_roundtrip():
    """Record one call, swap to a player constructed from that recording, replay."""
    response = {"message": {"role": "assistant", "content": "recorded result"}, "finish_reason": "stop"}
    provider = _make_provider(response)
    recorder = ReplayRecorder()

    iface = _make_iface(provider, recorder=recorder)
    msgs = [{"role": "user", "content": "what is 2+2"}]
    live_result = await iface.chat(msgs)
    assert live_result == response

    # Swap in a player built from the recording.
    iface.set_replay_recorder(None)
    iface.set_replay_player(ReplayPlayer(recorder.recording))
    replayed = await iface.chat(msgs)
    assert replayed == response

    # Provider was called once (the live call) and never again.
    assert provider.chat.await_count == 1
