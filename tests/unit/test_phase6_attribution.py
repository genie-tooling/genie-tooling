"""Phase 6A.6 + 6C.6 — attribution_tags + audit_artifact plumbing.

These tests verify the framework-level kwargs / context fields are
plumbed through the LLM interface and the execute_tool path:

- `attribution_tags` on `genie.llm.chat` / `genie.llm.generate` lands in
  the token-usage record's `custom_tags` AND in the trace event
  payload.
- `audit_artifact` returned by a tool lands in
  `genie.execute_tool.success` trace events.
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from genie_tooling.interfaces import LLMInterface


def _make_provider_mock(usage: Dict[str, int], response_text: str = "ok"):
    provider = MagicMock()
    provider._model_name = "test-model"
    provider.chat = AsyncMock(
        return_value={
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
            "usage": usage,
        }
    )
    provider.generate = AsyncMock(
        return_value={
            "text": response_text,
            "finish_reason": "stop",
            "usage": usage,
        }
    )
    return provider


@pytest.mark.asyncio
async def test_attribution_tags_flow_into_token_recorder_on_chat():
    """`attribution_tags=` kwarg lands in TokenUsageRecord.custom_tags."""
    captured: List[Dict[str, Any]] = []
    token_mgr = MagicMock()
    token_mgr.record_usage = AsyncMock(side_effect=lambda r: captured.append(dict(r)))

    provider = _make_provider_mock(
        usage={"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
    )
    provider_mgr = MagicMock()
    provider_mgr.get_llm_provider = AsyncMock(return_value=provider)

    iface = LLMInterface(
        llm_provider_manager=provider_mgr,
        default_provider_id="x_v1",
        output_parser_manager=MagicMock(),
        token_usage_manager=token_mgr,
    )

    await iface.chat(
        messages=[{"role": "user", "content": "hi"}],
        attribution_tags={"incident": "SEV2-1234", "team": "platform"},
        session_id="s-1",
        user_id="u-1",
    )

    assert len(captured) == 1
    rec = captured[0]
    assert rec["custom_tags"] == {"incident": "SEV2-1234", "team": "platform"}
    assert rec["session_id"] == "s-1"
    assert rec["user_id"] == "u-1"
    # The provider must not have seen the framework-level kwargs.
    forwarded_kwargs = provider.chat.await_args.kwargs
    assert "attribution_tags" not in forwarded_kwargs
    assert "session_id" not in forwarded_kwargs
    assert "user_id" not in forwarded_kwargs


@pytest.mark.asyncio
async def test_attribution_tags_flow_into_token_recorder_on_generate():
    captured: List[Dict[str, Any]] = []
    token_mgr = MagicMock()
    token_mgr.record_usage = AsyncMock(side_effect=lambda r: captured.append(dict(r)))

    provider = _make_provider_mock(
        usage={"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
    )
    provider_mgr = MagicMock()
    provider_mgr.get_llm_provider = AsyncMock(return_value=provider)

    iface = LLMInterface(
        llm_provider_manager=provider_mgr,
        default_provider_id="x_v1",
        output_parser_manager=MagicMock(),
        token_usage_manager=token_mgr,
    )

    await iface.generate(
        prompt="hello",
        attribution_tags={"pr": "1234", "team": "search"},
    )

    assert len(captured) == 1
    assert captured[0]["custom_tags"] == {"pr": "1234", "team": "search"}
    assert "attribution_tags" not in provider.generate.await_args.kwargs


@pytest.mark.asyncio
async def test_attribution_tags_in_chat_trace_event():
    """trace events on chat carry attribution_tags so audit can join."""
    trace_events: List[Dict[str, Any]] = []
    tracing_mgr = MagicMock()
    tracing_mgr.trace_event = AsyncMock(
        side_effect=lambda name, data, src, cid: trace_events.append(
            {"name": name, "data": data}
        )
    )

    provider = _make_provider_mock(
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    )
    provider_mgr = MagicMock()
    provider_mgr.get_llm_provider = AsyncMock(return_value=provider)

    iface = LLMInterface(
        llm_provider_manager=provider_mgr,
        default_provider_id="x_v1",
        output_parser_manager=MagicMock(),
        tracing_manager=tracing_mgr,
    )

    await iface.chat(
        messages=[{"role": "user", "content": "hi"}],
        attribution_tags={"incident": "ABC"},
    )

    starts = [e for e in trace_events if e["name"] == "llm.chat.start"]
    successes = [e for e in trace_events if e["name"] == "llm.chat.success"]
    assert starts and starts[0]["data"]["attribution_tags"] == {"incident": "ABC"}
    assert successes and successes[0]["data"]["attribution_tags"] == {"incident": "ABC"}


@pytest.mark.asyncio
async def test_audit_artifact_in_execute_tool_trace():
    """If a tool returns dict with `audit_artifact`, it lands in the success trace event."""
    from genie_tooling.config.features import FeatureSettings
    from genie_tooling.config.models import MiddlewareConfig
    from genie_tooling.genie import Genie

    trace_events: List[Dict[str, Any]] = []

    cfg = MiddlewareConfig(
        environment="development",
        auto_enable_registered_tools=False,
        features=FeatureSettings(llm="ollama", llm_ollama_model_name="dummy"),
    )

    # Hand-rolled minimal Genie — easier than full bootstrap for this assertion.
    # We exercise the trace_event capture on the success branch directly.
    genie = Genie.__new__(Genie)
    genie._config = cfg

    # Mock tool invoker that returns a dict with audit_artifact
    async def fake_invoke(tool_identifier, params, key_provider, context, invoker_config):
        return {"ok": True, "audit_artifact": {"command": "kubectl get pods -n default"}}

    genie._tool_invoker = MagicMock()
    genie._tool_invoker.invoke = AsyncMock(side_effect=fake_invoke)
    genie._key_provider = MagicMock()
    genie._plugin_manager = MagicMock()
    genie._guardrail_manager = None
    genie._tracing_manager = None

    obs = MagicMock()
    obs.trace_event = AsyncMock(
        side_effect=lambda name, data, src, cid: trace_events.append(
            {"name": name, "data": data}
        )
    )
    genie.observability = obs

    await genie.execute_tool(
        "fake_tool",
        context={"attribution_tags": {"incident": "SEV2-1"}},
        foo="bar",
    )

    successes = [e for e in trace_events if e["name"] == "genie.execute_tool.success"]
    assert successes
    assert successes[0]["data"]["audit_artifact"] == {"command": "kubectl get pods -n default"}
    assert successes[0]["data"]["attribution_tags"] == {"incident": "SEV2-1"}
