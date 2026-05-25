"""Phase 6C.2 — progress streaming tests."""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from genie_tooling.progress import ProgressEmitter
from genie_tooling.progress.impl.console import ConsoleProgressSinkPlugin
from genie_tooling.progress.types import ProgressEvent


class _CapturingSink:
    plugin_id = "capturing_sink"
    description = "capturing"

    def __init__(self):
        self.events: List[ProgressEvent] = []

    async def setup(self, config=None):
        pass

    async def teardown(self):
        pass

    async def emit(self, event):
        self.events.append(event)


class _FlakySink:
    plugin_id = "flaky_sink"
    description = "flaky"

    async def setup(self, config=None):
        pass

    async def teardown(self):
        pass

    async def emit(self, event):
        raise RuntimeError("kaboom")


@pytest.mark.asyncio
async def test_emit_fans_out_to_all_sinks():
    s1 = _CapturingSink()
    s2 = _CapturingSink()
    emitter = ProgressEmitter("run-1", "react_agent", sinks=[s1, s2], attribution_tags={"team": "platform"})

    await emitter.emit("started", "agent started")
    await emitter.emit("iterating", "iter 1", iteration=1)
    await emitter.emit("tool_call", "calling slack:postMessage", iteration=1, tool_id="slack:postMessage")

    assert len(s1.events) == 3
    assert len(s2.events) == 3
    assert s1.events[1].iteration == 1
    assert s1.events[2].tool_id == "slack:postMessage"
    assert s1.events[0].attribution_tags == {"team": "platform"}


@pytest.mark.asyncio
async def test_one_flaky_sink_does_not_break_others():
    good = _CapturingSink()
    bad = _FlakySink()
    emitter = ProgressEmitter("r1", "react", sinks=[bad, good])
    await emitter.emit("started", "hi")
    # Good sink still received the event even though bad one threw
    assert len(good.events) == 1


@pytest.mark.asyncio
async def test_no_sinks_no_op_returns_quickly():
    emitter = ProgressEmitter("r1", "react", sinks=[])
    assert not emitter.has_sinks
    # Should not raise
    await emitter.emit("started", "x")


@pytest.mark.asyncio
async def test_console_sink_logs_events(caplog):
    import logging
    caplog.set_level(logging.INFO, logger="genie_tooling.progress")
    sink = ConsoleProgressSinkPlugin()
    await sink.setup()
    emitter = ProgressEmitter("run-abc12345", "react", sinks=[sink])
    await emitter.emit("iterating", "iter 1 of 5", iteration=1)
    await sink.teardown()
    # Verify the log message contains our content
    matching = [r for r in caplog.records if "iter 1 of 5" in r.message]
    assert matching, f"No log record matched. Got: {[r.message for r in caplog.records]}"


@pytest.mark.asyncio
async def test_event_has_timestamp_and_run_id():
    sink = _CapturingSink()
    emitter = ProgressEmitter("run-xyz", "react", sinks=[sink])
    await emitter.emit("started", "go")
    e = sink.events[0]
    assert e.run_id == "run-xyz"
    assert e.agent_id == "react"
    assert e.timestamp > 0
    assert e.phase == "started"
