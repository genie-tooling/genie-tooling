"""Replay recorder — captures live LLM and tool boundary calls."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .types import Recording, RecordingEntry, call_hash

logger = logging.getLogger(__name__)


class ReplayRecorder:
    """Captures LLM and tool calls during a live run and writes them to a
    fixture file. The fixture is consumed later by :class:`ReplayPlayer`.

    Typical wiring is via a small adapter layer the caller writes — at the
    LLM provider boundary, call ``recorder.record_llm_chat(...)``; at the
    tool boundary, call ``recorder.record_tool_call(...)``. See the
    replay tests for a working example.
    """

    def __init__(self, fixture_path: Optional[str] = None, run_label: Optional[str] = None):
        self._fixture_path = Path(fixture_path) if fixture_path else None
        self._recording = Recording(run_label=run_label)

    def record_llm_chat(
        self,
        provider_id: str,
        messages: list,
        sampling_kwargs: Dict[str, Any],
        response: Any,
    ) -> str:
        payload = {
            "provider_id": provider_id,
            "messages": messages,
            "sampling_kwargs": _normalize_sampling(sampling_kwargs),
        }
        h = call_hash("llm_chat", payload)
        entry = RecordingEntry(
            kind="llm_chat",
            call_hash=h,
            request=payload,
            response=response,
            sequence=len(self._recording.entries),
        )
        self._recording.entries.append(entry)
        return h

    def record_llm_generate(
        self,
        provider_id: str,
        prompt: str,
        sampling_kwargs: Dict[str, Any],
        response: Any,
    ) -> str:
        payload = {
            "provider_id": provider_id,
            "prompt": prompt,
            "sampling_kwargs": _normalize_sampling(sampling_kwargs),
        }
        h = call_hash("llm_generate", payload)
        entry = RecordingEntry(
            kind="llm_generate",
            call_hash=h,
            request=payload,
            response=response,
            sequence=len(self._recording.entries),
        )
        self._recording.entries.append(entry)
        return h

    def record_tool_call(self, tool_id: str, params: Dict[str, Any], response: Any) -> str:
        payload = {"tool_id": tool_id, "params": params}
        h = call_hash("tool_call", payload)
        entry = RecordingEntry(
            kind="tool_call",
            call_hash=h,
            request=payload,
            response=response,
            sequence=len(self._recording.entries),
        )
        self._recording.entries.append(entry)
        return h

    def save(self, path: Optional[str] = None) -> Path:
        target = Path(path) if path else self._fixture_path
        if target is None:
            raise ValueError("save() needs a fixture path (constructor or argument).")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self._recording.to_dict(), f, indent=2, default=str)
        logger.info(f"Replay recording saved to {target} ({len(self._recording.entries)} entries).")
        return target

    @property
    def recording(self) -> Recording:
        return self._recording


def _normalize_sampling(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Strip framework-level keys that aren't part of the LLM request shape
    (attribution_tags, session_id, budget_scope) so that two recordings
    with different tags but the same actual prompt collide on the same
    hash. The point of strict replay is to match the request the LLM
    saw, not the request the framework was asked to send."""
    skip = {"attribution_tags", "session_id", "user_id", "budget_scope", "response_schema"}
    return {k: v for k, v in (kwargs or {}).items() if k not in skip}
