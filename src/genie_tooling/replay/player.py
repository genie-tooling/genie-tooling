"""Replay player — substitutes recorded responses for live calls."""
from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .recorder import _normalize_sampling
from .types import EntryKind, Recording, RecordingEntry, call_hash

logger = logging.getLogger(__name__)


class ReplayMiss(Exception):
    """Raised when a replay run requests a call that was never recorded.

    Carries the kind + the un-found hash so test failures point at the exact
    boundary that diverged.
    """

    def __init__(self, kind: str, expected_hash: str, request_summary: str):
        self.kind = kind
        self.expected_hash = expected_hash
        self.request_summary = request_summary
        super().__init__(
            f"Replay miss for {kind}: no recorded response for hash {expected_hash[:12]}…  "
            f"Request was: {request_summary}"
        )


class ReplayPlayer:
    """Reads a recording and serves matching responses to replay-mode calls.

    Same-hash requests across the recording are served in the order they
    were recorded — so a loop that hits the same LLM twice in sequence
    gets the recorded responses in the right order.
    """

    def __init__(self, recording: Recording, strict: bool = True):
        self._recording = recording
        # Group entries by hash, preserving recording order, so duplicate
        # hashes (same prompt called twice) deal out their responses in
        # sequence.
        self._queues: Dict[str, Deque[RecordingEntry]] = defaultdict(deque)
        for e in recording.entries:
            self._queues[e.call_hash].append(e)
        self._strict = strict
        self._hits = 0
        self._misses = 0

    @classmethod
    def from_fixture(cls, path: str | Path, strict: bool = True) -> "ReplayPlayer":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        recording = Recording.from_dict(data)
        return cls(recording, strict=strict)

    def play_llm_chat(self, provider_id: str, messages: list, sampling_kwargs: Dict[str, Any]) -> Any:
        payload = {
            "provider_id": provider_id,
            "messages": messages,
            "sampling_kwargs": _normalize_sampling(sampling_kwargs),
        }
        return self._pop("llm_chat", payload)

    def play_llm_generate(self, provider_id: str, prompt: str, sampling_kwargs: Dict[str, Any]) -> Any:
        payload = {
            "provider_id": provider_id,
            "prompt": prompt,
            "sampling_kwargs": _normalize_sampling(sampling_kwargs),
        }
        return self._pop("llm_generate", payload)

    def play_tool_call(self, tool_id: str, params: Dict[str, Any]) -> Any:
        payload = {"tool_id": tool_id, "params": params}
        return self._pop("tool_call", payload)

    def _pop(self, kind: EntryKind, payload: Dict[str, Any]) -> Any:
        h = call_hash(kind, payload)
        queue = self._queues.get(h)
        if not queue:
            self._misses += 1
            summary = json.dumps(payload, default=str)[:200]
            raise ReplayMiss(kind, h, summary)
        entry = queue.popleft()
        self._hits += 1
        return entry.response

    @property
    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "remaining_entries": sum(len(q) for q in self._queues.values())}

    def assert_exhausted(self) -> None:
        """For strict regression tests: verify every recorded response was
        consumed. Mismatch means the replay run did *fewer* calls than the
        recording — also a regression."""
        remaining: List[RecordingEntry] = []
        for q in self._queues.values():
            remaining.extend(q)
        if remaining:
            kinds = sorted({e.kind for e in remaining})
            raise AssertionError(
                f"Replay exhaust check failed: {len(remaining)} recorded entries "
                f"unconsumed (kinds: {kinds}). The replay run made fewer calls than "
                f"the recording — likely a regression in the agent loop."
            )
